"""Paired closed-loop evaluation of an immutable public Push-T SmolVLA checkpoint."""
import argparse,csv,json,hashlib,os
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,load_model
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
from closed_loop import ROOT,Forecaster,landmarks,choose,order_stat,PushTEnv,H
MODEL='smy1729/smolvlapusht';REV='c9525f2afb45503a8319e353aa33b597efe48359'
TEXT='HuggingFaceTB/SmolVLM2-500M-Video-Instruct';TEXT_REV='4cde9d208e19519c36917aa170fffc7e381aa0ad'
TASK='Push the T-shaped block onto the T-shaped target.\n'
def sources(policy,backbone):
    from huggingface_hub import snapshot_download
    if policy is None:policy=Path(snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors']))
    if backbone is None:backbone=Path(snapshot_download(TEXT,revision=TEXT_REV,allow_patterns=['*.json','*.txt']))
    return Path(policy),Path(backbone)
class ChunkPolicy:
    def __init__(self,p,b,device):
        self.device=device
        cfg=PreTrainedConfig.from_pretrained(str(p));cfg.vlm_model_name=str(b);cfg.device=device
        self.model=SmolVLAPolicy(cfg)
        load_model(self.model,str(p/'model.safetensors'),strict=True)
        self.model.to(device).eval()
        self.tokenizer=AutoTokenizer.from_pretrained(str(b));self.tokenizer.padding_side='right'
        self.pre=load_file(str(next(p.glob('policy_preprocessor*.safetensors'))))
        self.post=load_file(str(next(p.glob('policy_postprocessor*.safetensors'))))
    @torch.inference_mode()
    def propose(self,obs,pushers,noise):
        text=self.tokenizer([TASK]*len(obs),padding='longest',max_length=48,truncation=True,return_tensors='pt')
        image=torch.tensor(np.stack(obs)).permute(0,3,1,2).float()/255
        pos=torch.tensor(np.stack(pushers),dtype=torch.float32)
        pos=(pos-self.pre['observation.state.mean'])/(self.pre['observation.state.std']+1e-8)
        batch={'observation.image':image,'observation.state':pos,OBS_LANGUAGE_TOKENS:text.input_ids,OBS_LANGUAGE_ATTENTION_MASK:text.attention_mask.bool()}
        batch={k:v.to(self.device) for k,v in batch.items()}
        y=self.model.predict_action_chunk(batch,noise=torch.stack(noise).to(self.device)).float().cpu()
        a=y*self.post['action.std']+self.post['action.mean'];assert torch.isfinite(a).all()
        return np.clip(a[:,:H].numpy(),0,512)

def state(ob,info):return np.r_[ob['agent_pos'],info['block_pose']]
def play(policy,forecast,body,seeds,mode,q,limit,trace=None):
    tracks=[]
    for seed in seeds:
        env=PushTEnv(obs_type='pixels_agent_pos');ob,info=env.reset(seed=seed);s=state(ob,info);k=landmarks(s,body)
        tracks.append(dict(env=env,seed=seed,s=s,k=k,prev=k.copy(),last=s[:2].copy(),ob=ob,prev_ob=ob,steps=0,calls=0,authorized=0,fallback=0,bad=0,bad_all=0,mx=0.,peak=float(env._get_coverage()),done=False))
    while active:=[e for e in tracks if not e['done'] and e['steps']<limit]:
        noise=[]
        for e in active:
            # Same state-independent stream at a given reset seed and simulator step.
            gen=torch.Generator().manual_seed(e['seed']*1000+e['steps'])
            noise.append(torch.randn((50,32),generator=gen))
        if hasattr(policy,'propose_tracks'):
            chunks=policy.propose_tracks(active)
        else:
            chunks=policy.propose([e['ob']['pixels'] for e in active],[e['s'][:2] for e in active],noise)
        for e,actions in zip(active,chunks):
            yh,scale=forecast.forecast(e['k'],e['prev'],e['s'][:2],e['last'],actions)
            n=choose(mode,q,scale);e['calls']+=1;e['fallback']+=int(n==0)
            for h in range(min(max(1,n),limit-e['steps'])):
                e['prev']=e['k'];e['last']=e['s'][:2]
                ob,_,term,_,info=e['env'].step(actions[h].astype('f4'))
                e['prev_ob']=e['ob'];e['ob']=ob;e['s']=state(ob,info);e['k']=landmarks(e['s'],body);e['steps']+=1
                e['peak']=max(e['peak'],float(info['coverage']));e['done']=bool(term)
                err=np.linalg.norm(e['k']-yh[h],axis=-1).max()
                if trace is not None:
                    trace.append([e['seed'],mode,e['calls'],e['steps'],h+1,n,float(actions[h,0]),float(actions[h,1]),float(err),float(scale[h]),float(info['coverage']),int(term)])
                e['mx']=max(e['mx'],float(err/scale[h]));e['bad_all']+=int(err>32)
                if n:e['authorized']+=1;e['bad']+=int(err>32)
                if term:break
    for e in tracks:e['env'].close()
    return [[e['seed'],mode,int(e['done']),e['peak'],e['steps'],e['calls'],e['authorized'],e['fallback'],int(e['bad']>0),int(e['bad_all']>0),e['mx']] for e in tracks]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',type=Path);ap.add_argument('--backbone',type=Path);ap.add_argument('--device',default='cuda');ap.add_argument('--batch',type=int,default=6);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    p,b=sources(args.policy,args.backbone);policy=ChunkPolicy(p,b,args.device);fore=Forecaster();body=np.load(ROOT/'data/pusht_pose.npz')['body']
    if args.smoke:
        print(play(policy,fore,body,[44999],'fixed8',0,8));return
    header=['seed','mode','success','max_coverage','steps','calls','authorized_steps','empty_fallbacks','authorized_violation','executed_violation','max_score']
    trace=[]
    def suite(seeds,mode,q):
        result=[]
        for idx in range(0,len(seeds),args.batch):
            result+=play(policy,fore,body,seeds[idx:idx+args.batch],mode,q,120,trace)
            print(mode,len(result),flush=True)
        return result
    def save(name,rows):
        with (ROOT/'data'/name).open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(header);wr.writerows(rows)
    cal=suite(list(range(41000,41024)),'fixed8',0)
    save('frozen_vla_calibration.csv',cal);q=float(order_stat(np.array([r[-1] for r in cal])))
    rows=[];summary={}
    for name in ['fixed8','demo-gate','policy-gate']:
        rr=suite(list(range(42000,42024)),name,fore.q if name=='demo-gate' else q)
        rows+=rr;save('frozen_vla_episodes.csv',rows)
        summary[name]=dict(success=sum(v[2] for v in rr),peak_overlap=np.mean([v[3] for v in rr]),steps=sum(v[4] for v in rr),calls=sum(v[5] for v in rr),authorized=sum(v[6] for v in rr),fallback=sum(v[7] for v in rr),violations=sum(v[8] for v in rr),all_violations=sum(v[9] for v in rr))
    with (ROOT/'data/frozen_vla_actions.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['seed','mode','query','executed_step','chunk_step','authorized_prefix','target_x','target_y','max_landmark_error_px','scale_px','goal_overlap','success']);wr.writerows(trace)
    data=dict(model=MODEL,revision=REV,backbone=TEXT,backbone_revision=TEXT_REV,device=torch.cuda.get_device_name() if args.device=='cuda' else args.device,torch=torch.__version__,study=summary,calibration_q=q,demo_q=fore.q,batch=args.batch,budget=120)
    (ROOT/'data/frozen_vla_summary.json').write_text(json.dumps(data,indent=2))
    print(data)
if __name__=='__main__':main()

