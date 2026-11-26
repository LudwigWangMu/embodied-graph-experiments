"""Additional competent image-policy control using the public 175k-step Push-T weights."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import OBS_IMAGES,OBS_STATE
from frozen_vla import ROOT,Forecaster,play
from experiment import order_stat
REV='84a7c23178445c6bbf7e1a884ff497017910f653'
class ImagePolicy:
    def __init__(self,folder):
        cfg=PreTrainedConfig.from_pretrained(str(folder));cfg.device='cuda'
        model=DiffusionPolicy(cfg)
        saved=load_file(str(folder/'model.safetensors'))
        # This older LeRobot checkpoint embeds eight normalization buffers in the policy.
        weights={k:v for k,v in saved.items() if k.startswith('diffusion.')}
        assert len(weights)==len(saved)-8;model.load_state_dict(weights,strict=True)
        self.model=model.eval().to('cuda').diffusion
        self.buffers={k:v.cuda() for k,v in saved.items() if k not in weights}
    @torch.inference_mode()
    def propose_tracks(self,tracks):
        im=torch.from_numpy(np.stack([[e['prev_ob']['pixels'],e['ob']['pixels']] for e in tracks])).permute(0,1,4,2,3).cuda().float()/255
        pos=torch.from_numpy(np.stack([[e['prev_ob']['agent_pos'],e['ob']['agent_pos']] for e in tracks])).float().cuda()
        b=self.buffers;i='normalize_inputs.buffer_observation_image.';s='normalize_inputs.buffer_observation_state.';a='unnormalize_outputs.buffer_action.'
        im=(im-b[i+'mean'])/(b[i+'std']+1e-8);pos=2*(pos-b[s+'min'])/(b[s+'max']-b[s+'min']+1e-8)-1
        gc=self.model._prepare_global_conditioning({OBS_IMAGES:im.unsqueeze(2),OBS_STATE:pos})
        gens=[torch.Generator(device='cuda').manual_seed(e['seed']*1000+e['steps']) for e in tracks]
        prior=torch.stack([torch.randn(16,2,generator=g,device='cuda') for g in gens])
        samples=self.model.conditional_sample(len(tracks),global_cond=gc,generator=gens,noise=prior)
        actions=(samples[:,1:9]+1)/2*(b[a+'max']-b[a+'min'])+b[a+'min']
        assert torch.isfinite(actions).all()
        return np.clip(actions.cpu().numpy(),0,512)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',type=Path);ap.add_argument('--batch',type=int,default=12);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if args.policy is None:
        from huggingface_hub import snapshot_download
        args.policy=Path(snapshot_download('lerobot/diffusion_pusht',revision=REV,allow_patterns=['config.json','model.safetensors']))
    policy=ImagePolicy(args.policy);fore=Forecaster();body=np.load(ROOT/'data/pusht_pose.npz')['body']
    if args.smoke:
        print(play(policy,fore,body,[44000],'fixed8',0,8));return
    header=['seed','mode','success','max_coverage','steps','calls','authorized_steps','empty_fallbacks','authorized_violation','executed_violation','max_score']
    def run(seeds,mode,q):
        rows=[]
        for j in range(0,len(seeds),args.batch):
            rows+=play(policy,fore,body,seeds[j:j+args.batch],mode,q,120)
            print(mode,len(rows),flush=True)
        return rows
    def write(path,rows):
        with (ROOT/'data'/path).open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(header);wr.writerows(rows)
    cal=run(list(range(41000,41024)),'fixed8',0);write('visual_diffusion_calibration.csv',cal)
    q=float(order_stat(np.array([v[-1] for v in cal])))
    allrows=[];summary={}
    for mode in ['fixed8','policy-gate']:
        rows=run(list(range(42000,42024)),mode,q);allrows+=rows;write('visual_diffusion_episodes.csv',allrows)
        summary[mode]=dict(success=sum(v[2] for v in rows),peak_overlap=np.mean([v[3] for v in rows]),violating=sum(v[8] for v in rows),all_violating=sum(v[9] for v in rows),steps=sum(v[4] for v in rows),calls=sum(v[5] for v in rows),fallback=sum(v[7] for v in rows))
    report=dict(model='lerobot/diffusion_pusht',revision=REV,ddpm_steps=policy.model.num_inference_steps,calibration_q=q,results=summary)
    (ROOT/'data/visual_diffusion_summary.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()

