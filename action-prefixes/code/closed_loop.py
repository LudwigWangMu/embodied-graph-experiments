"""Closed-loop Push-T evaluation with a frozen nearest-demonstration chunk policy."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial import cKDTree
from gym_pusht.envs.pusht import PushTEnv
from experiment import H, ROOT,load_records,ridge,order_stat
import matplotlib.pyplot as plt

def rotation(a):
    c,s=np.cos(a),np.sin(a)
    return np.stack([c,-s,s,c],axis=-1).reshape((*np.shape(a),2,2))

class Forecaster:
    def __init__(self):
        ep,_,f,sc,y=load_records(None);ids=np.random.default_rng(1024).permutation(206)
        train,scale,cal,_=np.split(ids,[66,86,146]);self.fit_ids=train
        t=np.isin(ep,train);s=np.isin(ep,scale)
        self.w=[];self.v=[];scores=[]
        for h in range(H):
            x=f[:,h];w=ridge(x[t].reshape(-1,x.shape[-1]),y[t,h].reshape(-1,2))
            err=np.linalg.norm(y[:,h]-x@w,axis=-1).max(1)
            v=ridge(sc[s,h],np.log1p(err[s]))
            scl=np.maximum(1,np.expm1(np.clip(sc[:,h]@v,0,6)))
            self.w.append(w);self.v.append(v);scores.append(err/scl)
        self.w=np.array(self.w);self.v=np.array(self.v)
        score=np.array(scores).T
        self.q=float(order_stat(np.array([score[ep==i].max() for i in cal])))
    def forecast(self,k,prev,p,last,actions):
        rel=(p-k)/100;d=np.linalg.norm(rel,axis=1);g=np.exp(-d**2/.45**2)
        vel=(k-prev)/10;cent=(k.mean(0)-k)/100
        past=(p-last)/10;cmd=(actions-p)/100
        mean=np.cumsum(cmd,axis=0)/np.arange(1,H+1)[:,None]
        common=np.column_stack([np.ones(9),rel,cent,vel,np.tile(past,(9,1))])
        tile=lambda z:np.broadcast_to(z,(H,9,z.shape[-1]))
        near=(g[None,:,None]*(cmd[:,None,:]-rel[None,:,:])).mean(1)
        x=np.concatenate([tile(common),tile(cmd[:,None,:]),tile(mean[:,None,:]),tile(g[:,None]),g[None,:,None]*cmd[:,None,:],tile(g[:,None]*rel),tile(near[:,None,:])],axis=-1)
        cc=np.linalg.norm(cmd,axis=1)
        scal=np.column_stack([np.ones(H),cc,np.linalg.norm(mean,axis=1),np.full(H,np.linalg.norm(past)),np.full(H,g.max()),np.full(H,g.mean()),np.full(H,np.linalg.norm(vel,axis=1).max()),g.max()*cc])
        return k+x@self.w,np.maximum(1,np.expm1(np.clip((scal*self.v).sum(1),0,6)))

def policy_keys(state):
    p,c,a=state[:,:2],state[:,2:4],state[:,4]
    local=np.einsum('ni,nij->nj',p-c,rotation(a))
    return np.column_stack([local,c*.75,64*np.sin(a),64*np.cos(a)])

class RetrievalPolicy:
    def __init__(self,fit):
        d=np.load(ROOT/'data/pusht_lowdim.npz');p=np.load(ROOT/'data/pusht_pose.npz')
        self.body=p['body'];pose=p['pose'];self.actions=d['action'];self.pose=pose
        ends=d['ends'].astype(int); starts=np.r_[0,ends[:-1]]
        self.ix=np.concatenate([np.arange(starts[i],ends[i]-H) for i in fit])
        self.tree=cKDTree(policy_keys(pose[self.ix]))
    def __call__(self,state):
        j=self.ix[self.tree.query(policy_keys(state[None])[0])[1]]
        src=self.pose[j]; rel=(self.actions[j:j+H]-src[2:4])@rotation(src[4])
        return np.clip(rel@rotation(state[4]).T+state[2:4],0,512)

def landmarks(state,body):
    return body@rotation(state[4]).T+state[2:4]
def choose(mode,q,s):
    if mode.startswith('fixed'):return int(mode[5:])
    return int(np.count_nonzero(np.maximum.accumulate(q*s)<=32))

def episode(env,policy,model,seed,mode,q,budget=160):
    state,_=env.reset(seed=seed);k=landmarks(state,policy.body);prev=k.copy();last=state[:2].copy()
    steps=calls=allowed=empty=bad=bad_all=0;peak=env._get_coverage();mx=0.;done=False
    while steps<budget and not done:
        p=state[:2];actions=policy(state)
        forecast,scale=model.forecast(k,prev,p,last,actions)
        requested=choose(mode,q,scale);calls+=1
        if requested==0:empty+=1
        for h in range(min(max(1,requested),budget-steps)):
            prev=k;last=state[:2]
            state,_,done,_,info=env.step(actions[h].astype('f4'))
            k=landmarks(state,policy.body);steps+=1;peak=max(peak,info['coverage'])
            e=float(np.linalg.norm(k-forecast[h],axis=1).max())
            mx=max(mx,e/scale[h]);bad_all+=int(e>32)
            if requested>0:allowed+=1;bad+=int(e>32)
            if done:break
    return [seed,mode,int(done),peak,steps,calls,allowed,empty,int(bad>0),int(bad_all>0),mx]

def main():
    model=Forecaster();policy=RetrievalPolicy(model.fit_ids);env=PushTEnv(obs_type='state')
    cal=[episode(env,policy,model,seed,'fixed8',0) for seed in range(31000,31048)]
    q=float(order_stat(np.array([r[-1] for r in cal])))
    rows=[]
    for seed in range(32000,32060):
        for name in ['fixed1','fixed4','fixed8','demo-gate','policy-gate']:
            rows.append(episode(env,policy,model,seed,name,model.q if name=='demo-gate' else q))
    env.close()
    header=['seed','mode','success','max_coverage','steps','calls','authorized_steps','empty_fallbacks','authorized_violation','executed_violation','max_score']
    for file,rr in [('closed_loop_calibration',cal),('closed_loop_episodes',rows)]:
        with (ROOT/'data'/(file+'.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(header);writer.writerows(rr)
    out={'q_demonstrations':model.q,'q_fixed_policy':q}
    for name in ['fixed1','fixed4','fixed8','demo-gate','policy-gate']:
        sel=[r for r in rows if r[1]==name];out[name]=dict(success=sum(r[2] for r in sel),peak=np.mean([r[3] for r in sel]),total_steps=sum(r[4] for r in sel),calls=sum(r[5] for r in sel),authorized_steps=sum(r[6] for r in sel),fallback=sum(r[7] for r in sel),violating=sum(r[8] for r in sel),all_executed_violating=sum(r[9] for r in sel))
    (ROOT/'data/closed_loop_summary.json').write_text(json.dumps(out,indent=2))
    print(out)
    fig,ax=plt.subplots(1,2,figsize=(6.9,2.35));names=list(out)[2:];labels=['1','4','8','Demo','Policy']
    colors=['#88949a']*3+['#b66d35','#185d78']
    ax[0].bar(labels,[out[n]['peak'] for n in names],color=colors);ax[0].set_ylabel('Mean peak overlap')
    ax[1].bar(labels,[100*out[n]['calls']/out[n]['total_steps'] for n in names],color=colors);ax[1].set_ylabel('Calls per 100 steps')
    for axis in ax:axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.17);axis.tick_params(labelsize=8)
    fig.tight_layout();fig.savefig(ROOT/'figures/closed_loop.pdf');plt.close(fig)
if __name__=='__main__':main()

