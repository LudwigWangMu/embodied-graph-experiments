"""Test passage from a planar fork to a cyclic packing in two dynamics engines."""
from experiment import ROOT, RADIUS, DT, HORIZONS, inputs, features, rollout, intervals
import csv
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
import mujoco

def scene(rng, shape):
    if shape == 'fork':
        cs, sn = np.cos(.69), np.sin(.69)
        layout = np.array([[0,0],[1,0],[1+cs,sn],[1+cs,-sn],[1+2*cs,2*sn],[1+2*cs,-2*sn]])
    else:
        a = np.sqrt(.75)
        layout = np.array([[0,0],[a,.5],[a,-.5],[2*a,0],[2*a,1],[2*a,-1],[3*a,.5],[3*a,-.5]])
    theta = rng.uniform(-np.pi,np.pi)
    u = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    xy = (layout*rng.uniform(.254,.275)+rng.uniform(-.002,.002,layout.shape))@u
    n=len(xy); ids=np.r_[0,rng.permutation(np.arange(1,n))]
    phi=theta+rng.uniform(-.28,.28)
    return xy[ids],rng.uniform(.7,1.5,n),rng.normal(0,.035,(n,2)),rng.uniform(.9,1.7)*np.array([np.cos(phi),np.sin(phi)])

@lru_cache(None)
def mjmodel(n):
    bodies=''.join(f'<body name="b{i}"><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><geom type="sphere" size="{RADIUS}" mass="1" condim="1" solref=".012 .4"/></body>' for i in range(n))
    return mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0" timestep="0.002777777777777778"/><worldbody>'+bodies+'</worldbody></mujoco>')

def mjtruth(s):
    pos,mass,vel,imp=s; model=mjmodel(len(pos));model.body_mass[1:]=mass
    data=mujoco.MjData(model);mujoco.mj_setConst(model,data)
    def run(active):
        mujoco.mj_resetData(model,data);data.qpos[:]=pos.ravel();data.qvel[:]=vel.ravel()
        if active:data.qvel[:2]+=imp/mass[0]
        out=[]
        for i in range(1,max(HORIZONS)*4+1):
            mujoco.mj_step(model,data)
            if i in [h*4 for h in HORIZONS]:out.append(data.qpos.copy().reshape(-1,2))
        return np.array(out)
    return run(True)-run(False)

def ballistic(s):
    """Event-driven frictionless paired reference with equal, nominal masses."""
    x,_,v,j=s
    aa,bb=np.triu_indices(len(x),1)
    def integrate(factual):
        p=x.copy(); q=v.copy();out=[]
        if factual:q[0]+=j
        now=0
        for horizon in HORIZONS:
            end=horizon*DT
            for _ in range(512):
                if now>=end-1e-12:break
                dr=p[bb]-p[aa];dv=q[bb]-q[aa];speed=(dv*dv).sum(1)
                dot=(dr*dv).sum(1);dis=dot*dot-speed*((dr*dr).sum(1)-(2*RADIUS)**2)
                t=np.full(len(aa),np.inf);ok=(dis>=0)&(dot< -1e-10)&(speed>1e-14)
                t[ok]=np.maximum(0,(-dot[ok]-np.sqrt(dis[ok]))/speed[ok])
                k=t.argmin()
                if t[k]>end-now:p+=(end-now)*q;now=end;break
                p+=t[k]*q;now+=t[k]
                a,b=aa[k],bb[k];normal=p[b]-p[a];normal/=np.linalg.norm(normal)
                gain=.745*np.dot(q[a]-q[b],normal)*normal
                q[a]-=gain;q[b]+=gain
            else:raise RuntimeError('event cap')
            out.append(p.copy())
        return np.array(out)
    return integrate(True)-integrate(False)

def observe(s,rng,degraded):
    x,m,v,j=s; n=len(x); observed=x.copy()
    if degraded:
        observed+=rng.normal(0,.005,x.shape)
        hidden=int(rng.integers(1,n))
        # Last sighting 80 ms earlier, with a noisier position retained for this track.
        observed[hidden]=x[hidden]-.080*v[hidden]+rng.normal(0,.012,2)
    return observed,np.ones(n),v.copy(),j.copy()

def basis(s):
    x=features(inputs([s]),'Directional')[0]
    b=ballistic(s)
    return x,b

def main():
    rng=np.random.default_rng(230929); xtrain=[]; ytrain=[]; res=[]
    for i in range(650):
        s=scene(rng,'fork'); y=rollout(s,True)-rollout(s,False)
        x,b=basis(observe(s,rng,i%2==1)); design=x[1:].reshape(-1,x.shape[-1])
        xtrain.append(design);ytrain.append(y[:,1:].transpose(1,2,0).reshape(-1,3))
        res.append((y-b)[:,1:].transpose(1,2,0).reshape(-1,3))
    a=np.concatenate(xtrain)
    def fit(ty):return np.linalg.solve(a.T@a+1e-4*np.eye(a.shape[1]),a.T@np.concatenate(ty))
    wd,wr=fit(ytrain),fit(res);rows=[]
    for shape in ['fork','packing']:
        for i in range(240):
            s=scene(rng,shape);truths={'Chipmunk':rollout(s,True)-rollout(s,False),'MuJoCo':mjtruth(s)}
            for obs in ['clean','partial']:
                x,b=basis(observe(s,rng,obs=='partial'))
                hat=(x@wd).transpose(2,0,1);correct=b+(x@wr).transpose(2,0,1)
                for engine,y in truths.items():
                    for name,yp in [('Zero',np.zeros_like(y)),('Graph',hat),('Events',b),('Corrected',correct)]:
                        errors=1000*np.linalg.norm((yp-y)[:,1:],axis=-1).mean(1)
                        rows.extend((shape,engine,obs,i,name,h,er) for h,er in zip(HORIZONS,errors))
    path=ROOT/'data'/'branch_pairs.csv'
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['shape','engine','view','scene','model','step','passive_mae_mm']);w.writerows(rows)
    summary=[]
    for shape in ['fork','packing']:
        for engine in ['Chipmunk','MuJoCo']:
            for obs in ['clean','partial']:
                def col(n):return np.array([r[-1] for r in rows if r[:3]==(shape,engine,obs) and r[4:6]==(n,16)])
                err=[col(n).mean() for n in ['Zero','Graph','Events','Corrected']]
                diff=col('Corrected')-col('Events')
                summary.append((shape,engine,obs,*err,diff.mean(),*intervals(diff)))
    with (ROOT/'data'/'branch_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['shape','engine','view','zero','graph','events','corrected','paired_delta','ci_low','ci_high']);w.writerows(summary)
    fig,axs=plt.subplots(1,2,figsize=(7.0,2.48),sharey=True)
    for ax,engine in zip(axs,['Chipmunk','MuJoCo']):
        selected=[r for r in summary if r[1]==engine]
        for k,(name,color,mk) in enumerate([('Graph','#5b6374','^'),('Events','#b46b32','s'),('Corrected','#196779','o')]):
            ax.plot(range(4),[r[4+k] for r in selected],label=name,color=color,marker=mk,ms=4)
        ax.set_title(engine,fontsize=9);ax.set_xticks(range(4),['Fork\nclean','Fork\npartial','Packing\nclean','Packing\npartial'],fontsize=7)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Passive error (mm)',fontsize=8)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=8)
    fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(ROOT/'figures'/'branch_transfer.pdf');plt.close(fig)
    print(*summary,sep='\n')
if __name__=='__main__':main()

