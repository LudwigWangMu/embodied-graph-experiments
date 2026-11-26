"""Episode-level calibration on the public Push-T keypoint replay."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
import argparse, csv, hashlib, json, urllib.request
from pathlib import Path
import numpy as np
from prepare_pusht import read_archive
import matplotlib.pyplot as plt
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
URL='https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip'
DIGEST='63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9'
H=8
TOLS=(8,16,24,32,48,64,96)
METHODS=('Always eight','Pooled windows','Episode, flat scale','Episode, graph scale')

def window_features(k,p,t,commands):
    rel=(p[t]-k[t])/100;dist=np.linalg.norm(rel,axis=1)
    vel=(k[t]-k[t-1])/10
    centroid=(k[t].mean(0)-k[t])/100
    gate=np.exp(-dist**2/.45**2)
    past=p[t]-p[t-1]
    fs=[]; zs=[]; ys=[]
    for h in range(1,H+1):
        command=(commands[h-1]-p[t])/100
        mean=(commands[:h].mean(0)-p[t])/100
        cmd=np.broadcast_to(command,rel.shape)
        feature=np.column_stack([np.ones(9),rel,centroid,vel,past[None,:].repeat(9,axis=0)/10,
                     cmd,np.tile(mean,(9,1)),gate,gate[:,None]*cmd,gate[:,None]*rel,
                     np.tile((gate[:,None]*(cmd-rel)).mean(0),(9,1))])
        scale=[1,np.linalg.norm(command),np.linalg.norm(mean),np.linalg.norm(past)/10,
               gate.max(),gate.mean(),np.linalg.norm(vel,axis=1).max(),
               gate.max()*np.linalg.norm(command)]
        fs.append(feature);zs.append(scale);ys.append(k[t+h]-k[t])
    return fs,zs,ys

def load_records(path):
    arrays = read_archive(path) if path else np.load(ROOT/'data'/'pusht_lowdim.npz')
    k=arrays['landmarks'].astype('f8');p=arrays['pusher'].astype('f8')
    a=arrays['action'].astype('f8');ends=arrays['ends']
    rows=[];start=0
    for e,end in enumerate(ends):
        for t in range(start+1,int(end)-H,8):
            fs,zs,ys=window_features(k,p,t,a[t:t+H])
            rows.append((e,t-start,fs,zs,ys))
        start=int(end)
    return tuple(map(np.asarray,zip(*rows)))

def ridge(x,y,ridge_weight=1.):
    return np.linalg.solve(x.T@x+ridge_weight*np.eye(x.shape[1]),x.T@y)

def order_stat(values,alpha=.1):
    rank=int(np.ceil((len(values)+1)*(1-alpha)))
    return np.inf if rank>len(values) else np.partition(values,rank-1)[rank-1]

def dump(name,header,rows):
    with (ROOT/'data'/(name+'.csv')).open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(header);wr.writerows(rows)

def run(records,seed,tolerances=TOLS,detail_seed=1024,prefix=""):
    eps,anchors,feat,sc,y=records; ids=np.random.default_rng(seed).permutation(np.unique(eps))
    train,scale,cal,test=np.split(ids,[66,86,146])
    tr=np.isin(eps,train);scset=np.isin(eps,scale);ca=np.isin(eps,cal);te=np.isin(eps,test)
    err=[];scales=[]
    for h in range(H):
        x=feat[:,h];target=y[:,h]
        w=ridge(x[tr].reshape(-1,x.shape[-1]),target[tr].reshape(-1,2))
        residual=np.linalg.norm(target-x@w,axis=-1).max(1);err.append(residual)
        sw=ridge(sc[scset,h],np.log1p(residual[scset]))
        scales.append(np.maximum(1,np.expm1(np.clip(sc[:,h]@sw,0,6))))
    err=np.array(err).T;scalehat=np.array(scales).T
    flat=np.tile(np.maximum(1,np.median(err[scset],axis=0)),(len(eps),1))
    summary=[];raw=[];coverage=[]
    for method in METHODS:
        s=flat if method=='Episode, flat scale' else scalehat
        score=err/s
        if method=='Always eight':q=0.
        elif method=='Pooled windows':q=order_stat(score[ca].max(1))
        else:q=order_stat(np.array([score[eps==i].max() for i in cal]))
        bounds=s*q;endpoints=np.maximum.accumulate(bounds,axis=1)
        covered=[np.all(err[eps==i]<=bounds[eps==i]) for i in test]
        coverage.append([seed,method,q,int(sum(covered)),len(test)])
        for tol in tolerances:
            length=(endpoints<=tol).sum(1)
            fails=((np.arange(1,H+1)[None,:]<=length[:,None])&(err>tol)).any(1)
            byep=[(int(i),float(length[eps==i].mean()),bool(fails[eps==i].any()),
                   float((length[eps==i]==0).mean())) for i in test]
            nfail=sum(r[2] for r in byep);ci=binomtest(nfail,len(byep)).proportion_ci(method='wilson')
            summary.append([seed,method,tol,np.mean([r[1] for r in byep]),nfail,len(test),ci.low,ci.high,np.mean([r[3] for r in byep])])
            if seed==detail_seed:raw.extend([[i,method,tol,l,int(f),zero] for i,l,f,zero in byep])
    if seed==detail_seed:
        dump(prefix+'split',['episode','partition'],[(i,label) for label,part in zip(['fit','scale','calibration','test'],[train,scale,cal,test]) for i in part])
        dump(prefix+'anchor_errors',['episode','anchor','horizon','max_landmark_error_px','graph_scale_px','flat_scale_px'],[(int(eps[i]),int(anchors[i]),h+1,err[i,h],scalehat[i,h],flat[i,h]) for i in np.flatnonzero(te|ca) for h in range(H)])
        dump(prefix+'test_episodes',['episode','method','tolerance_px','mean_prefix','any_violation','abstention'],raw)
        print('counts',[(label,int(m.sum())) for label,m in zip(['fit','scale','cal','test'],[tr,scset,ca,te])])
    return summary,coverage

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,default=None);args=ap.parse_args()
    records=load_records(args.archive)
    rows=[];cv=[]
    for seed in range(1024,1036):
        s,c=run(records,seed);rows+=s;cv+=c
    dump('results',['seed','method','tolerance_px','episode_mean_prefix','violating_episodes','episodes','wilson_low','wilson_high','episode_mean_abstention'],rows)
    dump('coverage',['seed','method','q','simultaneously_covered','episodes'],cv)
    plt.rcParams.update({'pdf.fonttype':42,'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.65))
    styles=[('#686868','s'),('#ae5e24','^'),('#77528a','D'),('#12607a','o')]
    for method,(col,mk) in zip(METHODS,styles):
        sel=[x for x in rows if x[0]==1024 and x[1]==method]
        for ax,fn in zip(axes,[lambda x:x[3],lambda x:x[4]/x[5]]):
            ax.plot(TOLS,[fn(x) for x in sel],color=col,marker=mk,markersize=3,label=method)
    axes[0].set_ylabel('Mean authorized prefix (steps)');axes[1].set_ylabel('Episodes with a violation');axes[1].axhline(.1,ls=':',lw=1,c='#555555')
    for ax in axes:ax.set_xlabel('Landmark tolerance (scene pixels)');ax.grid(axis='y',alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,fontsize=7)
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(ROOT/'figures'/'tradeoff.pdf');plt.close(fig)
    # Show a reproducibly selected held-out trajectory window, with contemporaneous observations only.
    arrays=read_archive(args.archive) if args.archive else np.load(ROOT/'data'/'pusht_lowdim.npz')
    ends=arrays['ends'];idx=int(np.random.default_rng(1024).permutation(206)[146])
    start=0 if idx==0 else int(ends[idx-1]);t=start+max(1,(int(ends[idx])-start)//2)
    k=arrays['landmarks'][t];p=arrays['pusher'][t];a=arrays['action'][t:t+H]
    fig,ax=plt.subplots(figsize=(3.4,2.05))
    for pt in k:ax.plot([p[0],pt[0]],[p[1],pt[1]],color='#9aadb6',lw=.8,zorder=0)
    ax.scatter(*k.T,c='#145971',s=14,label='Block landmarks');ax.scatter(*p,c='#b46028',s=24,marker='s',label='Pusher')
    ax.plot(*a.T,c='#b46028',marker='.',lw=1,label='Logged targets')
    ax.set_aspect('equal');ax.invert_yaxis();ax.set_xlabel('Scene x (px)');ax.set_ylabel('Scene y (px)');ax.legend(fontsize=6,frameon=False,bbox_to_anchor=(1,.95),loc='upper left')
    fig.tight_layout();fig.savefig(ROOT/'figures'/'graph.pdf',bbox_inches='tight');plt.close(fig)
    print('primary',*[r for r in rows if r[0]==1024 and r[2]==32],sep='\n');print(cv[:4])
if __name__=='__main__':main()
