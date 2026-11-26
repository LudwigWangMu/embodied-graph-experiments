"""Vary the number of calibration trajectories while fixing the fitted models and test set."""
from pathlib import Path
import csv
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SIZES=(8,12,20,40,60)
RISKS=(.05,.10,.20)

def read(name):
    with (ROOT/"data"/name).open(newline="") as f:
        return list(csv.DictReader(f))

def save(name,fields,rows):
    with (ROOT/"data"/name).open("w",newline="") as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(rows)

def corpus(prefix,unit,tol):
    split=read(prefix+"split.csv")
    cal=sorted(int(r["episode"]) for r in split if r["partition"]=="calibration")
    test=sorted(int(r["episode"]) for r in split if r["partition"]=="test")
    raw=read(prefix+"anchor_errors.csv")
    raw.sort(key=lambda row: tuple(int(row[k]) for k in ("episode","anchor","horizon")))
    ep=np.array([int(r["episode"]) for r in raw]).reshape(-1,8)[:,0]
    anchor=np.array([int(r["anchor"]) for r in raw]).reshape(-1,8)[:,0]
    e=np.array([float(r["max_landmark_error_"+unit]) for r in raw]).reshape(-1,8)
    s=np.array([float(r["graph_scale_"+unit]) for r in raw]).reshape(-1,8)
    score=np.array([(e[ep==i]/s[ep==i]).max() for i in cal])
    use=np.isin(ep,test); t,st,err=ep[use],s[use],e[use]
    _,indices,counts=np.unique(t,return_index=True,return_counts=True)
    assert len(cal)==len(test)==60 and np.all(s>0)
    @lru_cache(None)
    def evaluate(q):
        length=(np.maximum.accumulate(st,axis=1)*q<=tol).sum(1)
        violation=((np.arange(1,9)<=length[:,None]) & (err>tol)).any(1)
        return (np.mean(np.add.reduceat(length,indices)/counts),
                np.logical_or.reduceat(violation,indices).sum(),
                np.mean(np.add.reduceat((length==0).astype(float),indices)/counts))
    rng=np.random.default_rng(223158); rows=[]
    for repetition in range(160):
        ordering=rng.permutation(score)
        for size in SIZES:
            ordered=np.sort(ordering[:size])
            for alpha in RISKS:
                k=int(np.ceil((size+1)*(1-alpha)))
                q=ordered[k-1] if k<=size else np.inf
                rows.append((prefix or "pusht",repetition,size,alpha,k,q,*evaluate(q)))
    q=np.sort(score)[54]
    # Select the first partially released window in lexicographic episode/frame order.
    for i in np.flatnonzero(use):
        length=int((np.maximum.accumulate(q*s[i])<=tol).sum())
        if 0<length<8:
            case=[(prefix or "pusht",ep[i],anchor[i],h+1,s[i,h],q*s[i,h],e[i,h],length) for h in range(8)]
            return rows,case
    raise ValueError("No partial window")

def main():
    allrows=[];examples=[]
    for prefix,unit,tol in [("","px",32),("physical_","mm",2)]:
        a,b=corpus(prefix,unit,tol);allrows.extend(a);examples.extend(b)
    save("calibration_subsets.csv",["corpus","draw","size","alpha","rank","q","mean_prefix","violating_episodes","zero_fraction"],allrows)
    save("worked_windows.csv",["corpus","episode","anchor","horizon","scale","bound","error","authorized_length"],examples)
    summary=[]
    for data in ["pusht","physical_"]:
        for n in SIZES:
            for alpha in RISKS:
                rr=np.array([x[6:9] for x in allrows if x[0]==data and x[2:4]==(n,alpha)])
                summary.append((data,n,alpha,*np.quantile(rr[:,0],[.1,.5,.9]),*np.quantile(rr[:,1],[.1,.5,.9]),np.median(rr[:,2])))
    save("calibration_subset_summary.csv",["corpus","size","alpha","prefix_p10","prefix_median","prefix_p90","fails_p10","fails_median","fails_p90","zero_median"],summary)
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.65),sharey=True)
    for ax,data,title in zip(axes,["pusht","physical_"],["Push-T at 32 pixels","Physical rectangle at 2 mm"]):
        for alpha,col,mk in zip(RISKS,["#724d7c","#156982","#9b4e22"],["s","o","^"]):
            rs=np.array([r[3:6] for r in summary if r[0]==data and r[2]==alpha])
            ax.plot(SIZES,rs[:,1],color=col,marker=mk,ms=3,label=f"{alpha:.0%} target")
            ax.fill_between(SIZES,rs[:,0],rs[:,2],color=col,alpha=.10)
        ax.set_title(title,fontsize=9);ax.set_xticks(SIZES);ax.tick_params(labelsize=8)
        ax.set_xlabel("Calibration episodes");ax.spines[["right","top"]].set_visible(False);ax.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Mean released steps");axes[1].legend(fontsize=7,loc="lower right",frameon=False)
    fig.tight_layout();fig.savefig(ROOT/"figures/calibration_budget.pdf",bbox_inches="tight");plt.close(fig)
if __name__=="__main__":main()

