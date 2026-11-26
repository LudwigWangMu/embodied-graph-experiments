"""Hold object count and sampled disturbances constant across three six-disc layouts."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from experiment import ROOT, HORIZONS, rollout
from branch_transfer import scene, observe, basis, mjtruth

def fit_residual():
    rng = np.random.default_rng(230929)
    xrows, yrows = [], []
    for index in range(650):
        state = scene(rng, "fork")
        truth = rollout(state, True) - rollout(state, False)
        x, event = basis(observe(state, rng, index % 2 == 1))
        xrows.append(x[1:].reshape(-1, x.shape[-1]))
        yrows.append((truth-event)[:,1:].transpose(1,2,0).reshape(-1,3))
    x = np.concatenate(xrows)
    return np.linalg.solve(x.T @ x + 1e-4*np.eye(x.shape[1]), x.T @ np.concatenate(yrows))

def layouts():
    cs, sn = np.cos(.69), np.sin(.69)
    a = np.sqrt(.75)
    return {"Line": [[i,0] for i in range(6)],
            "Fork": [[0,0],[1,0],[1+cs,sn],[1+cs,-sn],[1+2*cs,2*sn],[1+2*cs,-2*sn]],
            "Packing": [[0,0],[a,.5],[a,-.5],[2*a,0],[2*a,1],[2*a,-1]]}

def write(name, fields, rows):
    with (ROOT/"data"/name).open("w",newline="") as f:
        w=csv.writer(f); w.writerow(fields);w.writerows(rows)

def main():
    weights=fit_residual()
    rng=np.random.default_rng(763021)
    rows=[]; gaps=[]
    for key in range(200):
        angle=rng.uniform(-np.pi,np.pi)
        rot=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
        spacing=rng.uniform(.254,.275)
        jitter=rng.uniform(-.002,.002,(6,2))
        ids=np.r_[0,rng.permutation(np.arange(1,6))]
        mass=rng.uniform(.7,1.5,6)
        velocity=rng.normal(0,.035,(6,2))
        phi=angle+rng.uniform(-.28,.28)
        impulse=rng.uniform(.9,1.7)*np.array([np.cos(phi),np.sin(phi)])
        for name,layout in layouts().items():
            x=((np.array(layout)*spacing+jitter)@rot)[ids]
            nominal=(x,np.ones(6),velocity,impulse)
            design,event=basis(nominal)
            residual=(design @ weights).transpose(2,0,1)
            equal_truth={}
            for masses,m in [("Unit",np.ones(6)),("Hidden",mass)]:
                state=(x,m,velocity,impulse)
                for engine,truth in [("Chipmunk",rollout(state,True)-rollout(state,False)),("MuJoCo",mjtruth(state))]:
                    if masses=="Unit":
                        equal_truth[engine]=truth
                    else:
                        dist=np.linalg.norm((truth-equal_truth[engine])[:,1:],axis=2).mean(1)*500
                        gaps.extend((key,name,engine,h,bound) for h,bound in zip(HORIZONS,dist))
                    for amp in [0,.5,1]:
                        error=np.linalg.norm((event+amp*residual-truth)[:,1:],axis=2).mean(1)*1000
                        rows.extend((key,name,masses,engine,h,amp,loss) for h,loss in zip(HORIZONS,error))
    write("mass_completion_pairs.csv",["draw","layout","engine","horizon","lower_bound_mm"],gaps)
    write("matched_geometry.csv",["draw","layout","mass","engine","horizon","residual_gain","mae_mm"],rows)
    rng=np.random.default_rng(9332)
    ids=rng.integers(0,200,(4000,200));results=[]
    for name in layouts():
        for mass in ["Unit","Hidden"]:
            for engine in ["Chipmunk","MuJoCo"]:
                vals=np.array([[r[-1] for r in rows if r[1:6]==(name,mass,engine,16,a)] for a in [0,.5,1]])
                delta=vals[2]-vals[0];lo,hi=np.quantile(delta[ids].mean(1),[.025,.975])
                results.append((name,mass,engine,*vals.mean(1),delta.mean(),lo,hi,(delta<0).mean()))
    write("matched_geometry_summary.csv",["layout","mass","engine","event","half","full","difference","lower","upper","fraction_improved"],results)
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.65),sharex=True,sharey=True)
    for ax,m in zip(axes,["Unit","Hidden"]):
        for ix,(engine,color,mk) in enumerate([("Chipmunk","#146782","o"),("MuJoCo","#9e461e","s")]):
            rr=[r for r in results if r[1:3]==(m,engine)]
            y=np.arange(3)+(.11 if ix else -.11)
            means=np.array([r[6] for r in rr])
            ax.errorbar(means,y,xerr=[means-[r[7] for r in rr],[r[8] for r in rr]-means],fmt=mk,color=color,capsize=2,label=engine,ms=4)
        ax.axvline(0,color="#747474",lw=.9,ls="--");ax.set_title(m+" masses",fontsize=9)
        ax.set_yticks(range(3),list(layouts()));ax.set_xlabel("Corrected minus event error (mm)")
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis();axes[1].legend(fontsize=7,frameon=False)
    fig.tight_layout();fig.savefig(ROOT/"figures/matched_geometry.pdf",bbox_inches="tight");plt.close(fig)
if __name__=="__main__":
    main()

