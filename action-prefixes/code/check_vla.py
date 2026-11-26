"""Reconcile every closed-loop action with episode totals and plot paired contrasts."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
def read(name):
    with (ROOT/'data'/name).open() as f:return list(csv.DictReader(f))
def main():
    trial=read('frozen_vla_episodes.csv');cal=read('frozen_vla_calibration.csv');log=read('frozen_vla_actions.csv')
    stats=json.loads((ROOT/'data/frozen_vla_summary.json').read_text())
    assert len(trial)==72 and len(cal)==24
    by={}
    for row in log:by.setdefault((row['seed'],row['mode']),[]).append(row)
    for e in trial+cal:
        a=by[e['seed'],e['mode']];assert list(range(1,len(a)+1))==[int(row['executed_step']) for row in a]
        assert len(a)==int(e['steps']) and int(a[-1]['query'])==int(e['calls'])
        assert int(a[-1]['success'])==int(e['success'])
        assert sum(int(x['authorized_prefix'])==0 for x in a)==int(e['empty_fallbacks'])
        err=[float(x['max_landmark_error_px']) for x in a]
        assert any(v>32 for v in err)==bool(int(e['executed_violation']))
        assert any(v>32 and int(x['authorized_prefix'])>0 for x,v in zip(a,err))==bool(int(e['authorized_violation']))
        assert np.isclose(max(v/float(x['scale_px']) for x,v in zip(a,err)),float(e['max_score']))
    assert np.isclose(np.sort([float(x['max_score']) for x in cal])[22],stats['calibration_q'])
    ref=sorted((q for q in trial if q['mode']=='fixed8'),key=lambda q:q['seed'])
    ids=np.random.default_rng(50291).integers(0,24,(5000,24));rows=[]
    for name in ['demo-gate','policy-gate']:
        sel=sorted((q for q in trial if q['mode']==name),key=lambda q:q['seed']);assert [q['seed'] for q in sel]==[q['seed'] for q in ref]
        for end in ['success','authorized_violation']:
            delta=100*np.array([int(q[end])-int(b[end]) for q,b in zip(sel,ref)]);lo,hi=np.quantile(delta[ids].mean(1),[.025,.975])
            rows.append([name,end,delta.mean(),lo,hi])
    with (ROOT/'data/frozen_vla_paired.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['gate','endpoint','delta_percentage_points','ci_low','ci_high']);wr.writerows(rows)
    fig,ax=plt.subplots(figsize=(3.45,2.0))
    labels=['Success / demo','Exceedance / demo','Success / VLA','Exceedance / VLA']
    for i,(_,_,m,lo,hi) in enumerate(rows):
        c=('#17677b' if i<2 else '#a86131')
        ax.plot([lo,hi],[i,i],c=c,lw=1.4);ax.plot(m,i,'o',c=c,ms=3.7)
    ax.axvline(0,color='#565b60',lw=.8,ls=':');ax.set_yticks(range(4),labels,fontsize=7);ax.invert_yaxis()
    ax.set_xlabel('Difference from fixed eight (percentage points)',fontsize=7)
    ax.grid(axis='x',alpha=.18);ax.tick_params(axis='x',labelsize=7);ax.spines[['right','top']].set_visible(False)
    fig.tight_layout(pad=.45);fig.savefig(ROOT/'figures/frozen_vla_contrasts.pdf');plt.close(fig)
    print('action records reconciled',len(log),'paired contrast',rows)
if __name__=='__main__':main()

