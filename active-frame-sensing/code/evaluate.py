"""Integrate risks exactly, including compound sensor noise, and audit independent solvers."""
from pathlib import Path
import csv,json
import numpy as np
from decision import Tree,tied_min,signature_floor,exhaustive_check,first_features
from make_tasks import ROOT

def write(name,fields,rows):
    with (ROOT/'data'/name).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(rows)
def main():
    d=np.load(ROOT/'data/tasks.npz');sigs,costs=d['signature'],d['cost']
    with (ROOT/'data/tasks.csv').open() as f:meta=list(csv.DictReader(f))
    features=[];targets=[];rows=[];rob=[];floors=[];sample=None;gap=-1;maxerr=0
    rng=np.random.default_rng(743)
    for i,(s,c,info) in enumerate(zip(sigs,costs,meta)):
        base=Tree(s,c);features.append(first_features(s,c));targets.append(base.q)
        if info['split'] in ['fit','validation']:continue
        prefix=(i,info['split'],info['source'],info['pair'])
        floor=signature_floor(s,c);nclass=len(np.unique(s.T,axis=0))
        floors.append((*prefix,nclass,floor))
        for noise in [0,.1,.25]:
            tr=base if noise==.1 else Tree(s,c,noise)
            dist=tied_min(c.mean(0))
            rows.append((*prefix,noise,'blind',c.mean(0)@dist,(c==3).mean(0)@dist,(c==0).mean(0)@dist))
            one=(tr.one@c);last=tied_min(one);risk=one.min(-1).sum(1);best=tied_min(risk)
            bump=(last*(tr.one@(c==3).astype(float))).sum((1,2))@best
            prog=(last*(tr.one@(c==0).astype(float))).sum((1,2))@best
            rows.append((*prefix,noise,'one',risk@best,bump,prog))
            vals={}
            for method,(first,second) in tr.policies().items():
                val=tr.score(first,second);vals[method]=val[0]
                rows.append((*prefix,noise,method,*val))
            assert floor<=vals['adaptive']+1e-9 and vals['adaptive']<=min(vals['fixed-pair'],vals['greedy'],vals['entropy'],risk.min())+1e-9
            if noise==.1 and info['split']=='city' and vals['entropy']-vals['adaptive']>gap+1e-9:
                gap=vals['entropy']-vals['adaptive'];sample=i
        common=Tree(s,c,.1,shared=True)
        for method,z,world in [('independent-matched',base,base),('independent-on-common',base,common),('common-matched',common,common)]:
            rob.append((*prefix,method,*z.score(z.first,z.second,world)))
        if len(floors)<=16:
            explicit=exhaustive_check(s,c,.1);perm=rng.permutation(8);pa=rng.permutation(4);pu=rng.permutation(12)
            other=Tree(s[pu][:,perm],c[perm][:,pa]);maxerr=max(maxerr,np.abs(explicit-base.q).max(),np.abs(other.q-base.q[pu]).max())
    assert maxerr<1e-10
    np.savez_compressed(ROOT/'data/graph_training.npz',features=np.array(features,dtype='f4'),targets=np.array(targets,dtype='f4'))
    fields=['index','split','source','pair'];metrics=['risk','bump','progress']
    write('policy_risks.csv',fields+['noise','method']+metrics,rows)
    write('sensor_dependence.csv',fields+['model']+metrics,rob)
    write('ambiguity.csv',fields+['distinct_signatures','floor'],floors)
    (ROOT/'data/checks.json').write_text(json.dumps(dict(max_tree_and_permutation_error=float(maxerr),explicit_cases=16,example=int(sample)),indent=2))
if __name__=='__main__':main()

