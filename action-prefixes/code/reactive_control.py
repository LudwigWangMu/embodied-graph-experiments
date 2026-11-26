"""Fit a post-observation residual trigger at a comparable query budget."""
import csv,json
import numpy as np
from closed_loop import ROOT,Forecaster,RetrievalPolicy,PushTEnv,landmarks,episode,H
def reactive(env,pol,fore,seed,threshold):
    s,_=env.reset(seed=seed);k=landmarks(s,pol.body);prev=k.copy();last=s[:2].copy()
    steps=calls=violation=0;done=False;peak=env._get_coverage()
    while not done and steps<160:
        actions=pol(s);y,_=fore.forecast(k,prev,s[:2],last,actions);calls+=1
        for h in range(min(H,160-steps)):
            prev=k;last=s[:2];s,_,done,_,info=env.step(actions[h].astype('f4'));k=landmarks(s,pol.body)
            steps+=1;peak=max(peak,info['coverage']);err=float(np.linalg.norm(k-y[h],axis=1).max())
            if err>32:violation=1
            if done or err>threshold:break
    return [seed,threshold,int(done),float(peak),steps,calls,violation]
def save(n,rows):
    with (ROOT/'data'/n).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','trigger_px','success','max_coverage','steps','calls','any_violation']);w.writerows(rows)
def main():
    fore=Forecaster();pol=RetrievalPolicy(fore.fit_ids);env=PushTEnv(obs_type='state')
    q=json.loads((ROOT/'data/closed_loop_summary.json').read_text())['q_fixed_policy']
    tuning=range(33000,33020)
    target=[episode(env,pol,fore,j,'policy-gate',q) for j in tuning]
    cost=sum(r[5] for r in target)/sum(r[4] for r in target)
    options=[1,2,4,8,16,32];tune=[]
    for t in options:tune.extend(reactive(env,pol,fore,j,t) for j in tuning)
    def rate(t):
        a=[r for r in tune if r[1]==t];return sum(r[5] for r in a)/sum(r[4] for r in a)
    chosen=min(options,key=lambda t:abs(cost-rate(t)))
    save('reactive_tuning.csv',tune)
    rows=[reactive(env,pol,fore,j,t) for j in range(32000,32060) for t in sorted({32,chosen})]
    save('reactive_episodes.csv',rows);env.close()
    stat=dict(target_calls_per_100=cost*100,tuning_calls_per_100={str(t):100*rate(t) for t in options},chosen_px=chosen,test={})
    for t in sorted({32,chosen}):
        a=[r for r in rows if r[1]==t]
        stat['test'][str(t)]=dict(success=sum(r[2] for r in a),peak_overlap=np.mean([r[3] for r in a]),violating=sum(r[6] for r in a),calls_per_100=100*sum(r[5] for r in a)/sum(r[4] for r in a))
    (ROOT/'data/reactive_summary.json').write_text(json.dumps(stat,indent=2))
    print(stat)
if __name__=='__main__':main()

