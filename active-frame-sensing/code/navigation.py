"""Closed-loop simulator with known positions, latent fixed frame and persistent posteriors."""
import csv
from functools import lru_cache
import numpy as np
from make_tasks import ROOT,GA,GU,distances
from decision import likelihood
from evaluate import write

def choice(x,rng):
    ix=np.flatnonzero(np.isclose(x,np.min(x),atol=1e-10,rtol=0))
    return int(ix[rng.integers(len(ix))])
def setup_policy(bits,c,w,remaining,used,price):
    base=(w@c).min()
    if remaining==0:return base,None
    like=likelihood(bits,.1)
    one=w[None,None,:]*like
    if remaining==1:
        vals=(one@c).min(-1).sum(1)+price*w.sum()
    else:
        leaf=(one[:,:,None,None,:]*like[None,None,:,:,:])@c
        q2=leaf.min(-1).sum(-1)+price*one.sum(-1)[:,:,None]
        mask=np.eye(len(bits),dtype=bool)[:,None,:]
        b2=np.where(mask,np.inf,q2).min(-1)
        vals=price*w.sum()+np.minimum((one@c).min(-1),b2).sum(1)
    vals=vals.copy();vals[used]=np.inf
    if base<=vals.min()+1e-11:return base,None
    return vals.min(),vals
def track(free,goal,theta,uniform,method,tie_seed,d):
    pos=np.array([15,15]);w=np.ones(8)/8;steps=bumps=queries=0
    tie=np.random.default_rng(tie_seed);initial=d[tuple(pos)]
    for t in range(60):
        if np.array_equal(pos,goal):break
        nxt=pos+GA;inside=((nxt>=0)&(nxt<31)).all(-1)
        ij=nxt.clip(0,30);ok=inside & free[ij[:,:,0],ij[:,:,1]];nxt=np.where(ok[:,:,None],nxt,pos)
        c=1+d[nxt[:,:,0],nxt[:,:,1]]-d[tuple(pos)]+2*(~ok)
        jj=pos+GU;inside=((jj>=0)&(jj<31)).all(-1);ij=jj.clip(0,30)
        s=(inside & free[ij[:,:,0],ij[:,:,1]]).T
        used=[];limit=0 if method=='motion-only' or (method=='start' and t>0) else 2
        for k in range(limit):
            if np.count_nonzero(w)>1:
                _,values=setup_policy(s,c,w,limit-k,used,.10 if method=='selective' else 0)
                if values is None:break
                probe=choice(values,tie);used.append(probe);queries+=1
                bit=int(s[probe,theta])^int(uniform[t,probe]<.1)
                w*=likelihood(s,.1)[probe,bit];w/=w.sum()
            else:break
        action=choice(w@c,tie);new=nxt[theta,action].copy();bumps+=int(np.array_equal(new,pos))
        w*=np.all(nxt[:,action]==new,axis=1);w/=w.sum();pos=new;steps+=1
    return int(np.array_equal(pos,goal)),steps,bumps,queries,steps-initial,steps-initial+2*bumps+.10*queries
def main():
    with (ROOT/'data/tasks.csv').open() as f:meta=list(csv.DictReader(f))
    ds=np.load(ROOT/'data/tasks.npz');rows=[]
    for split,n in [('test',120),('city',12)]:
        for source in sorted(set(r['source'] for r in meta if r['split']==split)):
            ids=[int(r['index']) for r in meta if r['split']==split and r['source']==source][:n]
            for i in ids:
                free,goal=ds['free'][i],ds['goal'][i];d=distances(free,goal)
                for theta in range(8):
                    seed=9000+8*i+theta;u=np.random.default_rng(seed).random((60,12))
                    for method in ['motion-only','start','selective']:
                        rows.append((i,split,source,theta,method,*track(free,goal,theta,u,method,seed+317,d)))
    write('navigation.csv',['index','split','source','frame','method','success','actions','bumps','queries','extra_actions','charged_excess'],rows)
if __name__=='__main__':main()

