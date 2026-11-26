"""Residual graph readout on an axial impact solver; second independent evaluation seed."""
from axial_baseline import *
def analytic(data):
    return np.stack([collinear(s,True)-collinear(s,False) for s in zip(*data)])
def correction(data):
    raw=features(data,'Directional');axis=data[0][:,1]-data[0][:,0]
    axis=axis/np.linalg.norm(axis,axis=1,keepdims=True)
    along=np.einsum('bncf,bc->bnf',raw,axis)[:,:,None,:]*axis[:,None,:,None]
    return np.concatenate([raw,raw-along],axis=-1)
def fitted(x,target):
    a=x.reshape(-1,x.shape[-1]);b=target.transpose(0,2,3,1).reshape(len(a),-1)
    return np.linalg.solve(a.T@a+1e-4*np.eye(a.shape[1]),a.T@b)
def main():
    rng=np.random.default_rng(1729);train,y=sample(rng,1400,5,False)
    beta=fitted(correction(train),y-analytic(train))
    rows=[];paired=[]
    for seed in [1729,28107]:
        rng=np.random.default_rng(seed)
        if seed==1729:sample(rng,1400,5,False)
        for split,n,rot in [('In range',5,False),('Rotated',5,True),('Long chain',8,True)]:
            data,truth=sample(rng,480,n,rot);base=analytic(data);pred=base+(correction(data)@beta).transpose(0,3,1,2)
            err0=1000*np.linalg.norm((base-truth)[:,:,1:],axis=-1).mean(2)
            err1=1000*np.linalg.norm((pred-truth)[:,:,1:],axis=-1).mean(2)
            for k,h in enumerate(HORIZONS):
                diff=err1[:,k]-err0[:,k];rows.append([seed,split,h,err0[:,k].mean(),err1[:,k].mean(),diff.mean(),*intervals(diff)])
                if seed==28107 and h==16:paired.extend([split,i,e0,e1] for i,(e0,e1) in enumerate(zip(err0[:,k],err1[:,k])))
    for name,columns,data in [('hybrid',['scene_seed','split','horizon','axial_mm','corrected_mm','paired_delta_mm','low','high'],rows),
                              ('hybrid_pairs',['split','scene','axial_mm','corrected_mm'],paired)]:
        with (ROOT/'data'/(name+'.csv')).open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(columns);wr.writerows(data)
    print([r for r in rows if r[2]==16])
if __name__=='__main__':main()
