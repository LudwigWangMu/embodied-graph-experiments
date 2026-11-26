"""Exact two-query policies. All risks integrate the orientation and the sensor outcomes."""
import numpy as np
N=12
def tied_min(x,axis=-1):
    v=x.min(axis=axis,keepdims=True)
    mask=np.isclose(x,v,rtol=0,atol=1e-11);return mask/mask.sum(axis=axis,keepdims=True)
def likelihood(s,e):
    return np.stack([np.where(s, e, 1-e),np.where(s,1-e,e)],axis=1)
def signature_floor(s,c,prior=None):
    _,labels=np.unique(s.T,axis=0,return_inverse=True)
    prior=np.ones(len(labels))/len(labels) if prior is None else np.asarray(prior)
    return sum((c[labels==g]*prior[labels==g,None]).sum(0).min() for g in np.unique(labels))
def first_features(s,c):
    bins=np.stack([c==v for v in (0,2,3)],axis=-1).astype(float)
    # For each probe, preserve each action's cost histogram on either side of its binary split.
    return np.einsum('ubh,hak->uabk',likelihood(s,.10),bins).reshape(len(s),4,6)/8
class Tree:
    def __init__(self,s,c,noise=.1,shared=False,prior=None):
        self.s=s;self.c=c
        like=likelihood(s,noise)
        self.prior=np.ones(len(c))/len(c) if prior is None else np.asarray(prior,dtype=float)
        assert np.isclose(self.prior.sum(),1) and np.all(self.prior>=0)
        self.one=like*self.prior
        if shared:
            a=likelihood(s,0);b=likelihood(s,1)
            self.joint=((1-noise)*a[:,:,None,None,:]*a[None,None,:,:,:]+noise*b[:,:,None,None,:]*b[None,None,:,:,:])*self.prior
        else:self.joint=like[:,:,None,None,:]*like[None,None,:,:,:]*self.prior
        self.leaf=self.joint@c
        self.term=tied_min(self.leaf)
        self.twoval=self.leaf.min(-1).sum(-1)  # first probe, first bit, second probe
        self.valid=(~np.eye(len(s),dtype=bool))[:,None,:]
        self.second=tied_min(np.where(self.valid,self.twoval,np.inf))
        self.q=(self.second*self.twoval).sum((1,2))
        self.first=tied_min(self.q)
        self.greedy=tied_min((self.one@c).min(-1).sum(-1))
        p1=like[:,1]@self.prior
        self.info_first=tied_min(abs(p1-.5))
        # At matched independent symmetric noise, mutual information is maximized by
        # the most balanced predicted bit. Constant branch likelihood cancels in comparisons.
        p2=self.joint[:,:,:,1,:].sum(-1)
        half=self.one.sum(-1,keepdims=True)*.5
        self.info_second=tied_min(np.where(self.valid,abs(p2-half),np.inf))
    def policies(self):
        n=len(self.s);rnd=np.ones(n)/n
        rand2=np.broadcast_to(self.valid/(n-1),(n,2,n))
        pair=tied_min(np.where(self.valid[:,0,:],self.twoval.sum(1),np.inf).ravel()).reshape(n,n)
        f=pair.sum(-1);v=np.divide(pair,f[:,None],out=np.zeros_like(pair),where=f[:,None]>0)
        return {'random':(rnd,rand2),'entropy':(self.info_first,self.info_second),'greedy':(self.greedy,self.second),
                'fixed-pair':(f,np.repeat(v[:,None,:],2,axis=1)),'adaptive':(self.first,self.second)}
    def score(self,first,second,world=None):
        if world is None:world=self
        weights=first[:,None,None,None,None]*second[:,:,:,None,None]*self.term
        risk=float((weights*world.leaf).sum())
        bump=float((weights*(world.joint@(self.c==3).astype(float))).sum())
        progress=float((weights*(world.joint@(self.c==0).astype(float))).sum())
        return risk,bump,progress
def exhaustive_check(s,c,noise):
    # Enumerate ordered first probe and the two subsequent probes with explicit posterior sums.
    like=likelihood(s,noise);values=[]
    for u in range(len(s)):
        val=0
        for y in (0,1):
            opts=[]
            for v in range(len(s)):
                if v==u:continue
                zcost=0
                for z in (0,1):
                    mass=like[u,y]*like[v,z]/8
                    zcost+=min(sum(mass[h]*c[h,a] for h in range(8)) for a in range(4))
                opts.append(zcost)
            val+=min(opts)
        values.append(val)
    return np.array(values)

