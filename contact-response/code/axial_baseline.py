"""Known-parameter control: analytic motion and binary impacts on a straight line."""
from experiment import *
def collinear(s,apply):
    xy,m,v,j=s
    # Use arclength so adjacent clearances are unchanged by flattening the chain.
    axes=np.diff(xy,axis=0);dist=np.linalg.norm(axes,axis=1);u=axes[0]/dist[0]
    pos=np.r_[0,np.cumsum(dist)];initial=pos.copy();vel=v@u
    if apply:vel[0]+=j@u/m[0]
    yy=[];clock=0
    for h in HORIZONS:
        end=h*DT
        for _ in range(100):
            remaining=end-clock;closing=vel[:-1]-vel[1:];gap=np.maximum(0,np.diff(pos)-2*RADIUS)
            nxt=np.full(len(gap),np.inf);valid=closing>1e-9;nxt[valid]=gap[valid]/closing[valid]
            i=int(nxt.argmin());d=nxt[i]
            if d>remaining or remaining<=1e-12:
                pos+=vel*remaining;clock=end;break
            pos+=d*vel;clock+=d
            imp=(1+.7**2)*closing[i]/(1/m[i]+1/m[i+1])
            vel[i]-=imp/m[i];vel[i+1]+=imp/m[i+1]
        else:raise AssertionError('Impact iteration limit')
        yy.append((pos-initial)[:,None]*u[None,:])
    return np.stack(yy)
def main():
    rng=np.random.default_rng(1729)
    sample(rng,1400,5,False);rows=[]
    for split,n,r in [('In range',5,False),('Rotated',5,True),('Long chain',8,True)]:
        data,y=sample(rng,480,n,r);pred=np.stack([collinear(s,True)-collinear(s,False) for s in zip(*data)])
        errors=1000*np.linalg.norm((pred-y)[:,:,1:],axis=-1).mean(2)
        for k,h in enumerate(HORIZONS):
            loss=errors[:,k];rows.append([split,h,loss.mean(),*intervals(loss)])
    with (ROOT/'data'/'axial.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['split','steps','mae_mm','low','high']);wr.writerows(rows)
    print(rows)
if __name__=='__main__':main()
