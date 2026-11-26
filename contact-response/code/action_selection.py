"""Blind terminal target selection across five impulses from a common initial scene."""
from experiment import *
def main():
    train,y=sample(np.random.default_rng(1729),1400,5,False)
    weights={name:fit(train,y,name) for name in MODELS}
    rng=np.random.default_rng(5631);scenes=[];targets=[]
    for _ in range(320):
        angle=rng.uniform(1.7,4.5);s=draw_scene(rng,5,angle);xy,m,v,j=s
        direction=(xy[2]-xy[0])/np.linalg.norm(xy[2]-xy[0])
        targets.append(rng.uniform(.008,.060)*direction)
        for delta,mult in [(-.55,.72),(-.22,1.),(0,1.25),(.22,1.),(.55,.72)]:
            q=np.array([[np.cos(delta),-np.sin(delta)],[np.sin(delta),np.cos(delta)]])
            scenes.append((xy,m,v,mult*q@j))
    data=inputs(scenes);ys=np.stack([rollout(s,True)[-1,2]-rollout(s,False)[-1,2] for s in scenes])
    def costs(d):return np.linalg.norm(d.reshape(-1,5,2)-np.asarray(targets)[:,None],axis=-1)*1000
    truecost=costs(ys);best=truecost.min(axis=1);rows=[]
    for name in (*MODELS,'Uniform expectation'):
        if name=='Uniform expectation':real=truecost.mean(axis=1)
        else:
            pred=predict(data,name,weights[name])[:,-1,2]
            ix=costs(pred).argmin(axis=1);real=truecost[np.arange(len(ix)),ix]
        reg=real-best
        rows.append([name,real.mean(),reg.mean(),*intervals(reg)])
    with (ROOT/'data'/'actions.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['model','endpoint_mm','regret_mm','regret_low','regret_high']);wr.writerows(rows)
    print(rows)
if __name__=='__main__':main()
