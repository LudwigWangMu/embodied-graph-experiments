"""Paired impulse-response experiment. Run from any directory using Python 3.12."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import csv, json
import numpy as np
import pymunk
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (4, 8, 16)
DT = 1 / 90
RADIUS = 0.12
MODELS = ("Source only", "Pooled RFF", "Isotropic", "One hop", "No gap gate", "Directional")

def draw_scene(rng, n, angle):
    heading = angle + np.cumsum(rng.normal(0, 0.14, n-1))
    gaps = rng.uniform(0.001, 0.042, n-1)
    xy = np.vstack([np.zeros(2), np.cumsum((2*RADIUS + gaps)[:,None] *
                   np.stack([np.cos(heading), np.sin(heading)], axis=1), axis=0)])
    mass = rng.uniform(0.7, 1.5, n)
    vel = rng.normal(0, 0.035, (n, 2))
    phi = angle + rng.uniform(-0.65, 0.65)
    impulse = rng.uniform(0.65, 1.65)*np.array([np.cos(phi), np.sin(phi)])
    return xy, mass, vel, impulse

def rollout(scene, apply):
    xy, mass, vel, impulse = scene
    space = pymunk.Space()
    space.iterations = 20
    bodies = []
    for pos, m, v in zip(xy, mass, vel):
        body = pymunk.Body(m, pymunk.moment_for_circle(m, 0, RADIUS))
        body.position = tuple(pos); body.velocity = tuple(v)
        shape = pymunk.Circle(body, RADIUS); shape.elasticity = 0.70; shape.friction = 0.25
        space.add(body, shape); bodies.append(body)
    if apply: bodies[0].apply_impulse_at_local_point(tuple(impulse))
    samples = []
    for k in range(1,max(HORIZONS)+1):
        space.step(DT)
        if k in HORIZONS: samples.append([tuple(b.position) for b in bodies])
    return np.asarray(samples)

def inputs(scenes):
    return tuple(np.stack(t) for t in zip(*scenes))

def features(data, variant, missing_edge=None):
    xy, m, v, j = data
    batch, n, _ = xy.shape
    j0 = np.zeros_like(xy); j0[:,0] = j / m[:,0,None]
    if variant == "Pooled RFF":
        # A permutation-equivariant per-node nonlinear kernel with global pooling.
        rel = xy - xy[:,:1]
        u = np.concatenate([rel, v, m[:,:,None], j0, np.broadcast_to(j[:,None],xy.shape),
                            np.broadcast_to(rel.mean(1,keepdims=True),xy.shape)],axis=-1)
        rng = np.random.default_rng(412)
        hidden = np.cos(u @ rng.normal(0,2,(u.shape[-1],96)) + rng.uniform(0,2*np.pi,96))
        return np.concatenate([u,hidden,np.ones((batch,n,1))],axis=-1)
    if variant=="Source only":return j0[...,None]
    dr = xy[:,None,:,:] - xy[:,:,None,:]  # sender i, receiver j
    dist = np.linalg.norm(dr,axis=-1)
    unit = dr / np.maximum(dist[...,None],1e-12)
    gap = np.maximum(dist - 2*RADIUS,0)
    nbr = (dist>0) & (gap < 0.10)
    if missing_edge is not None:
        i,jedge=missing_edge
        nbr[:,i,jedge]=False
        nbr[:,jedge,i]=False
    ratio = 2*m[:,:,None] / (m[:,:,None]+m[:,None,:])
    out = [j0]
    for scale in (0.012,0.035,0.085):
        weight = nbr*ratio
        if variant!="No gap gate":weight=weight*np.exp(-gap/scale)
        current = j0.copy()
        for _ in range(1 if variant=="One hop" else 4):
            if variant=="Isotropic": msg=current[:,:,None,:]*np.ones_like(unit)
            else:
                dot=np.sum(current[:,:,None,:]*unit,axis=-1).clip(0)
                msg=dot[...,None]*unit
            current = (weight[...,None]*msg).sum(axis=1)
            out.append(current)
        if variant=="No gap gate":break
    return np.stack(out,axis=-1)

def fit(data,y,name):
    x = features(data,name)
    if name=="Pooled RFF":
        a=x.reshape(-1,x.shape[-1]); b=y.transpose(0,2,1,3).reshape(len(a),-1)
    else:
        a=x.reshape(-1,x.shape[-1]); b=y.transpose(0,2,3,1).reshape(len(a),-1)
    return np.linalg.solve(a.T@a+1e-4*np.eye(a.shape[1]),a.T@b)

def predict(data,name,w):
    x=features(data,name); z=x@w
    if name=="Pooled RFF": return z.reshape(len(z),z.shape[1],len(HORIZONS),2).transpose(0,2,1,3)
    return z.transpose(0,3,1,2)

def sample(rng,count,n,rotated):
    angle=(rng.uniform(1.7,4.5,count) if rotated else rng.uniform(-0.55,0.55,count))
    scenes=[draw_scene(rng,n,a) for a in angle]
    y=np.stack([rollout(s,True)-rollout(s,False) for s in scenes])
    return inputs(scenes),y

def intervals(loss):
    rng=np.random.default_rng(941); ids=rng.integers(0,len(loss),(1500,len(loss)))
    return [float(a) for a in np.quantile(loss[ids].mean(1),[.025,.975])]

def main():
    rng=np.random.default_rng(1729)
    train,target=sample(rng,1400,5,False)
    coefficients={m:fit(train,target,m) for m in MODELS}
    rows=[]; detail=[]; pred_example=None; tests={}
    for split,n,rot in [('In range',5,False),('Rotated',5,True),('Long chain',8,True)]:
        data,y=sample(rng,480,n,rot); tests[split]=(data,y)
        losses={}
        for m in MODELS:
            yp=predict(data,m,coefficients[m])
            err=1000*np.linalg.norm((yp-y)[:,:,1:,:],axis=-1).mean(2)
            for k,h in enumerate(HORIZONS):
                loss=err[:,k]; lo,hi=intervals(loss)
                rows.append([split,m,h,loss.mean(),lo,hi])
                detail.extend([split,m,h,i,l] for i,l in enumerate(loss))
            losses[m]=err[:,-1]
        gap=losses['Directional']-losses['One hop']
        print(split,'paired vs one hop',gap.mean(),intervals(gap))
    for name,header,records in [('results',['split','model','steps','mae_mm','ci_low','ci_high'],rows),
                                ('scene_errors',['split','model','steps','scene','mae_mm'],detail)]:
        with (ROOT/'data'/f'{name}.csv').open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(header);wr.writerows(records)
    data,y=tests['In range'];xy,m,v,j=data; theta=1.34
    q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    pred=predict(data,'Directional',coefficients['Directional'])
    rotated=predict((xy@q.T+np.array([3.,-2.]),m,v@q.T,j@q.T),'Directional',coefficients['Directional'])
    perm=[0,3,1,4,2]; pe=predict((xy[:,perm],m[:,perm],v[:,perm],j),'Directional',coefficients['Directional'])
    checks={'rotation_translation_max_m':float(np.max(abs(rotated-pred@q.T))),
            'passive_relabel_max_m':float(np.max(abs(pe-pred[:,:,perm]))),'train':1400,'test_per_split':480}
    (ROOT/'data'/'checks.json').write_text(json.dumps(checks,indent=2))
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.55),sharey=True)
    colors=['#707070','#9e5361','#6863a0','#be7020','#548573','#174c78']
    for ax,(split,_) in zip(axes,tests.items()):
        for model,col,mk in zip(MODELS,colors,["x","s","^","v","D","o"]):
            vals=[r for r in rows if r[0]==split and r[1]==model]
            ax.plot(HORIZONS,[r[3] for r in vals],marker=mk,markersize=3.0,color=col,label=model,linewidth=1.4)
            ax.fill_between(HORIZONS,[r[4] for r in vals],[r[5] for r in vals],alpha=.11,color=col)
        ax.set_yscale('log');ax.set_ylim(.65,170);ax.set_yticks([1,3,10,30,100],labels=['1','3','10','30','100']);ax.set_title(split);ax.set_xticks(HORIZONS);ax.set_xlabel('Physics steps');ax.grid(axis='y',alpha=.17)
    axes[0].set_ylabel('Passive-node error (mm)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=6,fontsize=7,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(ROOT/'figures'/'transfer.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.45,1.8)); case=int(np.argmax(np.linalg.norm(y[:,-1,2],axis=-1)))
    xy0=xy[case]; truth=y[case,-1]; estimate=pred[case,-1]
    for i,p in enumerate(xy0):
        ax.add_patch(Circle(p,RADIUS,facecolor='#d6e4ea' if i else '#f2d6b8',edgecolor='#476174',lw=.8))
        ax.text(*p,str(i),ha='center',va='center',fontsize=7)
        ax.arrow(*(p+np.array([0,.052])),*(truth[i]*2),color='#175b7d',width=.006,head_width=.031,length_includes_head=True)
        ax.arrow(*(p-np.array([0,.052])),*(estimate[i]*2),color='#bd641e',width=.006,head_width=.031,length_includes_head=True)
    ax.autoscale();ax.margins(.12);ax.set_aspect('equal');ax.axis('off')
    fig.tight_layout(pad=.2);fig.savefig(ROOT/'figures'/'response.pdf',bbox_inches='tight');plt.close(fig)
    print(checks)
if __name__=='__main__':main()
