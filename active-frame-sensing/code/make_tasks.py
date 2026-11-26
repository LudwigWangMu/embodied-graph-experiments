"""Build graph-calibration tasks, retaining maps, origins, goals, and split keys."""
from pathlib import Path
from collections import deque
import csv, hashlib, io, zipfile
import numpy as np
from scipy.ndimage import gaussian_filter
import requests

ROOT=Path(__file__).resolve().parents[1]
CITIES="Berlin Boston Denver London Milan Moscow NewYork Paris Shanghai Sydney".split()
A=np.array([[-1,0],[0,1],[1,0],[0,-1]],int)
U=np.vstack([A,3*A,[[2,2],[2,-2],[-2,2],[-2,-2]]])
M=np.array([[[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]]])
M=np.concatenate([M,M@np.diag([1,-1])])
GA= A @ M.transpose(0,2,1)
GU= U @ M.transpose(0,2,1)
ORIGIN=np.array([15,15])
def distances(free,start):
    h,w=free.shape
    dist=np.full((h,w),-1,dtype=int);y,x=map(int,start);dist[y,x]=0;q=deque([(y,x)])
    while q:
        y,x=q.popleft();nd=dist[y,x]+1
        for yy,xx in ((y-1,x),(y,x+1),(y+1,x),(y,x-1)):
            if 0<=yy<h and 0<=xx<w and free[yy,xx] and dist[yy,xx]<0:
                dist[yy,xx]=nd;q.append((yy,xx))
    return dist
def encode(free,goal):
    d=distances(free,goal);ds=d[15,15];assert ds>=0
    nbr=ORIGIN+GA;positive=free[nbr[:,:,0],nbr[:,:,1]]
    c=np.where(positive,1+d[nbr[:,:,0],nbr[:,:,1]]-ds,3)
    probe=ORIGIN+GU;sig=free[probe[:,:,0],probe[:,:,1]].T
    assert np.all(c.min(1)==0) and set(np.unique(c))<={0,2,3}
    return sig.astype('u1'),c.astype('u1'),ds
def city_maps():
    store=ROOT/'data/maps';store.mkdir(exist_ok=True);grids={};manifest=[]
    for city in CITIES:
        name=city+'_0_256.map';url='https://movingai.com/benchmarks/street/'+name+'.zip'
        file=store/(name+'.zip')
        if not file.exists():
            response=requests.get(url,timeout=35);response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:assert z.testzip() is None
            file.write_bytes(response.content)
        with zipfile.ZipFile(file) as z:source=z.read(next(n for n in z.namelist() if n.endswith('.map')))
        lines=source.splitlines()[4:];a=np.array([np.frombuffer(row,dtype='u1') for row in lines])
        assert a.shape==(256,256);grids[city]=np.isin(a,[ord('.'),ord('G'),ord('S')])
        manifest.append([city,url,hashlib.sha256(file.read_bytes()).hexdigest(),hashlib.sha256(source).hexdigest(),''.join(chr(int(k)) for k in np.unique(a))])
    with (ROOT/'data/map_sources.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['city','upstream_archive','zip_sha256','map_sha256','symbols']);w.writerows(manifest)
    return grids
def main():
    maps=city_maps();records=[];meta=[]
    def save(free,goal,split,source,group,y,x,seed):
        sig,c,dd=encode(free,goal);records.append((free.copy(),goal.copy(),sig,c))
        meta.append([len(records)-1,split,source,group,y,x,seed,dd])
    configs=[('fit','procedural',1400,8001),('validation','procedural',180,8002),('test','procedural',320,8003)]
    configs.extend(('city',city,40,8500+j) for j,city in enumerate(CITIES))
    for split,source,count,seed in configs:
        rng=np.random.default_rng(seed);accepted=0
        for attempt in range(100*count):
            if accepted==count:break
            if source=='procedural':
                field=gaussian_filter(rng.normal(size=(31,31)),rng.uniform(.45,1.65))
                free=field>np.quantile(field,rng.uniform(.18,.42));y=x=15
            else:
                y,x=rng.integers(15,241,2);free=maps[source][y-15:y+16,x-15:x+16].copy()
            if not free[15,15]:continue
            d=distances(free,ORIGIN);options=np.argwhere((d>=6)&(d<=18))
            if len(options)==0:continue
            goal=options[int(rng.integers(len(options)))];group=source+'-'+str(accepted)
            save(free,goal,split,source,group,y,x,seed)
            if split=='test':
                dg=distances(free,goal);nbr=ORIGIN+A
                beneficial=nbr[dg[nbr[:,0],nbr[:,1]]==dg[15,15]-1]
                blocked=beneficial[int(rng.integers(len(beneficial)))];fault=free.copy();fault[tuple(blocked)]=False
                if distances(fault,goal)[15,15]>=0:save(fault,goal,'closure',source,group,y,x,seed)
            accepted+=1
        assert accepted==count
    with (ROOT/'data/tasks.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','split','source','pair','origin_y','origin_x','seed','distance']);w.writerows(meta)
    f,g,s,c=map(np.stack,zip(*records));np.savez_compressed(ROOT/'data/tasks.npz',free=f,goal=g,signature=s,cost=c)
if __name__=='__main__':main()

