"""Measure repeatability of physical action contrasts in independently reset pushes."""
from pathlib import Path
import csv,json,hashlib,argparse,re,concurrent.futures,zipfile,zlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def prepare(cache):
    from range_archive import Remote
    from real_sources import payload
    cache.mkdir(exist_ok=True,parents=True)
    import urllib.request
    bad=set((ROOT/'data/mcube_excluded_trials.txt').read_text().splitlines())
    index=cache/'member_index.json'
    if index.exists():
        from types import SimpleNamespace
        fs=[SimpleNamespace(**row) for row in json.loads(index.read_text())]
    else:
        with zipfile.ZipFile(Remote()) as ar:fs=ar.infolist()
        keys=('filename','header_offset','CRC','compress_type','compress_size')
        index.write_text(json.dumps([{k:getattr(v,k) for k in keys} for v in fs]))
    groups={}
    for f in sorted(fs,key=lambda x:x.filename):
        m=re.search(r'_s=([\d.]+)_t=([-\d.]+)_rep=(\d+)\.json$',f.filename)
        if not m or Path(f.filename).name in bad:continue
        off,angle=float(m[1]),float(m[2]);rep=int(m[3])
        if off in [.1,.2,.3,.4,.6,.7,.8,.9] and abs(angle) in [.3,.8,1.2]:
            groups.setdefault((off,angle),[]).append((rep,f))
    selected=[(*key,rep,info) for key,files in sorted(groups.items()) for rep,info in files[:4]]
    assert len(selected)==192
    def read(it):
        off,angle,rep,info=it;p=cache/Path(info.filename).name
        if not p.exists():p.write_bytes(payload(info))
        raw=p.read_bytes();assert zlib.crc32(raw)==info.CRC
        j=json.loads(raw);pose=np.array(j['object_pose']);pose=pose[np.unique(pose[:,0],return_index=True)[1]]
        # Use the observed first and last object sample on each trial's native clock.
        return [off,angle,rep,info.filename,hashlib.sha256(raw).hexdigest(),*pose[0],*pose[-1]]
    with concurrent.futures.ThreadPoolExecutor(15) as pool:rows=list(pool.map(read,selected))
    with (ROOT/'data'/'physical_reset_endpoints.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['offset','angle','repeat','archive_member','sha256','t0','x0','y0','theta0','t1','x1','y1','theta1']);w.writerows(rows)

def run():
    groups={}
    with (ROOT/'data/physical_reset_endpoints.csv').open() as f:
        for row in csv.DictReader(f):
            off,ang=map(float,[row['offset'],row['angle']])
            x0,y0,th0,x1,y1,th1=[float(row[k]) for k in ['x0','y0','theta0','x1','y1','theta1']]
            c,s=np.cos(th0),np.sin(th0);theta=th1-th0
            center=1000*np.array([c*(x1-x0)+s*(y1-y0),-s*(x1-x0)+c*(y1-y0)])
            probes=np.array([[a,b] for a in [-30,0,30] for b in [-30,0,30]])
            result=center+probes@np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])-probes
            groups.setdefault((off,ang),[]).append(result)
    def norm(v):return float(np.linalg.norm(v,axis=-1).mean())
    rows=[]
    for off,ang in sorted(k for k in groups if k[1]>0):
        pos=np.array(groups[off,ang]);neg=np.array(groups[off,-ang])
        p0,p1=pos[:2].mean(0),pos[2:].mean(0);n0,n1=neg[:2].mean(0),neg[2:].mean(0)
        delta0=p0-n0;delta1=p1-n1
        rows.append([off,ang,norm((delta0+delta1)/2),.5*(norm(p0-p1)+norm(n0-n1)),norm(delta0-delta1),float(np.vdot(delta0,delta1)/(np.linalg.norm(delta0)*np.linalg.norm(delta1)))])
    with (ROOT/'data/physical_reset_pairs.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['offset','angle_magnitude','contrast_mm','within_command_split_mm','contrast_split_disagreement_mm','split_cosine']);w.writerows(rows)
    mat=np.array(rows);dif=mat[:,2]-mat[:,3];parts=np.array([dif[mat[:,0]==s].mean() for s in np.unique(mat[:,0])])
    sample=np.random.default_rng(626).choice(parts,(5000,len(parts)),replace=True).mean(1)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(3.45,2.15));maximum=1.02*max(mat[:,2:4].max(),1)
    ax.plot([0,maximum],[0,maximum],color='#69777a',ls=':',lw=.9)
    for angle,mk,col in [(0.3,'o','#21617b'),(0.8,'s','#b26733'),(1.2,'^','#75578c')]:
        pts=mat[mat[:,1]==angle]
        ax.scatter(pts[:,3],pts[:,2],s=17,marker=mk,c=col,label=f'{angle:.1f} rad',alpha=.88)
    ax.set(xlim=(0,1.07*mat[:,3].max()),ylim=(0,maximum),xlabel='Repeat discrepancy (mm)',ylabel='Signed-action contrast (mm)')
    ax.tick_params(labelsize=7);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8);ax.grid(alpha=.14)
    ax.legend(frameon=False,fontsize=6,ncol=3,loc='upper center');ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=.45);fig.savefig(ROOT/'figures/physical_contrasts.pdf');plt.close(fig)
    print('conditions',len(rows),'means',mat[:,2:].mean(0),'positive',sum(dif>0),'CI',np.quantile(sample,[.025,.975]))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fetch',type=Path);a=ap.parse_args()
    if a.fetch:prepare(a.fetch)
    run()

