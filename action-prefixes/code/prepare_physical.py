"""Select one non-blacklisted trial per condition from the MIT MCube repeatability data."""
import argparse,concurrent.futures,csv,hashlib,io,json,re,struct,time,urllib.request as U,zipfile,zlib
from pathlib import Path
import numpy as np
from range_archive import Remote,URL
ROOT=Path(__file__).resolve().parents[1]
BAD='https://dl.dropboxusercontent.com/s/buolrd3y9v3x31l/unreliable_files.txt'
def payload(info):
    start=info.header_offset
    for trial in range(3):
        try:
            req=U.Request(URL+'?segment='+str(start),headers={'Range':f'bytes={start}-{start+info.compress_size+510}'})
            with U.urlopen(req,timeout=25) as res:buf=res.read()
            header=struct.unpack('<IHHHHHIIIHH',buf[:30]);assert header[0]==0x04034b50
            offset=30+header[-1]+header[-2]
            assert buf[30:30+header[-2]].decode()==info.filename
            raw=buf[offset:offset+info.compress_size]
            data=zlib.decompress(raw,-15) if info.compress_type==zipfile.ZIP_DEFLATED else raw
            assert zlib.crc32(data)==info.CRC
            return data
        except Exception:
            if trial==2:raise
            time.sleep(1+trial)
def align_record(data):
    j=json.loads(data);tip=np.asarray(j['tip_pose']);obj=np.asarray(j['object_pose'])
    # Keep the first copy of duplicate timestamps; interpolate over common support only.
    def unique(a):return a[np.unique(a[:,0],return_index=True)[1]]
    tip=unique(tip);obj=unique(obj)
    lo=max(tip[0,0],obj[0,0]);hi=min(tip[-1,0],obj[-1,0]);tt=np.arange(lo,hi,.02)
    if len(tt)<10:raise ValueError('Too short')
    # Inputs at a grid time only use instrument samples that have already arrived.
    it=np.searchsorted(tip[:,0],tt,side='right')-1
    io=np.searchsorted(obj[:,0],tt,side='right')-1
    pxy=tip[it,1:3]*1000;xy=obj[io,1:3]*1000
    # Virtual body-frame probes require pose, without additional motion capture markers.
    theta=obj[io,3]
    cs,ss=np.cos(theta),np.sin(theta);rot=np.stack([cs,-ss,ss,cs],-1).reshape(-1,2,2)
    grid=np.array([[a,b] for a in (-30,0,30) for b in (-30,0,30)])
    tracks=xy[:,None]+np.einsum('tij,kj->tki',rot,grid)
    return pxy,tracks,hashlib.sha256(data).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,default=Path('mcube-cache'));cache=ap.parse_args().cache;cache.mkdir(exist_ok=True)
    bad=set(U.urlopen(BAD,timeout=20).read().decode().splitlines())
    with zipfile.ZipFile(Remote()) as archive:info=archive.infolist()
    groups={}
    for v in sorted(info,key=lambda x:x.filename):
        name=Path(v.filename).name
        m=re.search(r'_s=([-\d.]+)_t=([-\d.]+)_rep=(\d+)\.json$',name)
        if m and name not in bad:
            side,angle=map(float,m.groups()[:2])
            if abs(angle)<=1.5:groups.setdefault((side,angle),v)
    keys=sorted(groups);ids=np.random.default_rng(6174).permutation(len(keys))[:206];selected=[groups[keys[i]] for i in ids]
    def download(v):
        p=cache/Path(v.filename).name
        if not p.exists():p.write_bytes(payload(v))
        data=p.read_bytes()
        if zlib.crc32(data)!=v.CRC:raise ValueError('CRC mismatch')
        return align_record(data)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:records=list(pool.map(download,selected))
    np.savez_compressed(ROOT/'data'/'physical_tracks.npz',pusher=np.concatenate([r[0] for r in records]),landmarks=np.concatenate([r[1] for r in records]),ends=np.cumsum([len(r[0]) for r in records]))
    with (ROOT/'data'/'physical_sources.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['episode','archive_member','sha256','replay_frames'])
        for i,(v,r) in enumerate(zip(selected,records)):wr.writerow([i,v.filename,r[2],len(r[0])])
    print('eligible conditions',len(groups),'selected',len(records),'frames',sum(len(r[0]) for r in records))
if __name__=='__main__':main()
