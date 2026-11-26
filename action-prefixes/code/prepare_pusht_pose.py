"""Extract state and a fixed nine-probe template from the original Push-T archive."""
from pathlib import Path
import argparse,hashlib
import numpy as np
import zarr
from prepare_pusht import ROOT,DIGEST
def extract(archive):
    with archive.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=DIGEST:raise ValueError('Archive checksum mismatch')
    with zarr.ZipStore(str(archive),mode='r') as st:
        d=zarr.open_group(st,path='pusht/pusht_cchi_v7_replay.zarr',mode='r')
        pose=d['data/state'][:];k=d['data/keypoint'][:]
        c,s=np.cos(pose[:,4]),np.sin(pose[:,4])
        rot=np.stack([c,-s,s,c],-1).reshape(-1,2,2)
        body=(k-pose[:,None,2:4])@rot
        return dict(pose=pose,body=body[0].astype('f8'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path)
    np.savez_compressed(ROOT/'data/pusht_pose.npz',**extract(p.parse_args().archive))

