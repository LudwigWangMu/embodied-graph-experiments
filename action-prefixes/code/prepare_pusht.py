"""Extract the four low-dimensional arrays needed to reproduce the replay."""
import argparse,hashlib,urllib.request
from pathlib import Path
import numpy as np
import zarr
ROOT=Path(__file__).resolve().parents[1]
URL='https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip'
DIGEST='63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9'
def read_archive(path):
    if not path.exists():urllib.request.urlretrieve(URL,path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=DIGEST:raise ValueError('Unexpected archive checksum')
    with zarr.ZipStore(str(path),mode='r') as store:
        z=zarr.open_group(store,path='pusht/pusht_cchi_v7_replay.zarr',mode='r')
        return dict(landmarks=z['data/keypoint'][:],pusher=z['data/state'][:,:2],
                    action=z['data/action'][:],ends=z['meta/episode_ends'][:])
if __name__=='__main__':
    ag=argparse.ArgumentParser();ag.add_argument('--archive',type=Path,default=Path('pusht.zip'))
    np.savez_compressed(ROOT/'data'/'pusht_lowdim.npz',**read_archive(ag.parse_args().archive))
