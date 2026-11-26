"""Compare the RGB renderer with archived frames at matched measured poses."""
import argparse,csv
from pathlib import Path
import numpy as np
from gym_pusht.envs.pusht import PushTEnv
from prepare_pusht import ROOT,DIGEST
def prepare(archive):
    import hashlib,zarr
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==DIGEST
    with zarr.ZipStore(str(archive),mode='r') as src:
        z=zarr.open_group(src,path='pusht/pusht_cchi_v7_replay.zarr',mode='r');ends=z['meta/episode_ends'][:]
        test=np.random.default_rng(1024).permutation(len(ends))[146:]
        starts=np.r_[0,ends[:-1]];ids=((ends[test]+starts[test])//2).astype(int)
        np.savez_compressed(ROOT/'data/render_alignment_frames.npz',episode=test,frame=ids,pose=np.stack([z['data/state'][int(i)] for i in ids]),rgb=np.stack([z['data/img'][int(i)].astype('u1') for i in ids]))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path);a=ap.parse_args()
    if a.archive:prepare(a.archive)
    d=np.load(ROOT/'data/render_alignment_frames.npz');e=PushTEnv(obs_type='pixels_agent_pos');e.reset(seed=10);rows=[]
    for ep,fr,s,img in zip(d['episode'],d['frame'],d['pose'],d['rgb']):
        e.agent.position=tuple(s[:2]);e.block.angle=float(s[4]);e.block.position=tuple(s[2:4])
        e.space.reindex_shapes_for_body(e.block);e.space.reindex_shapes_for_body(e.agent)
        actual=e.get_obs()['pixels'];rr=np.abs(img.astype(float)-actual)
        rows.append([ep,fr,rr.mean(),(rr>0).mean(),rr.max()])
    e.close()
    with (ROOT/'data/render_alignment.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['episode','frame','mean_channel_difference_0_255','fraction_changed_channels','maximum_channel_difference']);wr.writerows(rows)
    print(np.mean([r[2] for r in rows]),np.max([r[2] for r in rows]))
if __name__=='__main__':main()

