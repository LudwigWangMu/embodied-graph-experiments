"""Hold the visual diffusion policy fixed and extend only the test episode cap."""
import argparse,csv
import numpy as np
from visual_diffusion import ImagePolicy,REV
from frozen_vla import ROOT,Forecaster,play
from pathlib import Path
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',type=Path);args=parser.parse_args()
    if args.policy is None:
        from huggingface_hub import snapshot_download
        args.policy=Path(snapshot_download('lerobot/diffusion_pusht',revision=REV,allow_patterns=['config.json','model.safetensors']))
    pol=ImagePolicy(args.policy);m=Forecaster();body=np.load(ROOT/'data/pusht_pose.npz')['body'];rr=[]
    for start in [42000,42012]:
        rr+=play(pol,m,body,list(range(start,start+12)),'fixed8',0,300);print('300-step horizon',len(rr),flush=True)
    csvfile=ROOT/'data/visual_diffusion_episodes.csv'
    with csvfile.open() as f:short=[r for r in csv.DictReader(f) if r['mode']=='fixed8']
    for x,y in zip(short,rr):
        assert int(x['seed'])==y[0] and y[3]+1e-5>=float(x['max_coverage'])
        if int(x['success']):assert int(x['steps'])==y[4] and y[2]==1
    with (ROOT/'data/visual_diffusion_horizon.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(list(short[0]));wr.writerows(rr)
    print('success',sum(v[2] for v in rr),'mean peak',np.mean([v[3] for v in rr]))
if __name__=='__main__':main()

