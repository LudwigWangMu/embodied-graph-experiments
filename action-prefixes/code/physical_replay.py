"""Independent calibration on metric poses in a public physical pushing experiment."""
from experiment import *
MM_PER_UNIT=.25
def main():
    z=np.load(ROOT/'data'/'physical_tracks.npz');rows=[];start=0
    for i,end in enumerate(z['ends']):
        p=z['pusher'][start:end]/MM_PER_UNIT;k=z['landmarks'][start:end]/MM_PER_UNIT
        for t in range(1,len(p)-H,8):
            commands=p[t]+np.arange(1,H+1)[:,None]*(p[t]-p[t-1])
            fs,zs,ys=window_features(k,p,t,commands);rows.append((i,t,fs,zs,ys))
        start=end
    records=tuple(map(np.asarray,zip(*rows)));ranges=(.25,.5,1,2,3,4,6)
    rr=[];cv=[]
    for seed in range(2048,2060):
        results,c=run(records,seed,[v/MM_PER_UNIT for v in ranges],detail_seed=2048,prefix='physical_')
        for row in results:row[2]*=MM_PER_UNIT
        rr+=results;cv+=c
    dump('physical_results',['seed','method','tolerance_mm','episode_mean_prefix','violating_episodes','episodes','wilson_low','wilson_high','episode_mean_abstention'],rr)
    dump('physical_coverage',['seed','method','q','simultaneously_covered','episodes'],cv)
    # Convert detail-file column names and residuals to millimeters.
    for fname,cols in [('physical_anchor_errors',['max_landmark_error_px','graph_scale_px','flat_scale_px']),('physical_test_episodes',['tolerance_px'])]:
        fp=ROOT/'data'/(fname+'.csv')
        with fp.open() as f:recs=list(csv.DictReader(f))
        head=list(recs[0]);values=[[float(r[c])*MM_PER_UNIT if c in cols else r[c] for c in head] for r in recs]
        dump(fname,[c.replace('_px','_mm') for c in head],values)
    print('episodes',len(np.unique(records[0])),'anchors',len(rows),'frames',int(z['ends'][-1]))
    print([r for r in rr if r[0]==2048 and r[2] in (1,2)])
    print(cv[:4])
if __name__=='__main__':main()
