"""Independent checks of map signatures, risk floors, dynamic programming and route accounting."""
import csv,hashlib,json,itertools
import numpy as np
from make_tasks import M,ROOT,GA,GU,ORIGIN,distances
from decision import Tree,signature_floor,exhaustive_check
def load(name):
    with (ROOT/'data'/name).open(newline='') as f:return list(csv.DictReader(f))
def verify_signature_classes(signatures):
    indices = {tuple(matrix.flat): index for index, matrix in enumerate(M)}
    products = np.array([[indices[tuple((left @ right).flat)] for right in M] for left in M])
    for signature in signatures:
        stabilizer = np.flatnonzero((signature == signature[:, [0]]).all(axis=0))
        for frame in range(len(M)):
            same = np.flatnonzero((signature == signature[:, [frame]]).all(axis=0))
            assert np.array_equal(same, np.sort(products[stabilizer, frame]))
    return len(signatures) * len(M)

def main():
    ds=np.load(ROOT/'data/tasks.npz');meta=load('tasks.csv');costs=ds['cost'];sig=ds['signature']
    assert len(ds['free'])==len(meta)==2612
    coset_checks=verify_signature_classes(sig)
    # Check fit/validation/test mask-and-goal combinations for exact repetition.
    seen={}
    for r in meta:
        i=int(r['index']);k=hashlib.sha256(ds['free'][i].tobytes()+ds['goal'][i].tobytes()).hexdigest()
        if r['split'] in ('fit','validation','test'):assert k not in seen;seen[k]=r['split']
    checks=load('policy_risks.csv')
    for r in checks:
        loss,bump,prog=(float(r[x]) for x in ['risk','bump','progress'])
        assert abs(loss-(2*(1-prog)+bump))<1e-9
    routes=load('navigation.csv')
    for r in routes:
        assert int(r['success'])==1 and int(r['extra_actions'])==int(r['actions'])-int(meta[int(r['index'])]['distance'])
        assert abs(float(r['charged_excess'])-(int(r['extra_actions'])+2*int(r['bumps'])+.1*int(r['queries'])))<1e-9
    subsets=[r for r in meta if r['split'] in ('test','closure','city')]
    rng=np.random.default_rng(21903);ids=[int(subsets[j]['index']) for j in rng.choice(len(subsets),24,replace=False)]
    error=0
    for i in ids:
        free,goal=ds['free'][i],ds['goal'][i]
        d=distances(free,goal)
        for h in range(8):
            for u,offset in enumerate(GU[h]):assert sig[i,u,h]==free[tuple(ORIGIN+offset)]
            for a,offset in enumerate(GA[h]):
                v=tuple(ORIGIN+offset)
                assert costs[i,h,a]==(1+d[v]-d[15,15] if free[v] else 3)
        for e in [0,.1,.25]:
            tree=Tree(sig[i],costs[i],e);expected=exhaustive_check(sig[i],costs[i],e)
            error=max(error,float(abs(expected-tree.q).max()))
        ro=np.random.default_rng(i);h=ro.permutation(8);u=ro.permutation(12);a=ro.permutation(4)
        r=Tree(sig[i][u][:,h],costs[i][h][:,a])
        assert np.allclose(r.q,Tree(sig[i],costs[i]).q[u])
    assert error<1e-10
    with (ROOT/'data/map_sources.csv').open() as f:
        for r in csv.DictReader(f):
            assert hashlib.sha256((ROOT/'data/maps'/(r['city']+'_0_256.map.zip')).read_bytes()).hexdigest()==r['zip_sha256']
    hand=load('handedness.csv');primary={r['index']:float(r['risk']) for r in checks if r['noise']=='0.1' and r['method']=='adaptive'}
    for r in hand:
        if float(r['reflection_probability'])==.5:assert abs(float(r['adaptive'])-primary[r['index']])<1e-10
    out=dict(tasks=len(meta),frame_classes_checked=coset_checks,scored_one_step_rows=len(checks),navigation_routes=len(routes),oracle_cases=len(ids),oracle_noise_levels=3,max_oracle_error=error,source_map_hashes=10,train_validation_test_duplicate_map_goal_pairs=0)
    (ROOT/'data/verification.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()

