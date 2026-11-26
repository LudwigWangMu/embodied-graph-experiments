"""Keep every map, cost and sensor fixed while varying the known prior on handedness."""
import csv
import numpy as np
from decision import Tree,signature_floor
from evaluate import write
from make_tasks import ROOT
def main():
    ds=np.load(ROOT/'data/tasks.npz')
    with (ROOT/'data/tasks.csv').open() as f:items=list(csv.DictReader(f))
    rows=[]
    for i,r in enumerate(items):
        if r['split'] not in ['test','closure','city']:continue
        s,c=ds['signature'][i],ds['cost'][i]
        for refl in [0,.05,.5]:
            p=np.repeat([(1-refl)/4,refl/4],4)
            tree=Tree(s,c,prior=p)
            a=tree.score(tree.first,tree.second)[0]
            ent=tree.score(tree.info_first,tree.info_second)[0]
            b=signature_floor(s,c,p)
            assert b<=a+1e-10<=ent+2e-10
            rows.append((i,r['split'],r['source'],refl,b,ent,a))
    write('handedness.csv',['index','split','source','reflection_probability','full_floor','entropy','adaptive'],rows)
if __name__=='__main__':main()

