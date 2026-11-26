from experiment import *
def main():
    rng=np.random.default_rng(1729);train,y=sample(rng,1400,5,False)
    w=fit(train,y,'Directional');data,ys=sample(rng,480,5,False);rows=[]
    for edge in [None,(0,1),(1,2),(2,3),(3,4)]:
        yp=(features(data,'Directional',edge)@w).transpose(0,3,1,2)
        loss=1000*np.linalg.norm((yp-ys)[:,-1,1:],axis=-1).mean(axis=1)
        rows.append([str(edge),loss.mean(),*intervals(loss)])
    with (ROOT/'data'/'edge_audit.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['edge_removed','passive_error_mm','low','high']);wr.writerows(rows)
    print(rows)
if __name__=='__main__':main()
