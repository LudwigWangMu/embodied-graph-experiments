from experiment import *
def main():
    rng=np.random.default_rng(1729);data,y=sample(rng,1400,5,False);w=fit(data,y,'Directional')
    for n,rot in [(5,False),(5,True),(8,True)]:data,y=sample(rng,480,n,rot)
    yp=predict(data,'Directional',w);tr=y[:,-1];pred=yp[:,-1]
    err=1000*np.linalg.norm(pred-tr,axis=-1);signal=1000*np.linalg.norm(tr,axis=-1)
    with (ROOT/'data'/'distance.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['graph_distance','target_magnitude_mm','mae_mm','fraction_target_above_1mm'])
        for i in range(1,8):wr.writerow([i,signal[:,i].mean(),err[:,i].mean(),(signal[:,i]>1).mean()])
if __name__=='__main__':main()
