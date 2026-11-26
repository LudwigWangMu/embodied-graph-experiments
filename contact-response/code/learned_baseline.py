"""Train generic four-round graph networks on the same paired physics data."""
import csv,json
from experiment import *
import torch
from torch import nn
torch.set_num_threads(3)
class MessageNet(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.inp=nn.Linear(5,width)
        self.edge=nn.Sequential(nn.Linear(width*2+3,width),nn.SiLU(),nn.Linear(width,width))
        self.node=nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,width))
        self.output=nn.Linear(width,6)
    def forward(self,x,m,v,j):
        b,n,_=x.shape
        impulse=torch.zeros_like(x);impulse[:,0]=j
        z=self.inp(torch.cat([m.unsqueeze(-1),v,impulse],-1))
        dr=x[:,None,:,:]-x[:,:,None,:];d=dr.norm(dim=-1,keepdim=True)
        mask=(d>0)&(d<(2*RADIUS+.1))
        for _ in range(4):
            zi=z[:,:,None,:].expand(-1,-1,n,-1);zj=z[:,None,:,:].expand(-1,n,-1,-1)
            e=self.edge(torch.cat([zi,zj,dr,10*(d-2*RADIUS)],-1))
            z=z+self.node(torch.cat([z,(e*mask).sum(1)],-1))
        return self.output(z).reshape(b,n,3,2).transpose(1,2)
def tensors(x):return [torch.tensor(a,dtype=torch.float32) for a in x]
def infer(net,data):
    with torch.no_grad():return np.concatenate([net(*x).numpy()*.05 for x in zip(*(z.split(96) for z in tensors(data)))])
def main():
    rng=np.random.default_rng(1729);train,y=sample(rng,1400,5,False)
    splits={s:sample(rng,480,n,r) for s,n,r in [('In range',5,False),('Rotated',5,True),('Long chain',8,True)]}
    train_t=tensors(train);target=torch.tensor(y/.05,dtype=torch.float32);rows=[]
    for augment in [False,True]:
        for seed in [11,23,37]:
            torch.manual_seed(seed);net=MessageNet();opt=torch.optim.AdamW(net.parameters(),lr=.002)
            for epoch in range(48):
                for inds in torch.randperm(len(y)).split(100):
                    x,m,v,j=[a[inds] for a in train_t];yy=target[inds]
                    if augment:
                        theta=torch.rand(len(inds))*2*np.pi;cs,ss=theta.cos(),theta.sin()
                        q=torch.stack([cs,-ss,ss,cs],-1).reshape(-1,2,2)
                        x=x@q;v=v@q;j=(j[:,None]@q).squeeze(1);yy=torch.einsum('bhni,bij->bhnj',yy,q)
                    out=net(x,m,v,j)
                    loss=(out-yy).square().mean();opt.zero_grad();loss.backward();opt.step()
            for s,(data,ys) in splits.items():
                pred=infer(net,data);err=np.linalg.norm((ys-pred)[:,:,1:,:],axis=-1).mean(2)*1000
                rows.extend([augment,seed,s,h,err[:,k].mean()] for k,h in enumerate(HORIZONS))
    with (ROOT/'data'/'learned.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['rotation_augmentation','train_seed','split','horizon','passive_error_mm']);w.writerows(rows)
    print('trainable parameters',sum(x.numel() for x in net.parameters()));print([r for r in rows if r[3]==16])
if __name__=='__main__':main()
