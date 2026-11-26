"""A shared message network trained to estimate first-probe continuation costs."""
import csv,torch,numpy as np
from torch import nn
from decision import Tree,tied_min
from make_tasks import ROOT
from evaluate import write
torch.set_num_threads(2)
class ProbeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge=nn.Sequential(nn.Linear(6,40),nn.ReLU(),nn.Linear(40,32),nn.ReLU())
        self.out=nn.Sequential(nn.Linear(64,40),nn.ReLU(),nn.Linear(40,1))
    def forward(self,x):
        z=self.edge(x).mean(2)
        return self.out(torch.cat((z,z.mean(1,keepdim=True).expand_as(z)),dim=-1)).squeeze(-1)
def main():
    with (ROOT/'data/tasks.csv').open() as f:meta=list(csv.DictReader(f))
    ds=np.load(ROOT/'data/tasks.npz');a=np.load(ROOT/'data/graph_training.npz');x=torch.from_numpy(a['features']);y=torch.from_numpy(a['targets'])
    sets={k:np.array([i for i,v in enumerate(meta) if v['split']==k]) for k in ['fit','validation','test','city','closure']}
    rows=[];hist=[]
    for seed in (83,113,191):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=ProbeNet();opt=torch.optim.Adam(model.parameters(),lr=.0025)
        best=None;vl=np.inf
        for epoch in range(1,81):
            for ix in np.array_split(rng.permutation(sets['fit']),16):
                estimate=model(x[ix]);loss=((estimate-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():val=float(((model(x[sets['validation']])-y[sets['validation']])**2).mean())
            hist.append((seed,epoch,val))
            if val<vl:vl=val;best={k:v.detach().clone() for k,v in model.state_dict().items()};chosen=epoch
        model.load_state_dict(best);torch.save(best,ROOT/'data'/f'probe_net_{seed}.pt')
        with torch.no_grad():pred=model(x).numpy()
        for i in np.concatenate([sets['test'],sets['closure'],sets['city']]):
            tr=Tree(ds['signature'][i],ds['cost'][i]);f=tied_min(pred[i]);val=tr.score(f,tr.second)
            rows.append((i,meta[i]['split'],meta[i]['source'],meta[i]['pair'],seed,chosen,*val))
    write('learned_risks.csv',['index','split','source','pair','train_seed','epoch','risk','bump','progress'],rows)
    write('validation_loss.csv',['train_seed','epoch','mse'],hist)
if __name__=='__main__':main()

