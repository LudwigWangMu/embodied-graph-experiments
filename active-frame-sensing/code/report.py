"""Recalculate all table entries and draw vector figures from the archived records."""
from pathlib import Path
from collections import defaultdict
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from make_tasks import ROOT,U,CITIES
from decision import Tree
from evaluate import write

def read(name):
    with (ROOT/'data'/(name+'.csv')).open() as f:return list(csv.DictReader(f))
def avg(rows,k):return np.mean([float(r[k]) for r in rows])
def interval(values):
    x=np.array(values);rng=np.random.default_rng(202611)
    return np.quantile(x[rng.integers(0,len(x),(6000,len(x)))].mean(1),[.025,.975])
def main():
    rr=read('policy_risks');nn=read('learned_risks');nav=read('navigation');fl=read('ambiguity');corr=read('sensor_dependence')
    methods=['blind','random','one','entropy','greedy','fixed-pair','learned','adaptive']
    sets=['test','closure','city'];result=[];ci=[];sens=[]
    for sp in sets:
        select=lambda m:[r for r in (nn if m=='learned' else rr) if r['split']==sp and (m=='learned' or (r['method']==m and float(r['noise'])==.1))]
        for m in methods:result.append((sp,m,*[avg(select(m),z) for z in ['risk','bump','progress']]))
        adapt=select('adaptive');base={r['index']:float(r['risk']) for r in adapt}
        for m in methods[:-1]:
            by=defaultdict(list)
            for r in select(m):by[r['source'] if sp=='city' else r['index']].append(float(r['risk'])-base[r['index']])
            dif=[np.mean(z) for z in by.values()];ci.append((sp,m,np.mean(dif),*interval(dif),len(dif)))
        for e in [0,.1,.25]:
            for m in ['entropy','fixed-pair','adaptive']:
                z=[r for r in rr if r['split']==sp and r['method']==m and float(r['noise'])==e]
                sens.append((sp,e,m,avg(z,'risk')))
    write('summary.csv',['split','method','risk','bump','progress'],result)
    write('paired_intervals.csv',['split','comparator','mean_advantage','ci_low','ci_high','resampling_units'],ci)
    write('noise_sweep.csv',['split','noise','method','risk'],sens)
    summary_nav=[];nav_ci=[]
    for sp in ['test','city']:
        for m in ['motion-only','start','selective']:
            z=[r for r in nav if r['split']==sp and r['method']==m]
            summary_nav.append((sp,m,len(z),*[avg(z,k) for k in ['success','actions','bumps','queries','extra_actions','charged_excess']]))
        paired=defaultdict(dict)
        for r in nav:
            if r['split']==sp:paired[r['index'],r['frame']][r['method']]=float(r['charged_excess'])
        source={r['index']:r['source'] for r in nav if r['split']==sp};unit=defaultdict(list)
        for (i,h),z in paired.items():unit[source[i] if sp=='city' else i].append(z['motion-only']-z['selective'])
        x=[np.mean(q) for q in unit.values()];nav_ci.append((sp,np.mean(x),*interval(x),len(x)))
    write('route_summary.csv',['split','method','runs','success','actions','bumps','queries','extra_actions','charged_excess'],summary_nav)
    write('route_intervals.csv',['split','motion_minus_selective','low','high','resampling_units'],nav_ci)
    plt.rcParams.update({'font.size':9,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    # Highest observed city entropy/adaptive gap; the caption identifies this selection.
    i=json.loads((ROOT/'data/checks.json').read_text())['example'];d=np.load(ROOT/'data/tasks.npz')
    free,goal,s,c=d['free'][i],d['goal'][i],d['signature'][i],d['cost'][i];tr=Tree(s,c)
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.48),gridspec_kw={'width_ratios':[1.05,1.15,.80]})
    ax=axs[0];ax.imshow(free,cmap=ListedColormap(['#34434b','#f2f2e9']),vmin=0,vmax=1)
    ax.scatter(15,15,marker='o',s=35,c='#b4431e',label='Start');ax.scatter(goal[1],goal[0],marker='*',s=95,c='#17808c',edgecolors='#184950',linewidths=.6,label='Goal')
    ax.scatter(15+U[:,1],15+U[:,0],s=9,c='#c68d29',marker='s',label='Probe offsets')
    ax.legend(fontsize=6,frameon=True,loc='lower left',borderpad=.35);ax.set(xticks=[],yticks=[],title='(a) Street-map task')
    ax=axs[1];ax.imshow(s.T,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#34434b','#efe4c4']),vmin=0,vmax=1)
    ax.set(xticks=[0,3,7,11],xticklabels=[1,4,8,12],yticks=[0,3,4,7],yticklabels=['0','3','4','7'],xlabel='Probe index',ylabel='Frame index',title='(b) Free-space bits')
    ax=axs[2];ax.imshow(c,aspect='auto',cmap=ListedColormap(['#e8f0ec','#cacaca','#e6c787','#be775f']),vmin=0,vmax=3)
    for y,x in np.ndindex(c.shape):ax.text(x,y,str(c[y,x]),ha='center',va='center',fontsize=7)
    ax.set(xticks=range(4),xticklabels=['N','E','S','W'],yticks=[],xlabel='Body command',title='(c) Action loss')
    for ax in axs:ax.title.set_fontsize(9)
    fig.tight_layout(pad=.6);fig.savefig(ROOT/'figures/decision_scene.pdf',bbox_inches='tight');plt.close(fig)
    labels={'test':'Procedural','closure':'Local closure','city':'Street maps'}
    fig,axs=plt.subplots(1,2,figsize=(7,2.7),gridspec_kw={'width_ratios':[1,1.2]})
    for z,col,mk in zip(sets,['#146c79','#a65d32','#625280'],['o','s','^']):
        a=[r for r in fl if r['split']==z];vals=np.array([int(r['distinct_signatures']) for r in a]);pcts=[(vals==g).mean()*100 for g in [1,2,4,8]]
        axs[0].plot(range(4),pcts,c=col,marker=mk,ms=4,label=labels[z])
        vals=[r[3] for r in sens if r[0]==z and r[2]=='adaptive'];axs[1].plot([0,.1,.25],vals,color=col,marker=mk,ms=4,label=labels[z])
        axs[1].axhline(avg(a,'floor'),c=col,ls=':',lw=.9)
    axs[0].set(xticks=range(4),xticklabels=['1','2','4','8'],xlabel='Distinct full signatures',ylabel='Tasks (%)',ylim=(-1,100))
    axs[1].set(xlabel='Independent bit-flip probability',ylabel='Two-query Bayes loss',xticks=[0,.1,.25])
    for ax in axs:ax.grid(axis='y',alpha=.18)
    axs[0].legend(frameon=False,fontsize=7);fig.tight_layout();fig.savefig(ROOT/'figures/ambiguity_noise.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,2.65));y=np.arange(6)
    for sp,c,dy,mk in zip(sets,['#146c79','#a65d32','#625280'],[-.19,0,.19],['o','s','^']):
        data=[next(r for r in ci if r[:2]==(sp,m)) for m in ['random','one','entropy','greedy','fixed-pair','learned']]
        vals=np.array([r[2] for r in data]);ax.errorbar(vals,y+dy,xerr=[vals-[r[3] for r in data],[r[4] for r in data]-vals],fmt=mk,ms=4,capsize=2,c=c,label=labels[sp])
    ax.set_yticks(y,['Random pair','Best one','Frame entropy','Greedy loss','Best fixed pair','Learned first']);ax.invert_yaxis();ax.axvline(0,lw=.7,c='grey');ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Mean excess loss above the adaptive optimum');ax.legend(frameon=False,fontsize=7);fig.tight_layout();fig.savefig(ROOT/'figures/paired_risks.pdf',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7,2.7),sharey=True);costfields=[('extra_actions','#34677e','Extra actions'),('bumps','#bf7951','Bump surcharge'),('queries','#8b889b','Query charge')]
    for ax,sp in zip(axs,['test','city']):
        z=[r for r in summary_nav if r[0]==sp];bottom=np.zeros(3);y=np.arange(3)
        for k,col,lab in costfields:
            v=np.array([r[{'extra_actions':7,'bumps':5,'queries':6}[k]] for r in z])*{'extra_actions':1,'bumps':2,'queries':.1}[k]
            ax.barh(y,v,left=bottom,color=col,label=lab);bottom+=v
        ax.set_yticks(y,['Motion only','At start','Selective']);ax.set_title(labels[sp],fontsize=9);ax.set_xlabel('Mean charged excess');ax.set_xlim(0,2.5)
    axs[0].invert_yaxis();fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=7)
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(ROOT/'figures/navigation.pdf',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()

