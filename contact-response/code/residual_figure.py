from experiment import *
def main():
    with (ROOT/'data'/'hybrid_pairs.csv').open() as f:rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(3.45,2.0));plt.rcParams['pdf.fonttype']=42
    for split,col,style in [('In range','#18617a','-'),('Rotated','#a95a25','--'),('Long chain','#694f83','-.')]:
        ds=np.sort([float(r['corrected_mm'])-float(r['axial_mm']) for r in rows if r['split']==split])
        ax.step(ds,np.arange(1,len(ds)+1)/len(ds),where='post',c=col,ls=style,lw=1.35,label=split)
        print(split,float(np.mean(ds<0)),float(np.quantile(ds,.9)))
    ax.axvline(0,c='#4b5054',lw=.75);ax.set_xlabel('Graph-corrected minus axial error (mm)',fontsize=8);ax.set_ylabel('Fraction of scenes',fontsize=8)
    ax.tick_params(labelsize=7);ax.legend(frameon=False,fontsize=7,loc='lower right');ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=.4);fig.savefig(ROOT/'figures'/'residual.pdf');plt.close(fig)
if __name__=='__main__':main()
