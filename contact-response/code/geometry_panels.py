"""Inspect the exact templates and the independent physical condition cells."""
from matched_geometry import layouts, ROOT
from experiment import RADIUS
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import csv
def main():
    fig,axes=plt.subplots(1,3,figsize=(6.9,2.2))
    for ax,(name,layout) in zip(axes,layouts().items()):
        xy=np.array(layout)*.265
        for i,v in enumerate(xy):
            for z in xy[i+1:]:
                if np.linalg.norm(v-z)-2*RADIUS<.10:
                    ax.plot([v[0],z[0]],[v[1],z[1]],lw=1,c="#495f65",zorder=0)
            ax.add_patch(Circle(v,.078,color="#b66529" if i==0 else "#bbd7df",ec="#284f5b",lw=.7))
        ax.set_aspect("equal");ax.autoscale_view();ax.set_title(name,fontsize=9)
        ax.set_xlim(-.15,1.48);ax.set_ylim(-.53,.53);ax.axis("off")
    fig.tight_layout(pad=.25);fig.savefig(ROOT/"figures/six_shapes.pdf",bbox_inches="tight");plt.close(fig)
    with (ROOT/"data/physical_reset_pairs.csv").open() as f:rows=list(csv.DictReader(f))
    angles=sorted({float(x["angle_magnitude"]) for x in rows});offs=sorted({float(x["offset"]) for x in rows})
    fig,axes=plt.subplots(1,2,figsize=(7,2.1))
    for ax,field,title in zip(axes,["contrast_mm","within_command_split_mm"],["Difference between directions (mm)","Same-command repeat discrepancy (mm)"]):
        z=np.array([[next(float(x[field]) for x in rows if float(x["angle_magnitude"])==a and float(x["offset"])==s) for s in offs] for a in angles])
        ax.imshow(z,aspect="auto",cmap="Blues",vmin=0,vmax=8)
        for i,j in np.ndindex(z.shape):ax.text(j,i,f"{z[i,j]:.1f}",ha="center",va="center",fontsize=6.7,color="white" if z[i,j]>4.7 else "#172c33")
        ax.set_xticks(range(8),[f"{s:.1f}" for s in offs]);ax.set_yticks(range(3),angles)
        ax.tick_params(labelsize=7);ax.set_xlabel("Contact position");ax.set_title(title,fontsize=8.2)
    axes[0].set_ylabel("Angle magnitude (rad)",fontsize=8);fig.tight_layout();fig.savefig(ROOT/"figures/physical_cells.pdf",bbox_inches="tight");plt.close(fig)
if __name__=="__main__":main()

