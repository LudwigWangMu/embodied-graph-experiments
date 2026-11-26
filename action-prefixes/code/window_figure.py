"""Show the first partially authorized test window in each frozen replay."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    with (ROOT/"data/worked_windows.csv").open() as f: rows=list(csv.DictReader(f))
    fig,axs=plt.subplots(1,2,figsize=(6.9,2.5))
    for ax,(key,unit,tol) in zip(axs,[("pusht","px",32),("physical_","mm",2)]):
        r=[t for t in rows if t["corpus"]==key];h=np.arange(1,9)
        ax.plot(h,[float(z["bound"]) for z in r],marker="s",ms=3,c="#165a76",label="Issue-time bound")
        ax.plot(h,[float(z["error"]) for z in r],marker="o",ms=3,c="#995028",label="Realized error")
        ax.axhline(tol,c="#656465",ls="--",lw=.9,label="Tolerance")
        ax.axvspan(6.5,8.15,color="#e7e9eb",zorder=-1)
        ax.set(xlabel="Horizon (steps)",ylabel="Landmark error ("+unit+")",xlim=(.8,8.15),ylim=(0,None));ax.set_xticks(h);ax.grid(axis="y",alpha=.16)
        ax.spines[["top","right"]].set_visible(False);ax.tick_params(labelsize=7)
        ax.set_title("Push-T, episode 6" if key=="pusht" else "Physical push, episode 2",fontsize=9)
    axs[0].legend(fontsize=7,frameon=False);fig.tight_layout();fig.savefig(ROOT/"figures/worked_windows.pdf",bbox_inches="tight");plt.close(fig)
if __name__=="__main__":main()

