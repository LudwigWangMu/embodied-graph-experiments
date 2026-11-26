"""Plot the tested changes in action commitment and episode horizon on aligned resets."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]


def by_seed(filename, mode):
    with (ROOT / "data" / filename).open(newline="", encoding="utf-8") as stream:
        records = [row for row in csv.DictReader(stream) if row["mode"] == mode]
    indexed = {int(row["seed"]): row for row in records}
    assert len(indexed) == len(records) == 24
    return indexed


def main():
    short = by_seed("visual_diffusion_episodes.csv", "fixed8")
    gated = by_seed("visual_diffusion_episodes.csv", "policy-gate")
    long = by_seed("visual_diffusion_horizon.csv", "fixed8")
    assert short.keys() == gated.keys() == long.keys()
    seeds = sorted(short, key=lambda key: float(short[key]["max_coverage"]))
    fixed = np.array([float(short[key]["max_coverage"]) for key in seeds])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.spines.right": False,
                         "axes.spines.top": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.42), sharey=True)
    x = np.arange(1, 25)
    for ax, alternative, title, name in zip(
        axes, (gated, long),
        ("A   Changing the commitment rule", "B   Extending the episode"),
        ("Calibrated, 120 steps", "Fixed eight, 300 steps"),
    ):
        y = np.array([float(alternative[key]["max_coverage"]) for key in seeds])
        ax.grid(axis="y", linewidth=.45, color="#dde1e4")
        ax.set_axisbelow(True)
        ax.vlines(x, fixed, y, color="#84939b", lw=1.0)
        ax.scatter(x, fixed, color="#007187", facecolors="white",
                   linewidths=.9, s=17, zorder=3)
        ax.scatter(x, y, color="#a34a18", marker="x", linewidths=1.0, s=17, zorder=4)
        ax.set_title(title, loc="left", fontsize=9, fontweight="medium", pad=6)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlim(.3, 24.7)
        ax.set_ylim(-.035, 1.055)
        ax.set_xlabel("Reset rank by fixed-eight overlap")
        handles = [
            Line2D([], [], ls="", marker="o", markerfacecolor="white", color="#007187",
                   markersize=4.4, label="Fixed eight, 120 steps"),
            Line2D([], [], ls="", marker="x", color="#a34a18", markersize=4.5, label=name),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="white",
                  framealpha=1, edgecolor="none", fontsize=6.8, handlelength=.8)
    axes[0].set_ylabel("Maximum goal overlap")
    fig.tight_layout(w_pad=1.35, pad=.25)
    fig.savefig(ROOT / "figures" / "visual_diffusion_pairs.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

