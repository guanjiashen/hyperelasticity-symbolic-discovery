"""
SI figure: CANN training convergence for the three Fig. 2 benchmarks.

One panel per benchmark (Mooney-Rivlin, Yeoh, Arruda-Boyce): training and
validation loss per L-BFGS outer iteration. For the Yeoh and Arruda-Boyce
cases the twelve restarts (init seeds 0-11) are shown as thin grey training
curves and the selected restart (lowest final training loss) is highlighted.

Input : figs/data/si_cann_convergence.json
        (written by scripts/si_collect_convergence.py, run inside the
        NN_invariant directory of the pipeline repository)
Output: figs/si_cann_convergence.pdf (+ .png preview)

    python scripts/si_cann_convergence.py
"""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figs" / "data" / "si_cann_convergence.json"
OUT = ROOT / "figs" / "si_cann_convergence"

# ------------------------------------------------------------------ style
for f in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):
    for d in (Path("/mnt/c/Windows/Fonts"), Path("C:/Windows/Fonts")):
        if (d / f).exists():
            font_manager.fontManager.addfont(str(d / f))
            break
INK, INK2, INK3, GRID = "#1c1c1c", "#5a5a5a", "#9a9a9a", "#e6e6e6"
C_VAL = "#2f6db3"
C_OTH = "#c9c9c9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": INK3, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.labelcolor": INK, "pdf.fonttype": 42,
})

CASES = [
    ("mooney_rivlin", "Mooney\u2013Rivlin", "single run"),
    ("yeoh", "Yeoh", "12 restarts"),
    ("arruda_boyce", "Arruda\u2013Boyce", "12 restarts"),
]


def main():
    data = json.loads(DATA.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.30, wspace=0.30)

    for j, (key, name, sub) in enumerate(CASES):
        ax = axes[j]
        d = data[key]
        sel = d["selected_seed"]
        runs = d["runs"]
        it = np.arange(1, len(runs[sel]["train"]) + 1)
        for s, r in runs.items():
            if s != sel:
                ax.plot(it, r["train"], "-", color=C_OTH, lw=0.6, zorder=2)
        ax.plot(it, runs[sel]["train"], "-", color=INK, lw=1.1, zorder=4)
        ax.plot(it, runs[sel]["val"], "--", color=C_VAL, lw=0.9, dashes=(3, 1.6), zorder=3)
        ax.set_yscale("log")
        ax.set_xlim(0, len(it))
        ax.set_xlabel("L-BFGS outer iteration", labelpad=1)
        if j == 0:
            ax.set_ylabel(r"stress loss $\mathcal{L}_{\mathrm{NN}}$ (MPa$^2$)", labelpad=2)
        ax.set_title(name, fontsize=8, fontweight="bold", color=INK, pad=8)
        ax.text(0.5, 1.015, sub, transform=ax.transAxes, fontsize=5.8, color=INK2,
                ha="center", va="bottom")
        if sub.startswith("12"):
            ax.text(0.97, 0.94, f"selected: seed {sel}", transform=ax.transAxes,
                    fontsize=5.8, color=INK2, ha="right", va="top")
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        fig.text(ax.get_position().x0 - 0.045, 0.965, "ABC"[j], fontsize=9,
                 fontweight="bold", va="top")

    h = [Line2D([], [], color=INK, ls="-", lw=1.1, label="training loss (selected restart)"),
         Line2D([], [], color=C_VAL, ls="--", dashes=(3, 1.6), lw=0.9,
                label="validation loss (selected restart)"),
         Line2D([], [], color=C_OTH, ls="-", lw=0.6, label="other restarts (training loss)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6,
               bbox_to_anchor=(0.53, 0.0), handlelength=2.2, columnspacing=1.4,
               handletextpad=0.5)

    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print("saved", OUT)


if __name__ == "__main__":
    main()