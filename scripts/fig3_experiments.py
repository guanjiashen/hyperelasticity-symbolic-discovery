"""
Fig. 3 of the PNAS manuscript: one procedure fits three measured materials
spanning three orders of magnitude in stiffness. Three columns, one per
material (Treloar's vulcanized rubber, the polymer gel of Bitoh et al.,
the human brain cortex of Budday et al.).
  row 1  : material, term chip, discovered energy, stress RMSE
  row 2  : measured nominal stresses (hollow markers) and the discovered
           law (solid lines); the brain column splits into P11 (tension/
           compression) and P12 (simple shear)
  row 3  : invariant derivatives dPsi/dI1 (solid) and dPsi/dI2 (dashed) versus
           I1 along the tested paths, from the pipeline prediction tables; the
           strip on the x axis marks the tested (calibrated) I1 range
  (The pruning traces of the former row 4 moved to the SI; insets collided
   with the data in the gel and brain panels.)

Input : data/figure_data/fig3_{treloar,yohsuke,brain}_predictions.csv
        data/figure_data/fig3_experiments.json   (written by scripts/fig3_collect.py)
Output: figs/fig3_experiments.pdf (+ .png preview)

    python scripts/fig3_experiments.py
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
from figstyle import INK, INK2, INK3, GRID, PATHS, C_ACCENT as ACCENT, C_REJ, C_BAND

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figure_data"
OUT = ROOT / "figs" / "fig3_experiments"

# ------------------------------------------------------------------ discovered laws
def w_treloar(I1, I2):
    a, b, c = 4.2272e-9, 0.16886, 0.18248
    return 5 * a * I1 ** 4 + b - c * I2 / I1 ** 2, c / I1


def psi_treloar(I1, I2):
    a, b, c, d = 4.2272e-9, 0.16886, 0.18248, -0.68906
    return a * I1 ** 5 + b * I1 + c * I2 / I1 + d


def w_yohsuke(I1, I2):
    a, b, c, d = 1.0700e-3, 0.36133, 7.3011e-3, 0.31033
    return 2 * a * I1 + b - d * I2 / I1 ** 2, c + d / I1


def psi_yohsuke(I1, I2):
    a, b, c, d, e = 1.0700e-3, 0.36133, 7.3011e-3, 0.31033, -1.4258
    return a * I1 ** 2 + b * I1 + c * I2 + d * I2 / I1 + e


BR_A, BR_B, BR_C, BR_D = 1.1462871, 1.636408, 19.059301, 11.9214990958232


def w_brain(I1, I2):
    return BR_A * I2 * (2 * I1 - BR_B) - BR_C, BR_A * I1 * (I1 - BR_B)


def psi_brain(I1, I2):
    return (I1 - BR_B) * (BR_A * I1 * I2 - BR_C) + BR_D


# ------------------------------------------------------------------ kinematics
LAM2 = {"UT": lambda l: l ** -0.5, "UC": lambda l: l ** -0.5,
        "PS": lambda l: np.ones_like(l), "ET": lambda l: l}


def p11(mode, l1, w):
    l2 = LAM2[mode](l1)
    l3 = 1.0 / (l1 * l2)
    I1 = l1 ** 2 + l2 ** 2 + l3 ** 2
    I2 = (l1 * l2) ** 2 + (l2 * l3) ** 2 + (l3 * l1) ** 2
    W1, W2 = w(I1, I2)
    return 2.0 * (l1 - l1 ** -3 * l2 ** -2) * (W1 + l2 ** 2 * W2)


def p12(gamma, w):
    I1 = 3.0 + gamma ** 2
    W1, W2 = w(I1, I1)
    return 2.0 * gamma * (W1 + W2)


# ------------------------------------------------------------------ data
def load(name):
    per = {}
    with open(DATA / f"fig3_{name}_predictions.csv") as fh:
        for r in csv.DictReader(fh):
            d = per.setdefault(r["mode"], {"x": [], "t": [], "p": [], "I1": [], "w1": [], "w2": []})
            d["x"].append(float(r["stretch"]))
            d["t"].append(float(r["stress_target"]))
            d["p"].append(float(r["stress_prediction"]))
            d["I1"].append(float(r["I1"]))
            d["w1"].append(float(r["dW_dI1"]))
            d["w2"].append(float(r["dW_dI2"]))
    for d in per.values():
        for k in d:
            d[k] = np.array(d[k])
    return per


style_axis = fs.style_axes


def scatter(ax, x, y, mode):
    col, mk = PATHS[mode]
    ax.scatter(x, y, s=9, marker=mk, facecolors="none", edgecolors=col,
               linewidths=0.7, zorder=3)


def fit_panel(ax, per, w, modes, unit, offsets=None):
    """Stress-fit panel for the three homogeneous paths of one material."""
    offsets = offsets or {}
    lmax = max(per[m]["x"].max() for m in modes)
    ends = []
    for m in modes:
        col, _ = PATHS[m]
        scatter(ax, per[m]["x"], per[m]["t"], m)
        lf = np.linspace(1.0, per[m]["x"].max(), 200)
        yf = p11(m, lf, w)
        ax.plot(lf, yf, "-", lw=1.4, color=col, zorder=4, solid_capstyle="round")
        ends.append((yf[-1], lf[-1], m, col))
    for yl, xl, m, col in ends:
        dx, dy = offsets.get(m, (0.0, 0.0))
        ax.text(xl + 0.03 * (lmax - 1) + dx, yl + dy, m, fontsize=6, color=col,
                va="center", ha="left", zorder=6)
    ax.set_xlim(1, lmax + 0.17 * (lmax - 1))
    top = max(per[m]["t"].max() for m in modes)
    ax.set_ylim(-0.04 * top, 1.14 * top)
    ax.set_xlabel(r"stretch $\lambda_1$", labelpad=1)
    ax.set_ylabel(rf"$P_{{11}}$ ({unit})", labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    style_axis(ax)


LAM2 = {"UT": lambda l: l ** -0.5, "UC": lambda l: l ** -0.5, "PS": lambda l: 0 * l + 1.0,
        "ET": lambda l: l}


def derivative_strips(fig, cell, per, modes, unit, note):
    """Three stacked strips sharing I1: dPsi/dI1, dPsi/dI2 and the stress-carrying
    modulus mu_eff = 2(Psi,1 + lambda2^2 Psi,2) (2(Psi,1+Psi,2) in simple shear),
    i.e. the factor that multiplies the kinematic term of Eq. [modes].
    Negative half-planes are tinted so that the sign is read at a glance."""
    sub = cell.subgridspec(3, 1, hspace=0.18)
    axes = [fig.add_subplot(sub[i, 0]) for i in range(3)]
    lo = min(per[m]["I1"].min() for m in modes)
    hi = max(per[m]["I1"].max() for m in modes)
    series = {0: [], 1: [], 2: []}
    for m in modes:
        col, _ = PATHS[m]
        o = np.argsort(per[m]["I1"])
        I1, w1, w2, lam = per[m]["I1"][o], per[m]["w1"][o], per[m]["w2"][o], per[m]["x"][o]
        if m == "SS":
            mu = 2 * (w1 + w2)
        else:
            mu = 2 * (w1 + LAM2[m](lam) ** 2 * w2)
        for i, y in enumerate((w1, w2, mu)):
            axes[i].plot(I1, y, "-", lw=1.2, color=col, zorder=4)
            series[i].append(y)
    labels = (r"$\partial\Psi/\partial I_1$", r"$\partial\Psi/\partial I_2$",
              r"$\mu_{\mathrm{eff}}$")
    for i, ax in enumerate(axes):
        ys = np.concatenate(series[i])
        ymin, ymax = min(0.0, ys.min()), max(0.0, ys.max())
        r = (ymax - ymin) or 1.0
        ax.set_ylim(ymin - 0.18 * r, ymax + 0.18 * r)
        ax.axhline(0, color=INK2, lw=0.6, ls="--", dashes=(3, 2), zorder=2)
        ax.axhspan(ymin - 0.18 * r, 0, color="#f6e3e0", lw=0, zorder=0)   # negative half-plane
        ax.set_xlim(lo - 0.02 * (hi - lo), hi + 0.06 * (hi - lo))
        ax.set_ylabel(f"{labels[i]}\n({unit})", labelpad=2, fontsize=6, linespacing=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        style_axis(ax)
        if i < 2:
            ax.tick_params(labelbottom=False)
    axes[2].axvspan(lo, hi, ymin=0.0, ymax=0.06, color="#bdbdbd", lw=0, zorder=1)
    axes[2].text(hi, 0.09, "tested range ", transform=axes[2].get_xaxis_transform(),
                 fontsize=5.2, color=INK3, ha="right", va="bottom", zorder=5)
    axes[2].set_xlabel(r"$I_1$", labelpad=1)
    axes[0].set_title(note, fontsize=6, color=INK2, loc="left", pad=3)
    return axes


def label_end(ax, xy, text, above=True, color=INK):
    ax.annotate(text, xy, xytext=(0, 4 if above else -4), textcoords="offset points",
                fontsize=6, color=color, ha="right", va="bottom" if above else "top", zorder=6)


EQS = {
    "treloar": [r"$\Psi=4.2272{\times}10^{-9}\,I_1^{5}+0.16886\,I_1$",
                r"$\quad +0.18248\,I_2/I_1-0.68906$"],
    "yohsuke": [r"$\Psi=1.0700{\times}10^{-3}\,I_1^{2}+0.36133\,I_1$",
                r"$\quad +7.3011{\times}10^{-3}\,I_2+0.31033\,I_2/I_1$",
                r"$\quad -1.4258$"],
    "brain": [r"$\Psi=1.1463\,I_1^{2}I_2-1.8758\,I_1I_2$",
              r"$\quad -19.059\,I_1+43.110$"],
}


def text_panel(ax, name, eq_lines, rmse_line, trace):
    """Header column: material, pruning chip, discovered law, stress RMSE."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n_raw = trace["pareto"][0]["term_count"]
    n_fin = [p["term_count"] for p in trace["pareto"] if p["accepted"]][-1]
    ax.text(0.0, 1.02, name, fontsize=9, fontweight="bold", color=INK, va="top")
    chip = f"{n_fin} term" + ("s" if n_fin > 1 else "")
    ax.text(0.012, 0.84, chip, fontsize=7.4, color="white", va="top", ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.35", fc=ACCENT, ec="none"))
    ax.text(0.0, 0.65, "discovered", fontsize=7.0, color=ACCENT, va="top",
            fontweight="bold")
    y = 0.52
    for line in eq_lines:
        ax.text(0.0, y, line, fontsize=7.8, color=INK, va="top")
        y -= 0.155
    ax.text(0.0, 0.0, rmse_line, fontsize=7.0, color=INK3, va="top")


def main():
    tre, yoh, bra = load("treloar"), load("yohsuke"), load("brain")
    traces = json.loads((DATA / "fig3_experiments.json").read_text())

    # sanity: analytic laws must reproduce the pipeline predictions
    for name, per, w in (("treloar", tre, w_treloar), ("yohsuke", yoh, w_yohsuke),
                         ("brain", bra, w_brain)):
        err = 0.0
        for m, d in per.items():
            if m in ("SS", "SN"):
                err = max(err, np.abs(p12(d["x"], w) - d["p"]).max())
            else:
                err = max(err, np.abs(p11(m, d["x"], w) - d["p"]).max())
        print(f"{name}: max |analytic - pipeline| = {err:.3e}")

    peaks = {k: max(np.abs(d["t"]).max() for d in per.values())
             for k, per in (("treloar", tre), ("yohsuke", yoh), ("brain", bra))}

    fig = plt.figure(figsize=(7.1, 6.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.85, 1.0, 1.45], hspace=0.45,
                          wspace=0.5, left=0.075, right=0.975, top=0.985,
                          bottom=0.09)

    # ---------------- row 1: material, chip, discovered law, RMSE
    titles = ["Vulcanized rubber", "Polymer gel", "Brain cortex"]
    rmse = [r"stress RMSE $6.37{\times}10^{-2}$ MPa",
            r"stress RMSE $2.92{\times}10^{-2}$ kPa",
            r"stress RMSE $1.89{\times}10^{-2}$ kPa"]
    for j, key in enumerate(("treloar", "yohsuke", "brain")):
        ax = fig.add_subplot(gs[0, j])
        text_panel(ax, titles[j], EQS[key], rmse[j], traces[key])

    # ---------------- row 2 (A-C): stress fits
    ax_a = fig.add_subplot(gs[1, 0])
    fit_panel(ax_a, tre, w_treloar, ("UT", "PS", "ET"), "MPa",
              offsets={"PS": (0.02, -0.18)})

    ax_b = fig.add_subplot(gs[1, 1])
    fit_panel(ax_b, yoh, w_yohsuke, ("UT", "PS", "ET"), "kPa",
              offsets={"PS": (0.03, -0.3)})

    gs_c = gs[1, 2].subgridspec(1, 2, wspace=0.75)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    for m in ("UC", "UT"):
        scatter(ax_c1, bra[m]["x"], bra[m]["t"], m)
        lf = np.linspace(min(bra[m]["x"].min(), 1.0), max(bra[m]["x"].max(), 1.0), 200)
        ax_c1.plot(lf, p11(m, lf, w_brain), "-", lw=1.4, color=PATHS[m][0],
                   zorder=4, solid_capstyle="round")
    ax_c1.text(1.035, 0.33, "UT", fontsize=6, color=PATHS["UT"][0],
               ha="center", va="center", zorder=6)
    ax_c1.text(0.968, -0.75, "UC", fontsize=6, color=PATHS["UC"][0],
               ha="center", va="center", zorder=6)
    ax_c1.set_xlim(0.885, 1.115)
    ax_c1.set_xticks([0.9, 1.0, 1.1])
    ax_c1.set_xlabel(r"stretch $\lambda_1$", labelpad=1)
    ax_c1.set_ylabel(r"$P_{11}$ (kPa)", labelpad=2)
    style_axis(ax_c1)

    ax_c2 = fig.add_subplot(gs_c[0, 1])
    g = np.concatenate([bra["SN"]["x"], bra["SS"]["x"]])
    t = np.concatenate([bra["SN"]["t"], bra["SS"]["t"]])
    scatter(ax_c2, g, t, "SS")
    gf = np.linspace(-0.2, 0.2, 200)
    ax_c2.plot(gf, p12(gf, w_brain), "-", lw=1.4, color=PATHS["SS"][0],
               zorder=4, solid_capstyle="round")
    ax_c2.set_xlim(-0.23, 0.23)
    ax_c2.set_xticks([-0.2, 0.0, 0.2])
    ax_c2.set_xlabel(r"shear $\gamma$", labelpad=1)
    ax_c2.set_ylabel(r"$P_{12}$ (kPa)", labelpad=2)
    ax_c2.text(0.06, 0.9, "SS", fontsize=6, color=PATHS["SS"][0],
               transform=ax_c2.transAxes, ha="left", va="top")
    style_axis(ax_c2)

    # ---------------- row 3 (D-F): sign of the invariant derivatives and of the
    # stress-carrying modulus along the tested paths
    bra_sh = dict(bra)
    bra_sh["SS"] = {k: np.concatenate([bra["SN"][k], bra["SS"][k]]) for k in bra["SS"]}
    derivative_strips(fig, gs[2, 0], tre, ("UT", "PS", "ET"), "MPa",
                      r"both derivatives $>0$: $\Psi$ increases in $I_1$ and $I_2$")
    derivative_strips(fig, gs[2, 1], yoh, ("UT", "PS", "ET"), "kPa",
                      r"both derivatives $>0$: $\Psi$ increases in $I_1$ and $I_2$")
    derivative_strips(fig, gs[2, 2], bra_sh, ("UT", "UC", "SS"), "kPa",
                      r"$\partial\Psi/\partial I_1<0$, yet $\mu_{\mathrm{eff}}>0$ on every path")

    # ---------------- panel letters and shared legend
    cells = ((1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))
    for (r, c), L in zip(cells, "ABCDEF"):
        p = gs[r, c].get_position(fig)
        fs.panel_letter(fig, p.x0 - 0.052, p.y1 + 0.006, L, va="bottom")
    handles = [
        Line2D([], [], color=INK2, marker="o", mfc="none", mew=0.7, ls="", ms=3.2,
               label="experiment"),
        Line2D([], [], color=INK2, ls="-", lw=1.4, label="discovered law"),
        Line2D([], [], color=INK2, ls="-", lw=1.2,
               label=r"along a tested path (colours as in A–C)"),
        plt.Rectangle((0, 0), 1, 1, fc="#f6e3e0", ec="none", label="negative half-plane"),
        Line2D([], [], color=INK2, ls="--", dashes=(3, 2), lw=0.6, label="zero"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=6, bbox_to_anchor=(0.53, 0.0), handlelength=2.0,
               columnspacing=1.0, handletextpad=0.5)

    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
