#!/usr/bin/env python3
"""Sub-panels of PNAS Fig. 4 (assembled in figs/fig4-deployment.tex).

  figs/fig4_fit.pdf    UT/PS/ET experiment + discovered (CSR) law on one axis
  figs/fig4_field.pdf  reference mesh vs OpenRadioss maximum in-plane stretch
  figs/fig4_path.pdf   stretch along the Zhan et al. path: experiment vs simulation
Only the production (coarse) mesh is shown.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from compare_meshes import add_numerical_panel, load_case, load_experiment
from plot_identification import MODES, load_data, make_eval, nominal_stress

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent.parent / "figs"
import sys
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
import figstyle as fs                                   # shared style of Figs. 2-4
from figstyle import PATHS, INK2, FIELD_CMAP
COLORS = {m: PATHS[m][0] for m in ("UT", "PS", "ET")}
MARKERS = {m: PATHS[m][1] for m in ("UT", "PS", "ET")}

# rcParams come from figstyle (Arial-first sans, 7 pt, pdf.fonttype 42)


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIGS / f"{name}.png", dpi=400, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)


def fit_panel():
    sr = HERE / "nn_cache" / "output" / "SR_output"
    package = json.loads((sr / "material_package.json").read_text())
    dW1 = make_eval(package["constitutive"]["dW_dI1"])
    dW2 = make_eval(package["constitutive"]["dW_dI2"])
    fig, ax = plt.subplots(figsize=(3.15, 1.9))
    for mode in MODES:
        lam_d, p_d = load_data(HERE / "figure15_data" / mode / "stress_stretch.txt")
        lam = np.linspace(1.0, lam_d.max(), 200)
        ax.plot(lam, nominal_stress(mode, lam, dW1, dW2), "-", color=COLORS[mode], lw=1.2,
                label=f"{mode}, discovered law")
        ax.plot(lam_d, p_d, MARKERS[mode], mfc="white", mec=COLORS[mode], ms=3.6, mew=0.8,
                ls="none", label=f"{mode}, experiment")
    ax.set_xlabel(r"stretch $\lambda_1$")
    ax.set_ylabel(r"$P_{11}$ (MPa)")
    ax.set_xlim(1.0, 2.25)
    ax.set_ylim(0.0, 1.32)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fs.style_axes(ax)
    # compact legend: marker = experiment, line = law
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], mfc="white", mec=COLORS[m],
                      ms=3.6, mew=0.8, lw=1.2, label=m) for m in MODES]
    handles += [Line2D([], [], color="0.3", marker="o", mfc="white", mec="0.3", ms=3.6, mew=0.8,
                       ls="none", label="experiment"),
                Line2D([], [], color="0.3", lw=1.2, label="discovered law")]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=6.5, ncol=2, columnspacing=0.9,
              handlelength=1.8, labelspacing=0.3)
    # discovered energy below the axes, styled after the Fig. 3 equation blocks
    # (coefficients of Eq. [eq:meunier] in the main text)
    ax.text(0.5, -0.33,
            r"$\Psi=0.20517\,I_1+5.2619{\times}10^{-2}\,I_2-2.0053{\times}10^{-2}\,I_1^{2}$" "\n"
            r"$\quad-6.7919{\times}10^{-3}\,I_1I_2+2.5884{\times}10^{-3}\,I_1^{3}-0.60164$",
            transform=ax.transAxes, fontsize=6.4, color="0.11",
            ha="center", va="top", multialignment="left", linespacing=1.25)
    save(fig, "fig4_fit")


def reference_mesh():
    """Front-surface triangles of the undeformed mesh, in the same (mm) frame as the simulation."""
    from postprocess import front_surface, read_legacy_vtk, read_msh2_nodes
    nodes = read_msh2_nodes(HERE / "figure10.msh")
    points, cells, node_ids = read_legacy_vtk(HERE / "selected_state.vtk")
    original = np.asarray([nodes.get(int(i), [np.nan] * 3) for i in node_ids])
    faces, _ = front_surface(cells, original)
    return [original[list(f), :2] for f in faces], original


def field_and_path_panels():
    from matplotlib.collections import PolyCollection
    result = load_case(HERE, "figure10.msh")
    norm = Normalize(1.0, result["cell_stretch_max"])

    # Mesh and simulation share ONE axis, so the reference mesh is drawn at
    # exactly the scale of the undeformed outline (dash-dot) in the simulation.
    shift = -76.0  # mm, mesh placed to the left of the simulation
    polys, original = reference_mesh()
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.add_collection(PolyCollection([q + [shift, 0.0] for q in polys], facecolors="white",
                                     edgecolors="#1f5fbf", linewidths=0.25))
    front = original[np.abs(original[:, 2] - 1.75) < 1e-6]
    top = front[np.abs(front[:, 1] - front[:, 1].max()) < 1e-6]
    bot = front[np.abs(front[:, 1] - front[:, 1].min()) < 1e-6]
    ax.plot(top[:, 0] + shift, top[:, 1], "o", color="#d62728", ms=1.1, mew=0, zorder=5)
    ax.plot(bot[:, 0] + shift, bot[:, 1], "o", color="black", ms=1.1, mew=0, zorder=5)
    add_numerical_panel(ax, result, norm, cmap=FIELD_CMAP)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.01)
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=0.84, bottom=0.0, top=0.95)
    ax.apply_aspect()
    box = ax.get_position()
    cax = fig.add_axes([box.x1 + 0.025, box.y0 + 0.1 * box.height, 0.022, 0.8 * box.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), cax=cax)
    cb.set_label(r"max. in-plane principal stretch $\lambda_1$", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    # scale information so that the TikZ panels can be drawn at the same mm scale
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)  # inches
    pad = 0.01
    saved_w, saved_h = tight.width + 2 * pad, tight.height + 2 * pad
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    box = ax.get_position()
    in_per_mm = box.height * fig.get_figheight() / (y1 - y0)
    y_bottom_in = box.y0 * fig.get_figheight() + (-82.5 - y0) * in_per_mm - (tight.y0 - pad)
    ys = np.concatenate([np.asarray(seg)[:, 1] for seg in result["deformed_boundary"]])
    info = {"saved_width_in": saved_w, "saved_height_in": saved_h, "in_per_mm": in_per_mm,
            "specimen_bottom_from_pdf_bottom_in": y_bottom_in,
            "deformed_height_mm": float(ys.max() - ys.min()),
            "deformed_top_mm": float(ys.max()), "deformed_bottom_mm": float(ys.min())}
    (FIGS / "fig4_field_scale.json").write_text(json.dumps(info, indent=2))
    print(info)
    save(fig, "fig4_field")

    s_exp, l_exp = load_experiment(HERE / "zhan_fig11_experiment.csv")
    fig, ax = plt.subplots(figsize=(3.15, 1.9))
    ax.scatter(s_exp, l_exp, s=9, facecolors="none", edgecolors=INK2, linewidths=0.7,
               zorder=4, label="experiment")
    for i, (a, b) in enumerate(result["path_segments"]):
        ax.plot(result["path_position"][a:b], result["path_stretch"][a:b], "-", color=COLORS["UT"],
                lw=1.4, zorder=3, label="simulation" if i == 0 else None)
    ax.set_xlabel(r"position along path $s$ (mm)")
    ax.set_ylabel(r"max. principal stretch $\lambda_1$")
    ax.set_xlim(0.0, 62.0)
    ax.set_ylim(1.0, 1.06 * max(float(np.nanmax(l_exp)), result["path_stretch_max"]))
    fs.style_axes(ax)
    ax.legend(frameon=False, loc="upper center", fontsize=6.5)
    save(fig, "fig4_path")


if __name__ == "__main__":
    fit_panel()
    field_and_path_panels()
    print("done")
