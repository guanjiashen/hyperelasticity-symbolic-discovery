#!/usr/bin/env python3
"""Two-panel PNAS Fig. 4: experiment vs. OpenRadioss simulation (coarse mesh only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from compare_meshes import add_numerical_panel, load_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=Path("."))
    parser.add_argument("--experiment-image", type=Path, default=Path("../../figs/meunier_fig12a.png"))
    parser.add_argument("--output", type=Path, default=Path("../../figs/heterogeneous_cann_csr_simulation.png"))
    args = parser.parse_args()

    result = load_case(args.case_dir, "figure10.msh")
    normalization = Normalize(1.0, result["cell_stretch_max"])
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7})

    figure_width, figure_height = 5.0, 5.0
    figure, axes = plt.subplots(1, 2, figsize=(figure_width, figure_height))
    axes[0].imshow(plt.imread(args.experiment_image), cmap="gray")
    axes[0].axis("off")
    add_numerical_panel(axes[1], result, normalization)
    for axis, title in zip(axes, ("(a) Experiment", "(b) Simulation")):
        axis.set_title(title, fontsize=8, pad=4)

    bottom, top = 0.02, 0.94
    left, colorbar_space = 0.015, 0.15
    panel_height = top - bottom
    widths = []
    for axis in axes:
        x_low, x_high = axis.get_xlim()
        y_low, y_high = axis.get_ylim()
        aspect = abs(x_high - x_low) / abs(y_high - y_low)
        widths.append(aspect * panel_height * figure_height / figure_width)
    gap = 1.0 - left - colorbar_space - sum(widths)
    position = left
    for axis, width in zip(axes, widths):
        axis.set_position([position, bottom, width, panel_height])
        position += width + gap
    colorbar_axis = figure.add_axes(
        [position - gap + 0.025, bottom + 0.1 * panel_height, 0.02, 0.8 * panel_height]
    )
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap="jet"), cax=colorbar_axis
    )
    colorbar.set_label(r"Maximum in-plane principal stretch, $\lambda_1$", fontsize=7)
    colorbar.ax.tick_params(labelsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(args.output, result["cell_stretch_max"])


if __name__ == "__main__":
    main()
