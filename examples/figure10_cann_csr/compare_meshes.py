#!/usr/bin/env python3
"""Compare the coarse and refined five-hole OpenRadioss calculations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
import numpy as np

from postprocess import (
    boundary_segments,
    contiguous_valid_segments,
    front_surface,
    maximum_stretch_from_geometry,
    nodal_projection,
    read_legacy_vtk,
    read_msh2_nodes,
    reference_path_endpoints,
    sample_reference_path,
)


def load_experiment(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return (
        np.asarray([float(row["path_position_mm"]) for row in rows]),
        np.asarray([float(row["maximum_principal_stretch"]) for row in rows]),
    )


def load_case(case_dir: Path, mesh_name: str):
    original_nodes = read_msh2_nodes(case_dir / mesh_name)
    points, cells, node_ids = read_legacy_vtk(case_dir / "selected_state.vtk")
    original = np.asarray(
        [original_nodes.get(int(node_id), [np.nan, np.nan, np.nan]) for node_id in node_ids]
    )
    faces, owners = front_surface(cells, original)
    cell_stretch = maximum_stretch_from_geometry(cells, original, points)
    projected = nodal_projection(faces, owners, original, cell_stretch)
    path_start, path_end = reference_path_endpoints()
    position, _, path_stretch, _ = sample_reference_path(
        faces, original, points, projected, path_start, path_end
    )
    metadata = json.loads((case_dir / "mesh_metadata.json").read_text(encoding="utf-8"))
    state = json.loads((case_dir / "selected_state.json").read_text(encoding="utf-8"))
    return {
        "polygons": [points[list(face), :2] for face in faces],
        "face_stretch": cell_stretch[owners],
        "reference_boundary": boundary_segments(faces, original),
        "deformed_boundary": boundary_segments(faces, points),
        "path_position": position,
        "path_stretch": path_stretch,
        "path_segments": contiguous_valid_segments(np.isfinite(path_stretch)),
        "metadata": metadata,
        "state": state,
        "cell_stretch_max": float(np.nanmax(cell_stretch)),
        "path_stretch_min": float(np.nanmin(path_stretch)),
        "path_stretch_max": float(np.nanmax(path_stretch)),
        "path_stretch_max_s": float(position[np.nanargmax(path_stretch)]),
    }


def add_numerical_panel(axis, result, normalization, cmap="jet", outline_color="0.35"):
    collection = PolyCollection(
        result["polygons"],
        array=result["face_stretch"],
        cmap=cmap,
        norm=normalization,
        edgecolors="none",
    )
    axis.add_collection(collection)
    axis.add_collection(
        LineCollection(
            result["reference_boundary"],
            colors=outline_color,
            linewidths=0.75,
            linestyles="dashdot",
            zorder=3,
        )
    )
    axis.add_collection(
        LineCollection(
            result["deformed_boundary"], colors="black", linewidths=0.65, zorder=4
        )
    )
    axis.autoscale()
    axis.margins(0.004)
    axis.set_aspect("equal")
    axis.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-dir", type=Path, default=Path("."))
    parser.add_argument("--fine-dir", type=Path, default=Path("../figure10_fine"))
    parser.add_argument("--experiment-image", type=Path, default=Path("../../figs/meunier_fig12a.png"))
    parser.add_argument("--experiment-data", type=Path, default=Path("zhan_fig11_experiment.csv"))
    parser.add_argument("--output", type=Path, default=Path("../../figs/heterogeneous_mesh_comparison.png"))
    parser.add_argument("--path-output", type=Path, default=Path("../../figs/heterogeneous_path_mesh_comparison.png"))
    parser.add_argument("--summary", type=Path, default=Path("../figure10_fine/mesh_comparison_summary.json"))
    args = parser.parse_args()

    coarse = load_case(args.coarse_dir, "figure10.msh")
    fine = load_case(args.fine_dir, "figure10_fine.msh")
    experiment_position, experiment_stretch = load_experiment(args.experiment_data)

    color_maximum = max(coarse["cell_stretch_max"], fine["cell_stretch_max"])
    normalization = Normalize(1.0, color_maximum)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7})

    figure_width, figure_height = 8.25, 5.0
    figure, axes = plt.subplots(1, 3, figsize=(figure_width, figure_height))
    axes[0].imshow(plt.imread(args.experiment_image), cmap="gray")
    axes[0].axis("off")
    add_numerical_panel(axes[1], coarse, normalization)
    add_numerical_panel(axes[2], fine, normalization)
    for axis, title in zip(axes, ("(a) Experiment", "(b) Coarse mesh", "(c) Refined mesh")):
        axis.set_title(title, fontsize=8, pad=4)

    # Each panel keeps the aspect ratio of its own content, so the axes boxes are
    # sized to that content and the two gaps between the three panels are equal.
    bottom, top = 0.02, 0.94
    left, colorbar_space = 0.015, 0.105
    panel_height = top - bottom
    aspects = []
    for axis in axes:
        x_low, x_high = axis.get_xlim()
        y_low, y_high = axis.get_ylim()
        aspects.append(abs(x_high - x_low) / abs(y_high - y_low))
    widths = [
        aspect * panel_height * figure_height / figure_width for aspect in aspects
    ]
    gap = (1.0 - left - colorbar_space - sum(widths)) / 2.0
    position = left
    for axis, width in zip(axes, widths):
        axis.set_position([position, bottom, width, panel_height])
        position += width + gap
    colorbar_axis = figure.add_axes(
        [position - gap + 0.018, bottom + 0.1 * panel_height, 0.014, 0.8 * panel_height]
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

    path_figure, path_axis = plt.subplots(figsize=(4.35, 3.30), constrained_layout=True)
    path_axis.scatter(
        experiment_position,
        experiment_stretch,
        s=20,
        facecolors="white",
        edgecolors="0.25",
        linewidths=0.7,
        zorder=4,
        label="Experiment",
    )
    for result, style, color, label in (
        (coarse, "--", "0.45", "Coarse mesh"),
        (fine, "-", "black", "Refined mesh"),
    ):
        for segment_index, (start, stop) in enumerate(result["path_segments"]):
            path_axis.plot(
                result["path_position"][start:stop],
                result["path_stretch"][start:stop],
                linestyle=style,
                color=color,
                linewidth=1.25,
                label=label if segment_index == 0 else None,
                zorder=3 if label == "Refined mesh" else 2,
            )
    path_axis.set_xlabel(r"Position along the reference path, $s$ (mm)")
    path_axis.set_ylabel(r"Maximum principal stretch, $\lambda_1$")
    path_axis.set_xlim(0.0, 62.0)
    path_axis.set_ylim(
        1.0,
        1.04 * max(
            float(np.nanmax(experiment_stretch)),
            coarse["path_stretch_max"],
            fine["path_stretch_max"],
        ),
    )
    path_axis.grid(True, color="0.86", linewidth=0.55, linestyle="--")
    path_axis.legend(frameon=False, loc="upper center")
    args.path_output.parent.mkdir(parents=True, exist_ok=True)
    path_figure.savefig(args.path_output, dpi=400, bbox_inches="tight", facecolor="white")
    path_figure.savefig(args.path_output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(path_figure)

    summary = {
        "common_stretch_color_limits": [1.0, color_maximum],
        "coarse": {
            "nodes": coarse["metadata"]["nodes"],
            "prisms": coarse["metadata"]["prisms"],
            "reaction_N": coarse["state"]["moving_grip_reaction_N"],
            "selected_time_s": coarse["state"]["selected_animation_time_s"],
            "elementwise_stretch_max": coarse["cell_stretch_max"],
            "path_stretch_min": coarse["path_stretch_min"],
            "path_stretch_max": coarse["path_stretch_max"],
            "path_stretch_max_s_mm": coarse["path_stretch_max_s"],
        },
        "fine": {
            "nodes": fine["metadata"]["nodes"],
            "prisms": fine["metadata"]["prisms"],
            "reaction_N": fine["state"]["moving_grip_reaction_N"],
            "selected_time_s": fine["state"]["selected_animation_time_s"],
            "elementwise_stretch_max": fine["cell_stretch_max"],
            "path_stretch_min": fine["path_stretch_min"],
            "path_stretch_max": fine["path_stretch_max"],
            "path_stretch_max_s_mm": fine["path_stretch_max_s"],
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
