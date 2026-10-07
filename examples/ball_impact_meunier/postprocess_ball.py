#!/usr/bin/env python3
"""Figures for the Section 4.6 hyperelastic-ball impact study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np

RADIUS = 25.0


# --------------------------------------------------------------------------
# Time history
# --------------------------------------------------------------------------
COLUMN = {"IE": 23, "KE": 24, "ZMOM": 25, "MASS": 26, "HE": 27, "ZCG": 28,
          "IMPULSE_Z": 31}


def read_history(path: Path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    time = data[:, 0]
    mass = data[0, COLUMN["MASS"]]
    momentum = data[:, COLUMN["ZMOM"]]
    history = {
        "t": time,
        "mass": mass,
        "IE": data[:, COLUMN["IE"]],
        "KE": data[:, COLUMN["KE"]],
        "ZCG": data[:, COLUMN["ZCG"]],
        "vz": momentum / mass,
        # The rigid wall stores the accumulated normal impulse; its time
        # derivative is the instantaneous contact force and coincides with the
        # rate of change of the vertical momentum of the ball.
        "Fz": np.gradient(momentum, time),
        "Fz_wall": -np.gradient(data[:, COLUMN["IMPULSE_Z"]], time),
    }
    return history


def history_metrics(history):
    time = history["t"]
    force = history["Fz"]
    contact = np.flatnonzero(force > 1.0e-2)
    peak = int(np.argmax(force))
    return {
        "mass_g": 1.0e3 * history["mass"] * 1.0e3,
        "impact_speed_mm_s": float(-history["vz"][0]),
        "initial_kinetic_energy_mJ": float(history["KE"][0]),
        "peak_contact_force_N": float(force[peak]),
        "time_of_peak_force_ms": float(1.0e3 * time[peak]),
        "peak_strain_energy_mJ": float(history["IE"].max()),
        "time_of_peak_strain_energy_ms": float(1.0e3 * time[np.argmax(history["IE"])]),
        "peak_strain_energy_fraction": float(history["IE"].max() / history["KE"][0]),
        "contact_start_ms": float(1.0e3 * time[contact[0]]),
        "contact_end_ms": float(1.0e3 * time[contact[-1]]),
        "contact_duration_ms": float(1.0e3 * (time[contact[-1]] - time[contact[0]])),
        "centroid_drop_mm": float(history["ZCG"][0] - history["ZCG"].min()),
        "rebound_speed_mm_s": float(history["vz"][-1]),
        "restitution": float(abs(history["vz"][-1] / history["vz"][0])),
        "final_total_energy_mJ": float(history["IE"][-1] + history["KE"][-1]),
        "energy_drift": float((history["IE"][-1] + history["KE"][-1])
                              / history["KE"][0] - 1.0),
        "maximum_absolute_energy_drift": float(np.max(np.abs(
            (history["IE"] + history["KE"]) / history["KE"][0] - 1.0))),
        "final_nontranslational_energy_fraction": float(
            (history["IE"][-1] + history["KE"][-1]
             - 0.5 * history["mass"] * history["vz"][-1]**2)
            / history["KE"][0]),
        "wall_force_vs_momentum_max_rel_diff": float(
            np.max(np.abs(history["Fz_wall"] - history["Fz"]))
            / np.max(np.abs(history["Fz"]))),
    }


# --------------------------------------------------------------------------
# Deformed mid-plane sections
# --------------------------------------------------------------------------
def read_vtk(path: Path):
    tokens = path.read_text(encoding="ascii").split()

    index = tokens.index("TIME")
    time = float(tokens[index + 4])

    index = tokens.index("POINTS")
    count = int(tokens[index + 1])
    points = np.asarray(tokens[index + 3: index + 3 + 3 * count],
                        dtype=float).reshape(-1, 3)

    index = tokens.index("CELLS")
    cell_count = int(tokens[index + 1])
    cursor = index + 3
    cells = []
    for _ in range(cell_count):
        size = int(tokens[cursor])
        cells.append(tuple(int(value) for value in tokens[cursor + 1: cursor + 1 + size]))
        cursor += size + 1

    index = tokens.index("CELL_TYPES")
    cell_types = np.asarray(tokens[index + 2: index + 2 + cell_count], dtype=int)

    index = tokens.index("3DELEM_Von_Mises")
    start = index + 5
    von_mises = np.asarray(tokens[start: start + cell_count], dtype=float)

    keep = cell_types == 10
    tetra = [cells[i] for i in np.flatnonzero(keep)]
    return time, points, tetra, von_mises[keep]


def slice_polygons(points, tetra, values, plane=0.0):
    """Intersect the deformed tetrahedra with the plane y = plane."""
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    polygons, colors = [], []
    for cell, value in zip(tetra, values):
        coordinates = points[list(cell)]
        offsets = coordinates[:, 1] - plane
        if offsets.min() > 0.0 or offsets.max() < 0.0:
            continue
        crossings = []
        for a, b in edges:
            fa, fb = offsets[a], offsets[b]
            if fa == fb:
                continue
            ratio = fa / (fa - fb)
            if -1.0e-12 <= ratio <= 1.0 + 1.0e-12:
                point = coordinates[a] + ratio * (coordinates[b] - coordinates[a])
                crossings.append((point[0], point[2]))
        if len(crossings) < 3:
            continue
        polygon = np.asarray(crossings)
        centre = polygon.mean(axis=0)
        order = np.argsort(np.arctan2(polygon[:, 1] - centre[1], polygon[:, 0] - centre[0]))
        polygons.append(polygon[order])
        colors.append(value)
    return polygons, np.asarray(colors)


def read_reference_nodes(path: Path):
    """Reference nodal coordinates from the Starter deck."""
    coordinates = {}
    active = False
    for row in path.read_text(encoding="ascii").splitlines():
        if row.startswith("/NODE"):
            active = True
            continue
        if active:
            if row.startswith("/") or row.startswith("#"):
                break
            coordinates[int(row[:10])] = (
                float(row[10:30]), float(row[30:50]), float(row[50:70]))
    return coordinates


def invariant_range(vtk_path: Path, starter_path: Path):
    """Principal stretches and invariants reached in a deformed state."""
    _, points, tetra, _ = read_vtk(vtk_path)
    tokens = vtk_path.read_text(encoding="ascii").split()
    index = tokens.index("NODE_ID")
    count = points.shape[0]
    node_ids = np.asarray(tokens[index + 5: index + 5 + count], dtype=int)
    reference_by_id = read_reference_nodes(starter_path)
    reference = np.asarray([reference_by_id.get(node_id, (np.nan,) * 3)
                            for node_id in node_ids])

    stretch_min, stretch_max, first, second = [], [], [], []
    for cell in tetra:
        nodes = list(cell)
        if not np.isfinite(reference[nodes]).all():
            continue
        origin = nodes[0]
        reference_edges = (reference[nodes[1:]] - reference[origin]).T
        current_edges = (points[nodes[1:]] - points[origin]).T
        if abs(np.linalg.det(reference_edges)) < 1.0e-12:
            continue
        gradient = current_edges @ np.linalg.inv(reference_edges)
        eigenvalues = np.linalg.eigvalsh(gradient.T @ gradient)
        stretches = np.sqrt(np.clip(eigenvalues, 1.0e-12, None))
        jacobian = float(np.prod(stretches))
        isochoric = stretches / jacobian ** (1.0 / 3.0)
        stretch_min.append(isochoric.min())
        stretch_max.append(isochoric.max())
        first.append(float(np.sum(isochoric ** 2)))
        second.append(float(np.sum((isochoric[0] * isochoric[1]) ** 2
                                   + (isochoric[1] * isochoric[2]) ** 2
                                   + (isochoric[2] * isochoric[0]) ** 2)))
    return {
        "min_isochoric_stretch": float(np.min(stretch_min)),
        "max_isochoric_stretch": float(np.max(stretch_max)),
        "max_I1bar": float(np.max(first)),
        "max_I2bar": float(np.max(second)),
    }


def plot_sequence(vtk_files, output: Path, history):
    frames = [read_vtk(path) for path in vtk_files]
    sections = [slice_polygons(points, tetra, values)
                for _, points, tetra, values in frames]
    vmax = max(float(colors.max()) for _, colors in sections)
    norm = Normalize(0.0, vmax)

    figure, axes = plt.subplots(1, len(frames), figsize=(2.05 * len(frames), 3.05))
    for axis, (time, _, _, _), (polygons, colors) in zip(axes, frames, sections):
        collection = PolyCollection(polygons, array=colors, cmap="viridis",
                                    norm=norm, edgecolors="face", linewidths=0.25)
        axis.add_collection(collection)
        centroid = np.interp(time, history["t"], history["ZCG"])
        angle = np.linspace(0.0, 2.0 * np.pi, 241)
        axis.plot(RADIUS * np.cos(angle), centroid + RADIUS * np.sin(angle),
                  color="0.55", linestyle=(0, (4, 3)), linewidth=0.7)
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.fill_between([-34.0, 34.0], -5.0, 0.0, color="0.85",
                          hatch="///", edgecolor="0.6", linewidth=0.0)
        axis.set_xlim(-34.0, 34.0)
        axis.set_ylim(-5.0, 58.0)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for side in axis.spines.values():
            side.set_visible(False)
        axis.set_title(rf"$t={1.0e3 * time:.2f}$ ms", fontsize=9, pad=3)

    figure.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.24, wspace=0.02)
    bar_axis = figure.add_axes([0.28, 0.15, 0.44, 0.035])
    bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                          cax=bar_axis, orientation="horizontal")
    bar.set_label("von Mises stress (MPa)", fontsize=8.5, labelpad=2)
    bar.ax.tick_params(labelsize=8)
    figure.savefig(output, dpi=400)
    plt.close(figure)


def plot_history(coarse, fine, output: Path):
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.15))
    coarse_style = dict(color="#1f4e9c", linewidth=1.5)
    fine_style = dict(color="#c62828", linewidth=1.2, linestyle="--")

    axis = axes[0]
    axis.plot(1.0e3 * coarse["t"], coarse["Fz"], label="coarse mesh", **coarse_style)
    axis.plot(1.0e3 * fine["t"], fine["Fz"], label="refined mesh", **fine_style)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Contact force (N)")
    axis.set_xlim(0.0, 6.0)
    axis.legend(frameon=False, fontsize=8.5)

    axis = axes[1]
    axis.plot(1.0e3 * coarse["t"], coarse["KE"], color="#1f4e9c", linewidth=1.5,
              label=r"kinetic $\mathcal{K}$")
    axis.plot(1.0e3 * coarse["t"], coarse["IE"], color="#e08214", linewidth=1.5,
              label=r"strain $\mathcal{U}$")
    axis.plot(1.0e3 * coarse["t"], coarse["KE"] + coarse["IE"], color="0.25",
              linewidth=1.0, linestyle=":", label=r"$\mathcal{K}+\mathcal{U}$")
    axis.plot(1.0e3 * fine["t"], fine["KE"], label="refined mesh", **fine_style)
    axis.plot(1.0e3 * fine["t"], fine["IE"], **fine_style)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Energy (mJ)")
    axis.set_xlim(0.0, 6.0)
    axis.legend(frameon=False, fontsize=7.8, loc="center right",
                bbox_to_anchor=(1.02, 0.42), handlelength=2.2)

    axis = axes[2]
    axis.plot(1.0e3 * coarse["t"], coarse["vz"] * 1.0e-3, **coarse_style)
    axis.plot(1.0e3 * fine["t"], fine["vz"] * 1.0e-3, **fine_style)
    axis.axhline(-coarse["vz"][0] * 1.0e-3, color="0.55", linewidth=0.8,
                 linestyle=(0, (4, 3)))
    axis.annotate(rf"$e={abs(coarse['vz'][-1] / coarse['vz'][0]):.3f}$",
                  xy=(4.6, coarse["vz"][-1] * 1.0e-3), fontsize=9,
                  xytext=(0, -14), textcoords="offset points")
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Centre-of-mass velocity (m/s)")
    axis.set_ylim(-5.6, 5.6)
    axis.set_xlim(0.0, 6.0)

    for label, axis in zip(("(a)", "(b)", "(c)"), axes):
        axis.tick_params(labelsize=9)
        axis.xaxis.label.set_size(9.5)
        axis.yaxis.label.set_size(9.5)
        axis.text(0.5, -0.27, label, transform=axis.transAxes, ha="center",
                  va="top", fontsize=10, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    figure.savefig(output)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=Path("."))
    parser.add_argument("--coarse-prefix", default="BALL_IMPACT")
    parser.add_argument("--fine-prefix", default="BALL_FINE")
    parser.add_argument("--frames", nargs="+", default=["003", "007", "011", "015", "021"])
    parser.add_argument("--figure-dir", type=Path, default=Path("../../figs/meunier_ball_impact"))
    args = parser.parse_args()

    coarse = read_history(args.case_dir / f"{args.coarse_prefix}T01.csv")
    fine = read_history(args.case_dir / f"{args.fine_prefix}T01.csv")

    vtk_files = [args.case_dir / f"vtk_{args.coarse_prefix}"
                 / f"{args.coarse_prefix}A{index}.vtk" for index in args.frames]
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plot_sequence(vtk_files, args.figure_dir / "ball_impact_sequence.pdf", coarse)
    plot_history(coarse, fine, args.figure_dir / "ball_impact_history.pdf")

    metrics = {"coarse": history_metrics(coarse), "fine": history_metrics(fine)}
    metrics["coarse"]["peak_compression_state"] = invariant_range(
        vtk_files[2], args.case_dir / f"{args.coarse_prefix}_0000.rad")
    metrics["coarse"]["sequence_maximum_von_mises_MPa"] = float(max(
        read_vtk(path)[3].max() for path in vtk_files))
    metrics["mesh_difference"] = {
        key: abs(metrics["fine"][key] - metrics["coarse"][key])
             / max(abs(metrics["coarse"][key]), 1.0e-30)
        for key in ("peak_contact_force_N", "contact_duration_ms",
                    "peak_strain_energy_mJ", "centroid_drop_mm", "restitution")
    }
    (args.case_dir / "ball_impact_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
