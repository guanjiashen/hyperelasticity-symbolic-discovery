#!/usr/bin/env python3
"""Create the experimental/numerical comparison and path-profile figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
import numpy as np


def read_msh2_nodes(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("$Nodes")
    count = int(lines[start + 1])
    result = {}
    for row in lines[start + 2 : start + 2 + count]:
        fields = row.split()
        result[int(fields[0])] = np.asarray(fields[1:4], dtype=float)
    return result


def read_legacy_vtk(path: Path):
    tokens = path.read_text(encoding="ascii").split()
    point_pos = tokens.index("POINTS")
    point_count = int(tokens[point_pos + 1])
    point_start = point_pos + 3
    points = np.asarray(tokens[point_start : point_start + 3 * point_count], dtype=float).reshape(-1, 3)

    cell_pos = tokens.index("CELLS")
    cell_count = int(tokens[cell_pos + 1])
    cursor = cell_pos + 3
    cells = []
    for _ in range(cell_count):
        size = int(tokens[cursor])
        cursor += 1
        cells.append(tuple(map(int, tokens[cursor : cursor + size])))
        cursor += size

    for index, token in enumerate(tokens):
        if token == "SCALARS" and tokens[index + 1] == "NODE_ID":
            components = int(tokens[index + 3])
            start = index + 6
            node_ids = np.asarray(
                tokens[start : start + point_count * components], dtype=float
            ).reshape(point_count, components)[:, 0].astype(int)
            return points, cells, node_ids
    raise KeyError("NODE_ID")


def front_surface(cells, original_by_vtk_index, z_front=1.75, tolerance=1.0e-6):
    owners = {}
    counts = Counter()
    for cell_index, cell in enumerate(cells):
        if len(cell) == 4:
            faces = (
                (cell[0], cell[1], cell[2]),
                (cell[0], cell[1], cell[3]),
                (cell[0], cell[2], cell[3]),
                (cell[1], cell[2], cell[3]),
            )
        elif len(cell) in (6, 8):
            # anim_to_vtk exports each PENTA6 as either a six-node wedge or a
            # degenerate eight-entry brick with repeated nodes.  Recover each
            # triangular end face geometrically from its three distinct nodes
            # on a common through-thickness layer.
            cell_z = np.asarray([original_by_vtk_index[index][2] for index in cell])
            faces = []
            for z_level in (np.nanmin(cell_z), np.nanmax(cell_z)):
                layer = tuple(
                    dict.fromkeys(
                        cell[index]
                        for index, value in enumerate(cell_z)
                        if abs(value - z_level) < tolerance
                    )
                )
                if len(layer) == 3 and layer not in faces:
                    faces.append(layer)
        else:
            continue
        for face in faces:
            key = tuple(sorted(face))
            counts[key] += 1
            owners[key] = cell_index
    faces = []
    owner_ids = []
    for face, count in counts.items():
        if count != 1:
            continue
        if all(abs(original_by_vtk_index[index][2] - z_front) < tolerance for index in face):
            faces.append(face)
            owner_ids.append(owners[face])
    if not faces:
        raise ValueError(
            "No front-surface solid faces were identified; "
            f"cell-size counts={dict(Counter(map(len, cells)))}."
        )
    return faces, np.asarray(owner_ids, dtype=int)


def maximum_stretch_from_geometry(cells, original, deformed):
    values = np.full(len(cells), np.nan)
    for cell_index, cell in enumerate(cells):
        unique_cell = tuple(dict.fromkeys(cell))
        if len(unique_cell) not in (4, 6):
            continue
        reference = original[list(unique_cell)]
        current = deformed[list(unique_cell)]
        if len(unique_cell) == 4:
            reference_edges = np.column_stack(
                (reference[1] - reference[0], reference[2] - reference[0], reference[3] - reference[0])
            )
            current_edges = np.column_stack(
                (current[1] - current[0], current[2] - current[0], current[3] - current[0])
            )
            deformation_gradient = current_edges @ np.linalg.inv(reference_edges)
        else:
            # A linear wedge does not have one vertex joined to three
            # independent edges.  Fit the cell-centre deformation gradient
            # to all six centred nodal positions in a least-squares sense:
            # x_a - x_bar = F (X_a - X_bar).
            reference_centered = reference - np.mean(reference, axis=0)
            current_centered = current - np.mean(current, axis=0)
            deformation_gradient = np.linalg.lstsq(
                reference_centered, current_centered, rcond=None
            )[0].T
        values[cell_index] = np.linalg.svd(deformation_gradient, compute_uv=False)[0]
    return values


def boundary_segments(faces, coordinates):
    edge_counts = Counter()
    for n1, n2, n3 in faces:
        edge_counts[tuple(sorted((n1, n2)))] += 1
        edge_counts[tuple(sorted((n2, n3)))] += 1
        edge_counts[tuple(sorted((n3, n1)))] += 1
    return [
        (coordinates[n1, :2], coordinates[n2, :2])
        for (n1, n2), count in edge_counts.items()
        if count == 1
    ]


def nodal_projection(faces, owners, coordinates, cell_values):
    """Area-weight cell-centred values onto the front-surface nodes."""
    weighted_values = np.zeros(len(coordinates))
    weights = np.zeros(len(coordinates))
    for face, owner in zip(faces, owners):
        triangle = coordinates[list(face), :2]
        area = 0.5 * abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        for node in face:
            weighted_values[node] += area * cell_values[owner]
            weights[node] += area
    projected = np.full(len(coordinates), np.nan)
    valid = weights > 0.0
    projected[valid] = weighted_values[valid] / weights[valid]
    return projected


def reference_path_endpoints():
    """Return the boundary intersections of the path shown by Zhan et al."""
    # Digitized from Fig. 10(a) of Zhan et al. (2023).  This published path
    # rises slightly from left to right and crosses both upper holes without
    # being constrained to their tabulated centres.
    slope = 0.0233
    intercept = -22.95

    right = np.asarray([0.0, intercept])
    # The left outline joins (-62, 0) to (-61.5, -82.5), hence
    # x = -62 - y/165. Intersect it with y = slope*x + intercept.
    left_x = (-62.0 - intercept / 165.0) / (1.0 + slope / 165.0)
    left = np.asarray([left_x, slope * left_x + intercept])
    return right, left


def sample_reference_path(
    faces,
    reference,
    deformed,
    nodal_values,
    path_start,
    path_end,
    spacing=0.05,
):
    """Sample a reference-coordinate line and map its material points forward."""
    path_vector = path_end - path_start
    path_length = np.linalg.norm(path_vector)
    sample_count = int(round(path_length / spacing)) + 1
    path_position = np.linspace(0.0, path_length, sample_count)
    reference_points = (
        path_start[None, :]
        + path_position[:, None] * path_vector[None, :] / path_length
    )
    values = np.full(sample_count, np.nan)
    current_points = np.full((sample_count, 2), np.nan)

    triangles = np.asarray([reference[list(face), :2] for face in faces])
    current_triangles = np.asarray([deformed[list(face), :2] for face in faces])
    triangle_values = np.asarray([nodal_values[list(face)] for face in faces])
    minimum = np.min(triangles, axis=1)
    maximum = np.max(triangles, axis=1)

    tolerance = 1.0e-9
    for sample_index, point in enumerate(reference_points):
        x_coordinate, y_coordinate = point
        candidates = np.flatnonzero(
            (minimum[:, 0] - tolerance <= x_coordinate)
            & (x_coordinate <= maximum[:, 0] + tolerance)
            & (minimum[:, 1] - tolerance <= y_coordinate)
            & (y_coordinate <= maximum[:, 1] + tolerance)
        )
        for triangle_index in candidates:
            a, b, c = triangles[triangle_index]
            matrix = np.column_stack((b - a, c - a))
            determinant = np.linalg.det(matrix)
            if abs(determinant) < tolerance:
                continue
            coordinates_bc = np.linalg.solve(matrix, point - a)
            barycentric = np.asarray(
                [1.0 - coordinates_bc.sum(), coordinates_bc[0], coordinates_bc[1]]
            )
            if np.min(barycentric) < -tolerance or np.max(barycentric) > 1.0 + tolerance:
                continue
            values[sample_index] = barycentric @ triangle_values[triangle_index]
            current_points[sample_index] = barycentric @ current_triangles[triangle_index]
            break
    return path_position, reference_points, values, current_points


def contiguous_valid_segments(valid):
    """Return half-open index ranges for contiguous True runs."""
    padded = np.r_[False, valid, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts, stops))


def write_path_csv(path, position, reference_points, values, current_points):
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(values)
    segments = contiguous_valid_segments(valid)
    segment_by_index = {
        index: segment_number
        for segment_number, (start, stop) in enumerate(segments, start=1)
        for index in range(start, stop)
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "segment",
                "path_position_mm",
                "reference_x_mm",
                "reference_y_mm",
                "deformed_x_mm",
                "deformed_y_mm",
                "maximum_principal_stretch",
            )
        )
        for index in np.flatnonzero(valid):
            writer.writerow(
                (
                    segment_by_index[index],
                    f"{position[index]:.6f}",
                    f"{reference_points[index, 0]:.6f}",
                    f"{reference_points[index, 1]:.6f}",
                    f"{current_points[index, 0]:.6f}",
                    f"{current_points[index, 1]:.6f}",
                    f"{values[index]:.9f}",
                )
            )


def read_experimental_path(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    position = np.asarray([float(row["path_position_mm"]) for row in rows])
    stretch = np.asarray([float(row["maximum_principal_stretch"]) for row in rows])
    return position, stretch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, default=Path("figure10.msh"))
    parser.add_argument("--vtk", type=Path, default=Path("selected_state.vtk"))
    parser.add_argument("--state", type=Path, default=Path("selected_state.json"))
    parser.add_argument(
        "--experiment-image", type=Path, default=Path("../../figs/meunier_fig12a.png")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("../../figs/heterogeneous_fig10_results.png")
    )
    parser.add_argument(
        "--path-output", type=Path,
        default=Path("../../figs/heterogeneous_path_stretch.png"),
    )
    parser.add_argument(
        "--path-data", type=Path, default=Path("path_maximum_stretch.csv")
    )
    parser.add_argument(
        "--experimental-path-data", type=Path,
        default=Path("zhan_fig11_experiment.csv"),
    )
    args = parser.parse_args()

    original_nodes = read_msh2_nodes(args.mesh)
    points, cells, node_ids = read_legacy_vtk(args.vtk)
    original = np.asarray(
        [original_nodes.get(int(node_id), [np.nan, np.nan, np.nan]) for node_id in node_ids]
    )
    faces, owners = front_surface(cells, original)
    maximum_stretch = maximum_stretch_from_geometry(cells, original, points)

    metadata = json.loads(Path("mesh_metadata.json").read_text(encoding="utf-8"))
    state = json.loads(args.state.read_text(encoding="utf-8"))
    top_ids = set(metadata["top_node_ids"])
    top_indices = [index for index, node_id in enumerate(node_ids) if int(node_id) in top_ids]
    top_displacements = points[top_indices, 1] - original[top_indices, 1]
    all_displacements = np.linalg.norm(points - original, axis=1)

    polygons_deformed = [points[list(face), :2] for face in faces]
    face_stretch = maximum_stretch[owners]
    reference_boundary = boundary_segments(faces, original)
    deformed_boundary = boundary_segments(faces, points)
    path_start, path_end = reference_path_endpoints()
    projected_stretch = nodal_projection(
        faces, owners, original, maximum_stretch
    )
    (
        path_position,
        path_reference_points,
        path_stretch,
        path_current_points,
    ) = sample_reference_path(
        faces,
        original,
        points,
        projected_stretch,
        path_start,
        path_end,
    )
    path_valid = np.isfinite(path_stretch)
    path_segments = contiguous_valid_segments(path_valid)
    write_path_csv(
        args.path_data,
        path_position,
        path_reference_points,
        path_stretch,
        path_current_points,
    )
    experimental_position, experimental_stretch = read_experimental_path(
        args.experimental_path_data
    )

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 5.3), constrained_layout=True,
        gridspec_kw={"width_ratios": [0.78, 1.22]},
    )

    experiment = plt.imread(args.experiment_image)
    axes[0].imshow(experiment, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("(a) Experiment")
    axes[1].set_title("(b) OpenRadioss prediction")

    stretch_vmin = 1.0
    stretch_vmax = max(1.05, float(np.nanmax(face_stretch)))
    stretch_collection = PolyCollection(
        polygons_deformed,
        array=face_stretch,
        cmap="jet",
        norm=Normalize(stretch_vmin, stretch_vmax),
        edgecolors="none",
    )
    axes[1].add_collection(stretch_collection)
    axes[1].add_collection(
        LineCollection(
            reference_boundary, colors="0.35", linewidths=0.9,
            linestyles="dashdot", zorder=3,
        )
    )
    axes[1].add_collection(
        LineCollection(deformed_boundary, colors="black", linewidths=0.75, zorder=4)
    )
    axes[1].autoscale()
    axes[1].margins(0.025)
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    colorbar = fig.colorbar(stretch_collection, ax=axes[1], fraction=0.050, pad=0.025)
    colorbar.set_label(r"Maximum in-plane principal stretch, $\lambda_1$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    path_figure, path_axis = plt.subplots(figsize=(4.25, 3.25), constrained_layout=True)
    path_axis.scatter(
        experimental_position,
        experimental_stretch,
        s=20,
        facecolors="white",
        edgecolors="0.25",
        linewidths=0.7,
        zorder=3,
        label="Experiment",
    )
    for segment_index, (start, stop) in enumerate(path_segments):
        path_axis.plot(
            path_position[start:stop], path_stretch[start:stop],
            color="black", linewidth=1.2,
            label="OpenRadioss" if segment_index == 0 else None,
        )
    path_axis.set_xlabel(r"Position along the reference path, $s$ (mm)")
    path_axis.set_ylabel(r"Maximum principal stretch, $\lambda_1$")
    path_axis.set_xlim(0.0, 62.0)
    path_axis.set_ylim(
        1.0,
        max(
            2.5,
            1.04 * float(np.nanmax(path_stretch)),
            1.04 * float(np.nanmax(experimental_stretch)),
        ),
    )
    path_axis.grid(True, color="0.86", linewidth=0.55, linestyle="--")
    path_axis.legend(frameon=False, loc="upper center")
    args.path_output.parent.mkdir(parents=True, exist_ok=True)
    path_figure.savefig(
        args.path_output, dpi=400, bbox_inches="tight", facecolor="white"
    )
    path_figure.savefig(
        args.path_output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"
    )
    plt.close(path_figure)

    qa = {
        **state,
        "vtk_points": len(points),
        "vtk_cells": len(cells),
        "front_surface_triangles": len(faces),
        "maximum_nodal_displacement_mm": float(np.nanmax(all_displacements)),
        "maximum_principal_stretch_min": float(np.nanmin(maximum_stretch)),
        "maximum_principal_stretch_max": float(np.nanmax(maximum_stretch)),
        "top_vertical_displacement_min_mm": float(np.nanmin(top_displacements)),
        "top_vertical_displacement_max_mm": float(np.nanmax(top_displacements)),
        "top_vertical_displacement_spread_mm": float(
            np.nanmax(top_displacements) - np.nanmin(top_displacements)
        ),
        "colormap": "jet",
        "stretch_color_limits": [stretch_vmin, stretch_vmax],
        "reference_outline": "gray dash-dot",
        "path_reference_coordinates_mm": {
            "start": path_start.tolist(),
            "end": path_end.tolist(),
            "slope_dy_dx": float(
                (path_end[1] - path_start[1])
                / (path_end[0] - path_start[0])
            ),
            "source": "digitized from Fig. 10(a) of Zhan et al. (2023)",
            "position_range": [0.0, float(np.linalg.norm(path_end - path_start))],
        },
        "path_material_segments_mm": [
            [float(path_position[start]), float(path_position[stop - 1])]
            for start, stop in path_segments
        ],
        "path_maximum_principal_stretch_min": float(np.nanmin(path_stretch)),
        "path_maximum_principal_stretch_max": float(np.nanmax(path_stretch)),
        "path_maximum_principal_stretch_s_mm": float(
            path_position[np.nanargmax(path_stretch)]
        ),
        "path_sampling_spacing_mm": float(path_position[1] - path_position[0]),
        "path_projection": "area-weighted element-to-node projection and barycentric interpolation",
        "experimental_path_points": len(experimental_position),
        "experimental_path_source": "digitized from Fig. 11 of Zhan et al. (2023)",
    }
    Path("qa_summary.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
