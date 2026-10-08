"""Render a Gmsh 2.2 mesh with loading and fixed boundary nodes."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def read_msh2(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes = {}
    triangles = []

    node_start = lines.index("$Nodes")
    node_count = int(lines[node_start + 1])
    for row in lines[node_start + 2 : node_start + 2 + node_count]:
        fields = row.split()
        nodes[int(fields[0])] = (float(fields[1]), float(fields[2]))

    element_start = lines.index("$Elements")
    element_count = int(lines[element_start + 1])
    for row in lines[element_start + 2 : element_start + 2 + element_count]:
        fields = row.split()
        element_type = int(fields[1])
        tag_count = int(fields[2])
        if element_type == 2:
            triangles.append(tuple(map(int, fields[3 + tag_count : 6 + tag_count])))
    return nodes, triangles


def main(mesh_path: Path, output_path: Path):
    nodes, triangles = read_msh2(mesh_path)
    edges = set()
    for n1, n2, n3 in triangles:
        edges.update(
            {
                tuple(sorted((n1, n2))),
                tuple(sorted((n2, n3))),
                tuple(sorted((n3, n1))),
            }
        )

    segments = [(nodes[n1], nodes[n2]) for n1, n2 in edges]
    top = [xy for xy in nodes.values() if abs(xy[1]) < 1.0e-8]
    bottom = [xy for xy in nodes.values() if abs(xy[1] + 82.5) < 1.0e-8]

    fig, ax = plt.subplots(figsize=(4.7, 6.15), dpi=300)
    ax.add_collection(LineCollection(segments, colors="#075be8", linewidths=0.22))
    ax.scatter(*zip(*top), s=3.2, c="#d62728", edgecolors="none", zorder=3)
    ax.scatter(*zip(*bottom), s=3.2, c="black", edgecolors="none", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-62.0, 0.0)
    ax.set_ylim(-82.5, 0.0)
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)

    print(
        f"nodes={len(nodes)} triangles={len(triangles)} "
        f"top_nodes={len(top)} bottom_nodes={len(bottom)}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_heterogeneous_mesh.py INPUT.msh OUTPUT.png")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
