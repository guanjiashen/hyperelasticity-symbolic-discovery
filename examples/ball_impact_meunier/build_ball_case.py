#!/usr/bin/env python3
"""Build the Section 4.6 hyperelastic-ball impact case for OpenRadioss.

The Gmsh v2 tetrahedral mesh of a ball of radius R centred on the origin is
translated so that its lowest point sits GAP above the rigid floor z = 0, given
a uniform downward initial velocity, and released.  Units are Mg / mm / s, so
stresses are in MPa, densities in Mg/mm^3, forces in N and energies in mJ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUN_NAME = "BALL_IMPACT"
DENSITY = 1.1e-9          # Mg/mm^3
POISSON = 0.495
GAP = 0.5                 # mm, initial stand-off between ball and floor


def read_msh2(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes: dict[int, tuple[float, float, float]] = {}
    tetra: list[tuple[int, tuple[int, int, int, int]]] = []
    i = 0
    while i < len(lines):
        token = lines[i].strip()
        if token == "$Nodes":
            count = int(lines[i + 1])
            for row in lines[i + 2 : i + 2 + count]:
                fields = row.split()
                nodes[int(fields[0])] = tuple(map(float, fields[1:4]))
            i += count + 2
        elif token == "$Elements":
            count = int(lines[i + 1])
            for row in lines[i + 2 : i + 2 + count]:
                fields = row.split()
                element_type = int(fields[1])
                number_of_tags = int(fields[2])
                connectivity = list(map(int, fields[3 + number_of_tags :]))
                if element_type == 4:
                    tetra.append((int(fields[0]), tuple(connectivity)))
            i += count + 2
        else:
            i += 1
    if not nodes or not tetra:
        raise ValueError("The mesh must contain nodes and TETRA4 elements.")
    return nodes, tetra


def renumber(nodes, tetra):
    """Keep only the nodes used by the tetrahedra and renumber them from 1."""
    used = sorted({node_id for _, conn in tetra for node_id in conn})
    remap = {old: new for new, old in enumerate(used, start=1)}
    new_nodes = {remap[old]: nodes[old] for old in used}
    new_tetra = [
        (index, tuple(remap[node_id] for node_id in conn))
        for index, (_, conn) in enumerate(tetra, start=1)
    ]
    return new_nodes, new_tetra


def write_starter(path: Path, nodes, tetra, package: str, velocity: float, run_name: str):
    node_rows = [
        f"{node_id:10d}{xyz[0]:20.12e}{xyz[1]:20.12e}{xyz[2]:20.12e}"
        for node_id, xyz in sorted(nodes.items())
    ]
    tetra_rows = [
        f"{element_id:10d}{conn[0]:10d}{conn[1]:10d}{conn[2]:10d}{conn[3]:10d}"
        for element_id, conn in tetra
    ]
    variables = "".join(f"{name:<10s}" for name in ("IE", "KE", "ZMOM", "MASS", "HE", "ZCG"))
    text = f"""#RADIOSS STARTER
/BEGIN
{run_name}
      2026         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/TITLE
Hyperelastic ball impacting a rigid floor - CSR Meunier law
/DEF_SOLID
#  I_SOLID    ISMSTR     ICPRE             ITETRA4  ITETRA10      IMAS    IFRAME  ICONTROL
         0         0         0                   3         0         0         0         1
/MAT/USER01/1
Meunier_CSR
#              RHO_I
{DENSITY:20.4e}
# MODE
PACKAGE
# NN_INVARIANT material package
{package}
#                  NU              SIGCUT
{POISSON:20.4f}               1e+30
#               IFORM
                     2
/NODE
{chr(10).join(node_rows)}
/PART/1
Hyperelastic_ball
         1         1         0
/TETRA4/1
{chr(10).join(tetra_rows)}
/PROP/TYPE14/1
Nearly_incompressible_tetrahedra
#   Isolid    Ismstr      Iale     Icpre  Itetra10     Inpts   Itetra4    Iframe                  Dn
        24         0         0         0         0         0         3         0                   0
#                q_a                 q_b                   h            LAMBDA_V                MU_V
                   0                   0                   0                   0                   0
#         deltaT_min            vdef_min            vdef_max             ASP_max             COL_min
                   0                   0                   0                   0                   0
#     Ndir sphpartID  Icontrol
         0         0         1
/GRNOD/PART/100
All_ball_nodes
       1
/INIVEL/TRA/1
Downward_impact_velocity
#                 Vx                  Vy                  Vz   Gnod_id   Skew_id
                   0                   0{-abs(velocity):20.12e}       100         0
#             tstart   sens_ID
                   0         0
/RWALL/PLANE/1
Rigid_floor
#  node_ID     Slide  grnd_ID1  grnd_ID2     Iform
         0         0       100         0         0
#                  d                fric                 ifq
                   0                                       0
#                 XM                  YM                  ZM
                   0                   0                   0
#                XM1                 YM1                 ZM1
                   0                   0                 100
/TH/PART/1
Ball_global_quantities
#      var       var       var       var       var       var
{variables}
#      Obj
         1
/TH/RWALL/2
Floor_normal_force
#      var
FN
#      Obj
         1
/END
"""
    path.write_text(text, encoding="ascii")


def write_engine(path: Path, stop_time: float, animation_dt: float, run_name: str):
    text = f"""#RADIOSS ENGINE
/TITLE
Hyperelastic ball impact - CSR Meunier law
/VERS/2026
/TFILE
1.000000e-005
/RFILE
1000000 0 0
/PRINT/100
/RUN/{run_name}/1
{stop_time:20.12e}
/ANIM/DT
0.000000e+000{animation_dt:20.12e}
/ANIM/ELEM/ENER
/ANIM/ELEM/VONM
/ANIM/BRICK/TENS/STRESS
/ANIM/BRICK/TENS/STRAIN
/ANIM/VECT/VEL
/MON/ON
/DT/NODA/STOP
0.67 0
"""
    path.write_text(text, encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, default=Path("ball.msh"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--package", default="meunier_csr.flat")
    parser.add_argument("--radius", type=float, default=25.0)
    parser.add_argument("--velocity", type=float, default=5.0e3,
                        help="Downward impact speed in mm/s.")
    parser.add_argument("--stop-time", type=float, default=1.5e-2)
    parser.add_argument("--animation-dt", type=float, default=2.5e-4)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nodes, tetra = read_msh2(args.mesh)
    nodes, tetra = renumber(nodes, tetra)

    z_min = min(xyz[2] for xyz in nodes.values())
    shift = GAP - z_min
    nodes = {node_id: (xyz[0], xyz[1], xyz[2] + shift) for node_id, xyz in nodes.items()}

    prefix = args.prefix or RUN_NAME
    write_starter(args.output_dir / f"{prefix}_0000.rad", nodes, tetra,
                  args.package, args.velocity, prefix)
    write_engine(args.output_dir / f"{prefix}_0001.rad", args.stop_time,
                 args.animation_dt, prefix)

    volume = 0.0
    for _, conn in tetra:
        p0, p1, p2, p3 = (nodes[node_id] for node_id in conn)
        a = [p1[k] - p0[k] for k in range(3)]
        b = [p2[k] - p0[k] for k in range(3)]
        c = [p3[k] - p0[k] for k in range(3)]
        det = (a[0] * (b[1] * c[2] - b[2] * c[1])
               - a[1] * (b[0] * c[2] - b[2] * c[0])
               + a[2] * (b[0] * c[1] - b[1] * c[0]))
        volume += abs(det) / 6.0

    metadata = {
        "radius_mm": args.radius,
        "gap_mm": GAP,
        "nodes": len(nodes),
        "tetrahedra": len(tetra),
        "mesh_volume_mm3": volume,
        "mass_Mg": DENSITY * volume,
        "impact_speed_mm_s": args.velocity,
        "initial_kinetic_energy_mJ": 0.5 * DENSITY * volume * args.velocity ** 2,
        "stop_time_s": args.stop_time,
        "animation_dt_s": args.animation_dt,
        "material_package": args.package,
    }
    (args.output_dir / f"{prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
