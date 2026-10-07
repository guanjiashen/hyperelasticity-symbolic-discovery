#!/usr/bin/env python3
"""Convert the Figure-10 Gmsh v2 mesh to a reproducible OpenRadioss case."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


X_MIN, X_MAX = -62.0, 0.0
Y_MIN, Y_MAX = -82.5, 0.0
THICKNESS = 1.75
TARGET_DISPLACEMENT = 65.0
LOAD_DURATION = 0.18
ANIMATION_DT = 0.0025


def read_msh2(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes: dict[int, tuple[float, float, float]] = {}
    tetra: list[tuple[int, tuple[int, int, int, int]]] = []
    penta: list[tuple[int, tuple[int, int, int, int, int, int]]] = []
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
                element_id = int(fields[0])
                element_type = int(fields[1])
                number_of_tags = int(fields[2])
                connectivity = list(map(int, fields[3 + number_of_tags :]))
                if element_type == 4:
                    if len(connectivity) != 4:
                        raise ValueError(f"Unexpected TETRA4 record: {row}")
                    tetra.append((element_id, tuple(connectivity)))
                elif element_type == 6:
                    if len(connectivity) != 6:
                        raise ValueError(f"Unexpected PENTA6 record: {row}")
                    penta.append((element_id, tuple(connectivity)))
            i += count + 2
        else:
            i += 1
    if not nodes or not (tetra or penta):
        raise ValueError("The mesh must contain nodes and first-order solid elements.")
    return nodes, tetra, penta


def ids_at_y(nodes, y_value: float, tolerance: float = 1.0e-7):
    return sorted(node_id for node_id, xyz in nodes.items() if abs(xyz[1] - y_value) < tolerance)


def format_node_group(node_ids):
    rows = []
    for offset in range(0, len(node_ids), 8):
        rows.append("".join(f"{node_id:10d}" for node_id in node_ids[offset : offset + 8]))
    return "\n".join(rows)


def format_th_nodes(node_ids):
    return "\n".join(f"{node_id:10d}{0:10d}" for node_id in node_ids)


def smooth_velocity_function(points: int = 25):
    rows = []
    for index in range(points + 1):
        time = LOAD_DURATION * index / points
        value = math.sin(math.pi * time / LOAD_DURATION)
        rows.append(f"{time:20.12e}{value:20.12e}")
    return "\n".join(rows)


def write_starter(path: Path, nodes, tetra, penta, bottom_ids, top_ids, material_mode: str):
    master_id = max(nodes) + 1
    master_xyz = ((X_MIN + X_MAX) / 2.0, Y_MAX, THICKNESS / 2.0)
    node_rows = [
        f"{node_id:10d}{xyz[0]:20.12e}{xyz[1]:20.12e}{xyz[2]:20.12e}"
        for node_id, xyz in sorted(nodes.items())
    ]
    node_rows.append(
        f"{master_id:10d}{master_xyz[0]:20.12e}{master_xyz[1]:20.12e}{master_xyz[2]:20.12e}"
    )
    tetra_rows = [
        f"{element_id:10d}{conn[0]:10d}{conn[1]:10d}{conn[2]:10d}{conn[3]:10d}"
        for element_id, conn in tetra
    ]
    penta_rows = [
        f"{element_id:10d}{''.join(f'{node_id:10d}' for node_id in conn)}"
        for element_id, conn in penta
    ]
    solid_cards = []
    if tetra_rows:
        solid_cards.append(f"/TETRA4/1\n{chr(10).join(tetra_rows)}")
    if penta_rows:
        solid_cards.append(f"/PENTA6/1\n{chr(10).join(penta_rows)}")
    velocity_scale = TARGET_DISPLACEMENT * math.pi / (2.0 * LOAD_DURATION)
    if material_mode == "eq48":
        material_title = "Equation_4_8"
        material_card = """# A(linear x) B(quadratic x) C(log(1+x)) D(log(1+y))
      0.1067627535     0.01291284071     0.02134404274     0.04810179865
#                  NU              SIGCUT
               0.495               1e+30
#               IFORM
                   2"""
    elif material_mode == "train":
        material_title = "Meunier_CANN_CSR"
        material_card = """# MODE
TRAIN
# NN_INVARIANT config
cann_csr_config.json
# NN_INVARIANT cache
nn_cache
#                  NU              SIGCUT
               0.495               1e+30
#               IFORM
                   2"""
    else:
        material_title = "Meunier_CANN_CSR"
        material_card = """# MODE
PACKAGE
# NN_INVARIANT material package
meunier_csr.flat
#                  NU              SIGCUT
               0.495               1e+30
#               IFORM
                   2"""
    text = f"""#RADIOSS STARTER
/BEGIN
FIG10_CANN_CSR
      2026         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/TITLE
Five-hole silicone coupon with {material_title} material
/DEF_SOLID
#  I_SOLID    ISMSTR     ICPRE             ITETRA4  ITETRA10      IMAS    IFRAME  ICONTROL
         0         0         0                   3         0         0         0         1
/MAT/USER01/1
{material_title}
#              RHO_I
              1.1E-9
{material_card}
/NODE
{chr(10).join(node_rows)}
/BCS/1
Fixed_bottom_face
#  Tra rot   skew_ID  grnod_ID
   111 111         0       100
/GRNOD/NODE/100
Fixed_bottom_nodes
{format_node_group(bottom_ids)}
/BCS/2
Moving_grip_constraints
#  Tra rot   skew_ID  grnod_ID
   101 111         0       101
/GRNOD/NODE/101
Moving_grip_master
{master_id:10d}
/PART/1
Perforated_silicone_coupon
         1         1         0
{chr(10).join(solid_cards)}
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
/FUNCT/1
Smooth_half_sine_grip_velocity
#                  X                   Y
{smooth_velocity_function()}
/RBODY/1
Moving_top_grip
#     RBID     ISENS     NSKEW    ISPHER                MASS   Gnod_id     IKREM      ICOG   Surf_id
{master_id:10d}         0         0         0                   0       102         0         0         0
#                Jxx                 Jyy                 Jzz
                   0                   0                   0
#                Jxy                 Jyz                 Jxz
                   0                   0                   0
#  Ioptoff               Ifail
         0         0         0
/GRNOD/NODE/102
Moving_top_secondary_nodes
{format_node_group(top_ids)}
/IMPVEL/1
Uniform_vertical_grip_motion
#funct_IDT       Dir   skew_ID sensor_ID  grnod_ID  frame_ID     Icoor
         1         Y         0         0       101         0         0
#           Ascale_x            Fscale_Y              Tstart               Tstop
                   1{velocity_scale:20.12e}                   0{LOAD_DURATION:20.12e}
/TH/NODE/3
Fixed_bottom_reaction
#     var1      var2      var3      var4      var5      var6      var7      var8      var9     var10
REACY
#    NODid     Iskew                                           NODname
{format_th_nodes(bottom_ids)}
/END
"""
    path.write_text(text, encoding="ascii")
    return master_id


def write_engine(path: Path, run_duration: float = LOAD_DURATION):
    text = f"""#RADIOSS ENGINE
/TITLE
Five-hole silicone coupon - CANN-CSR
/VERS/2026
/TFILE
1.000000e-004
/RFILE
1000000 0 0
/PRINT/1000
/RUN/FIG10_CANN_CSR/1
{run_duration:20.12e}
/ANIM/DT
0.000000e+000{ANIMATION_DT:20.12e}
/ANIM/ELEM/ENER
/ANIM/ELEM/HOURG
/ANIM/ELEM/VONM
/ANIM/NODA/DMAS
/ANIM/BRICK/TENS/STRESS
/ANIM/BRICK/TENS/STRAIN
/MON/ON
/DT/NODA/STOP
0.67 0
"""
    path.write_text(text, encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, default=Path("figure10.msh"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--material-mode", choices=("package", "eq48", "train"), default="train"
    )
    parser.add_argument(
        "--run-duration", type=float, default=LOAD_DURATION,
        help="Engine stop time; the prescribed loading history remains unchanged.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nodes, tetra, penta = read_msh2(args.mesh)
    bottom_ids = ids_at_y(nodes, Y_MIN)
    top_ids = ids_at_y(nodes, Y_MAX)
    if not bottom_ids or not top_ids:
        raise ValueError("Could not identify the y=-82 mm or y=10 mm grip nodes.")
    master_id = write_starter(
        args.output_dir / "FIG10_CANN_CSR_0000.rad",
        nodes,
        tetra,
        penta,
        bottom_ids,
        top_ids,
        args.material_mode,
    )
    write_engine(
        args.output_dir / "FIG10_CANN_CSR_0001.rad", args.run_duration
    )
    metadata = {
        "geometry_mm": {
            "x_range": [X_MIN, X_MAX],
            "y_range": [Y_MIN, Y_MAX],
            "thickness": THICKNESS,
            "hole_diameter": 20.0,
            "hole_centers": [[-47.5, -21.2], [-14.0, -23.0], [-31.5, -40.5], [-47.5, -59.0], [-14.5, -58.0]],
            "cut_width": 1.5,
        },
        "nodes": len(nodes),
        "tetrahedra": len(tetra),
        "prisms": len(penta),
        "bottom_nodes": len(bottom_ids),
        "top_nodes": len(top_ids),
        "top_node_ids": top_ids,
        "master_node_id": master_id,
        "maximum_prescribed_displacement_mm": TARGET_DISPLACEMENT,
        "load_duration_s": LOAD_DURATION,
        "analysis_end_time_s": args.run_duration,
        "animation_dt_s": ANIMATION_DT,
        "material_mode": args.material_mode,
    }
    (args.output_dir / "mesh_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
