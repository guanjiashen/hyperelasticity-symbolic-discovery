#!/usr/bin/env bash
# Section 4.6: hyperelastic ball impacting a rigid floor, driven by the
# symbolic law discovered from Meunier data (main-text Fig. 4).
set -euo pipefail

OPENRADIOSS_ROOT="${OPENRADIOSS_ROOT:-/home/guanjs/OpenRadioss-latest-20251211}"
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USERLIB="$OPENRADIOSS_ROOT/tools/userlib/examples/law291_userlib/libraduser_law291.so"
GMSH="${GMSH:-/home/guanjs/.local/bin/gmsh}"

export LD_LIBRARY_PATH="$OPENRADIOSS_ROOT/extlib/hm_reader/linux64:${LD_LIBRARY_PATH:-}"
export RAD_CFG_PATH="$OPENRADIOSS_ROOT/hm_cfg_files"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OMP_STACKSIZE=400m

cd "$CASE_DIR"
STAGE="${1:-all}"
MESH_SIZE="${MESH_SIZE:-3.0}"
PREFIX="${PREFIX:-BALL_IMPACT}"
VELOCITY="${VELOCITY:-5000}"

if [[ "$STAGE" == "all" || "$STAGE" == "starter" ]]; then
  rm -f ${PREFIX}A* ${PREFIX}T0* ${PREFIX}_000*.out ${PREFIX}_*.rst
  if [[ ! -f "${PREFIX}.msh" ]]; then
    "$GMSH" ball.geo -3 -format msh2 -setnumber lc "$MESH_SIZE" -o "${PREFIX}.msh" -v 2
  fi
  python3 build_ball_case.py --mesh "${PREFIX}.msh" --output-dir . \
    --package meunier_csr.flat --velocity "$VELOCITY" --prefix "$PREFIX"
  "$OPENRADIOSS_ROOT/exec/starter_linux64_gf" -dylib "$USERLIB" -i "${PREFIX}_0000.rad"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "engine" ]]; then
  "$OPENRADIOSS_ROOT/exec/engine_linux64_gf" -dylib "$USERLIB" -i "${PREFIX}_0001.rad"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "post" ]]; then
  "$OPENRADIOSS_ROOT/exec/th_to_csv_linux64_gf" "${PREFIX}T01" "${PREFIX}T01"
  mkdir -p "vtk_${PREFIX}"
  for f in ${PREFIX}A*; do
    "$OPENRADIOSS_ROOT/exec/anim_to_vtk_linux64_gf" "$f" > "vtk_${PREFIX}/${f}.vtk"
  done
fi
echo "STAGE $STAGE finished."
