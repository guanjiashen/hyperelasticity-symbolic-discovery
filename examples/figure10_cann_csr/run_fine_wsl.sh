#!/usr/bin/env bash
# Section 4.4 mesh-refinement check: same exported CANN-CSR material package
# (nn_cache/material.flat, hash 5c4133ef4ad76fa7) on a refined prismatic mesh.
set -euo pipefail

OPENRADIOSS_ROOT="${OPENRADIOSS_ROOT:?Set OPENRADIOSS_ROOT to your OpenRadioss source/build tree}"
NN_ROOT="/home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant"
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINE_DIR="$CASE_DIR/../figure10_cann_csr_fine"
USERLIB="$OPENRADIOSS_ROOT/tools/userlib/examples/law291_userlib/libraduser_law291.so"
NN_PYTHON="$NN_ROOT/.venv/bin/python"

export LD_LIBRARY_PATH="$OPENRADIOSS_ROOT/extlib/hm_reader/linux64:${LD_LIBRARY_PATH:-}"
export RAD_CFG_PATH="$OPENRADIOSS_ROOT/hm_cfg_files"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OMP_STACKSIZE=400m

STAGE="${1:-all}"
mkdir -p "$FINE_DIR"
cd "$FINE_DIR"

if [[ "$STAGE" == "all" || "$STAGE" == "starter" ]]; then
  rm -f FIG10_CANN_CSRA* FIG10_CANN_CSRT01* FIG10_CANN_CSR_000*.out FIG10_CANN_CSR_*.rst
  cp "$CASE_DIR/nn_cache/material.flat" meunier_csr.flat
  cp "$CASE_DIR/zhan_fig11_experiment.csv" .
  gmsh "$CASE_DIR/figure10.geo" -3 -format msh2 \
    -setnumber meshSizeMin 0.75 \
    -setnumber meshSizeMax 1.50 \
    -setnumber curvaturePoints 36 \
    -o figure10_fine.msh
  python3 "$CASE_DIR/build_radioss_case.py" \
    --mesh figure10_fine.msh --output-dir . --material-mode package \
    --run-duration 0.155
  "$OPENRADIOSS_ROOT/exec/starter_linux64_gf" -dylib "$USERLIB" -i FIG10_CANN_CSR_0000.rad
fi

if [[ "$STAGE" == "all" || "$STAGE" == "engine" ]]; then
  "$OPENRADIOSS_ROOT/exec/engine_linux64_gf" -dylib "$USERLIB" -i FIG10_CANN_CSR_0001.rad
fi

if [[ "$STAGE" == "all" || "$STAGE" == "post" ]]; then
  "$OPENRADIOSS_ROOT/exec/th_to_csv_linux64_gf" FIG10_CANN_CSRT01 FIG10_CANN_CSRT01
  animation_file="$(python3 "$CASE_DIR/select_state.py" FIG10_CANN_CSRT01.csv 2>selected_state.log)"
  echo "$animation_file" > selected_animation.txt
  "$OPENRADIOSS_ROOT/exec/anim_to_vtk_linux64_gf" "$animation_file" > selected_state.vtk
  "$NN_PYTHON" "$CASE_DIR/postprocess.py" --mesh figure10_fine.msh --vtk selected_state.vtk \
    --state selected_state.json --experiment-image ../../figs/meunier_fig12a.png \
    --experimental-path-data zhan_fig11_experiment.csv \
    --path-data path_maximum_stretch.csv \
    --output ../../figs/heterogeneous_cann_csr_fine_results.png \
    --path-output ../../figs/heterogeneous_cann_csr_fine_path.png
  cd "$CASE_DIR"
  "$NN_PYTHON" compare_meshes.py --fine-dir ../figure10_cann_csr_fine \
    --output ../../figs/heterogeneous_cann_csr_mesh_comparison.png \
    --path-output ../../figs/heterogeneous_cann_csr_path_mesh_comparison.png \
    --summary ../figure10_cann_csr_fine/mesh_comparison_summary.json
fi
echo "STAGE $STAGE finished."