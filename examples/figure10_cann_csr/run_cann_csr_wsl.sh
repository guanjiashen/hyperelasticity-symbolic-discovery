#!/usr/bin/env bash
# Section 4.4 heterogeneous specimen: CANN-CSR discovery executed by the
# OpenRadioss Starter (TRAIN card of /MAT/USER01) followed by the explicit run.
set -euo pipefail

OPENRADIOSS_ROOT="${OPENRADIOSS_ROOT:?Set OPENRADIOSS_ROOT to your OpenRadioss source/build tree}"
NN_ROOT="/home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant"
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USERLIB="$OPENRADIOSS_ROOT/tools/userlib/examples/law291_userlib/libraduser_law291.so"
NN_PYTHON="$NN_ROOT/.venv/bin/python"

export LD_LIBRARY_PATH="$OPENRADIOSS_ROOT/extlib/hm_reader/linux64:${LD_LIBRARY_PATH:-}"
export RAD_CFG_PATH="$OPENRADIOSS_ROOT/hm_cfg_files"
export RAD_NN_EXPORTER="$OPENRADIOSS_ROOT/tools/nn_invariant/export_openradioss.py"
export RAD_NN_PYTHON="python3"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OMP_STACKSIZE=400m

cd "$CASE_DIR"
STAGE="${1:-all}"

if [[ "$STAGE" == "all" || "$STAGE" == "starter" ]]; then
  rm -f FIG10_CANN_CSRA* FIG10_CANN_CSRT01* FIG10_CANN_CSR_000*.out FIG10_CANN_CSR_*.rst
  gmsh figure10.geo -3 -format msh2 -o figure10.msh
  python3 build_radioss_case.py --mesh figure10.msh --output-dir . --material-mode train
  if [[ ! -f "$USERLIB" ]]; then
    "$OPENRADIOSS_ROOT/tools/userlib/examples/law291_userlib/build.sh"
  fi
  # Starter invokes the CANN-CSR deployment wrapper (training + PySR + pruning + export).
  "$OPENRADIOSS_ROOT/exec/starter_linux64_gf" -dylib "$USERLIB" -i FIG10_CANN_CSR_0000.rad
fi

if [[ "$STAGE" == "all" || "$STAGE" == "engine" ]]; then
  "$OPENRADIOSS_ROOT/exec/engine_linux64_gf" -dylib "$USERLIB" -i FIG10_CANN_CSR_0001.rad
fi

if [[ "$STAGE" == "all" || "$STAGE" == "post" ]]; then
  "$OPENRADIOSS_ROOT/exec/th_to_csv_linux64_gf" FIG10_CANN_CSRT01 FIG10_CANN_CSRT01
  animation_file="$(python3 select_state.py FIG10_CANN_CSRT01.csv 2>selected_state.log)"
  echo "$animation_file" > selected_animation.txt
  "$OPENRADIOSS_ROOT/exec/anim_to_vtk_linux64_gf" "$animation_file" > selected_state.vtk
  "$NN_PYTHON" postprocess.py --mesh figure10.msh --vtk selected_state.vtk \
    --state selected_state.json --experiment-image ../../figs/meunier_fig12a.png \
    --output ../../figs/heterogeneous_cann_csr_results.png \
    --path-output ../../figs/heterogeneous_cann_csr_path.png
fi
echo "STAGE $STAGE finished."
