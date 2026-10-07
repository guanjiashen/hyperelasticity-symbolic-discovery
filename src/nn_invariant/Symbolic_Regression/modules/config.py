from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
ENTRYPOINT_PATH = SCRIPT_DIR / 'nn_to_symbolic.py'
NN_INVARIANT_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[3]
FFNN_ROOT = WORKSPACE_ROOT / 'FFNN'
OUTPUT_ROOT = NN_INVARIANT_DIR / 'output'
DEFAULT_NN_OUTPUT_DIR = OUTPUT_ROOT / 'NN_output'
DEFAULT_SR_OUTPUT_DIR = OUTPUT_ROOT / 'SR_output'
DEFAULT_MODEL_PATH = DEFAULT_NN_OUTPUT_DIR / 'ffbp_model.pt'
DEFAULT_DATASET_PATH = DEFAULT_SR_OUTPUT_DIR / 'nn_energy_grid.csv'
DEFAULT_FIT_OUTPUT_DIR = DEFAULT_SR_OUTPUT_DIR
DEFAULT_EQUATION_PATH = DEFAULT_FIT_OUTPUT_DIR / 'best_equation.txt'
DEFAULT_UT_DATASET = FFNN_ROOT / 'fitting-data-PK' / 'Treloar_1944' / 'UT' / 'stress_stretch.txt'
DEFAULT_PS_DATASET = FFNN_ROOT / 'fitting-data-PK' / 'Treloar_1944' / 'PS' / 'stress_stretch.txt'
DEFAULT_ET_DATASET = FFNN_ROOT / 'fitting-data-PK' / 'Treloar_1944' / 'ET' / 'stress_stretch.txt'
DEFAULT_BIAXIAL_BASE_DIR = FFNN_ROOT / 'fitting-data-PK' / 'Kawabata_1981' / 'BT_small'
DEFAULT_PYSR_ROOT = WORKSPACE_ROOT / 'SymbolicRegression' / 'PySR-master'
DEFAULT_FFNN_PYTHON = FFNN_ROOT / '.venv' / 'bin' / 'python'
DEFAULT_PYSR_PYTHON = DEFAULT_PYSR_ROOT / '.venv' / 'bin' / 'python'


def add_local_import_paths() -> None:
	paths = [NN_INVARIANT_DIR, DEFAULT_PYSR_ROOT]
	for path in paths:
		path_str = str(path)
		if path.exists() and path_str not in sys.path:
			sys.path.insert(0, path_str)