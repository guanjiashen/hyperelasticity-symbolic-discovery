from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
	DEFAULT_BIAXIAL_BASE_DIR,
	DEFAULT_DATASET_PATH,
	DEFAULT_EQUATION_PATH,
	DEFAULT_ET_DATASET,
	DEFAULT_FIT_OUTPUT_DIR,
	DEFAULT_MODEL_PATH,
	DEFAULT_PS_DATASET,
	DEFAULT_UT_DATASET,
)


def configure_sampling_arguments(parser: argparse.ArgumentParser) -> None:
	parser.add_argument(
		'--model-path',
		type=Path,
		default=DEFAULT_MODEL_PATH,
		help='Path to the trained invariant NN weights.',
	)
	parser.add_argument(
		'--sampling-mode',
		choices=('grid', 'loadcases'),
		default='grid',
		help='Sampling strategy for NN-generated data: a 2D lambda1-lambda2 grid or UT/PS/ET loadcase paths.',
	)
	parser.add_argument('--lambda1-min', type=float, default=1.0, help='Minimum lambda1.')
	parser.add_argument('--lambda1-max', type=float, default=3.0, help='Maximum lambda1.')
	parser.add_argument('--lambda1-points', type=int, default=81, help='Grid points along lambda1.')
	parser.add_argument('--lambda2-min', type=float, default=1.0, help='Minimum lambda2.')
	parser.add_argument('--lambda2-max', type=float, default=3.0, help='Maximum lambda2.')
	parser.add_argument('--lambda2-points', type=int, default=81, help='Grid points along lambda2.')
	parser.add_argument(
		'--ut-stretch-range',
		nargs=2,
		type=float,
		metavar=('MIN', 'MAX'),
		default=(1.0, 3.0),
		help='Stretch range for uniaxial-tension sampling when --sampling-mode loadcases is used.',
	)
	parser.add_argument('--ut-points', type=int, default=81, help='Sample count for UT loadcase export.')
	parser.add_argument(
		'--ps-stretch-range',
		nargs=2,
		type=float,
		metavar=('MIN', 'MAX'),
		default=(1.0, 3.0),
		help='Stretch range for pure-shear sampling when --sampling-mode loadcases is used.',
	)
	parser.add_argument('--ps-points', type=int, default=81, help='Sample count for PS loadcase export.')
	parser.add_argument(
		'--et-stretch-range',
		nargs=2,
		type=float,
		metavar=('MIN', 'MAX'),
		default=(1.0, 3.0),
		help='Stretch range for equibiaxial-tension sampling when --sampling-mode loadcases is used.',
	)
	parser.add_argument('--et-points', type=int, default=81, help='Sample count for ET loadcase export.')
	parser.add_argument(
		'--uc-stretch-range',
		nargs=2,
		type=float,
		metavar=('MIN', 'MAX'),
		default=(0.9, 1.0),
		help='Stretch range for uniaxial-compression sampling when --sampling-mode loadcases is used.',
	)
	parser.add_argument('--uc-points', type=int, default=0, help='Sample count for UC loadcase export (0 disables).')
	parser.add_argument(
		'--ss-shear-range',
		nargs=2,
		type=float,
		metavar=('MIN', 'MAX'),
		default=(0.0, 0.2),
		help='Amount-of-shear range for simple-shear sampling when --sampling-mode loadcases is used.',
	)
	parser.add_argument('--ss-points', type=int, default=0, help='Sample count for SS loadcase export (0 disables).')
	parser.add_argument(
		'--no-reference-shift',
		action='store_true',
		help='Do not subtract the undeformed-state energy W(lambda1=1, lambda2=1).',
	)


def add_fit_arguments(parser: argparse.ArgumentParser) -> None:
	parser.add_argument(
		'--feature-space',
		choices=('invariants', 'stretches'),
		default='invariants',
		help='Features used by PySR. invariants fits W(I1, I2); stretches fits W(lambda1, lambda2, lambda3).',
	)
	parser.add_argument(
		'--valanis-landel',
		action='store_true',
		help='Constrain stretch-based regression to W = w(lambda1) + w(lambda2) + w(lambda3).',
	)
	parser.add_argument(
		'--output-dir',
		type=Path,
		default=DEFAULT_FIT_OUTPUT_DIR,
		help='Directory where PySR outputs and summaries are written.',
	)
	parser.add_argument('--niterations', type=int, default=60, help='PySR search iterations.')
	parser.add_argument('--population-size', type=int, default=50, help='Population size.')
	parser.add_argument('--populations', type=int, default=10, help='Number of populations.')
	parser.add_argument('--maxsize', type=int, default=18, help='Maximum expression complexity.')
	parser.add_argument(
		'--binary-operators',
		nargs='+',
		default=['+', '-', '*'],
		help='Binary operators allowed in PySR (for example: + - * /).',
	)
	parser.add_argument(
		'--unary-operators',
		nargs='*',
		default=['square'],
		help='Unary operators allowed in PySR.',
	)
	parser.add_argument('--seed', type=int, default=42, help='Random seed for PySR and NumPy.')
	parser.add_argument(
		'--energy-loss-weight',
		type=float,
		default=1.0,
		help='Weight of the NN-derived strain-energy MSE term in the PySR search objective.',
	)
	parser.add_argument(
		'--stress-loss-weight',
		type=float,
		default=0.0,
		help='Weight of the NN-derived nominal-stress MSE term in the PySR search objective. Set to 0 to disable stress loss during search.',
	)
	parser.add_argument(
		'--deterministic',
		action='store_true',
		help='Run PySR in deterministic serial mode.',
	)
	parser.add_argument(
		'--material-package-json',
		type=Path,
		default=None,
		help='Optional material-package JSON output. Defaults to <output-dir>/material_package.json.',
	)
	parser.add_argument(
		'--bulk-model',
		choices=('quadratic',),
		default='quadratic',
		help='Volumetric energy model metadata exported for OpenRadioss integration.',
	)
	parser.add_argument(
		'--bulk-kappa',
		type=float,
		default=0.0,
		help='Bulk modulus parameter exported with the material package.',
	)
	parser.add_argument(
		'--singularity-threshold',
		type=float,
		default=1.0e-3,
		help='Warn when the symbolic-expression denominator comes closer than this threshold to zero on exported samples.',
	)
	parser.add_argument(
		'--simplify-expression',
		action='store_true',
		help='Prune additive invariant terms after PySR and refit the remaining coefficients.',
	)
	parser.add_argument(
		'--simplify-rmse-tolerance',
		type=float,
		default=0.02,
		help='Maximum relative RMSE increase accepted during post-fit pruning.',
	)
	parser.add_argument(
		'--simplify-energy-weight',
		type=float,
		default=0.0,
		help='Normalized NN-energy loss weight used when refitting a simplified expression.',
	)
	parser.add_argument(
		'--simplify-stress-weight',
		type=float,
		default=1.0,
		help='Normalized experimental-stress loss weight used when refitting a simplified expression.',
	)


def add_experiment_prediction_arguments(
	parser: argparse.ArgumentParser,
	*,
	include_equation_file: bool = False,
	include_skip: bool = False,
) -> None:
	if include_equation_file:
		parser.add_argument(
			'--equation-file',
			type=Path,
			default=DEFAULT_EQUATION_PATH,
			help='Text file containing the fitted symbolic expression.',
		)
	parser.add_argument(
		'--ut-dataset',
		type=Path,
		default=DEFAULT_UT_DATASET,
		help='Path to the UT experimental dataset.',
	)
	parser.add_argument(
		'--ps-dataset',
		type=Path,
		default=DEFAULT_PS_DATASET,
		help='Path to the PS experimental dataset.',
	)
	parser.add_argument(
		'--et-dataset',
		type=Path,
		default=DEFAULT_ET_DATASET,
		help='Path to the ET experimental dataset.',
	)
	parser.add_argument(
		'--uc-dataset',
		type=Path,
		default=None,
		help='Optional path to a uniaxial-compression experimental dataset.',
	)
	parser.add_argument(
		'--ss-dataset',
		type=Path,
		default=None,
		help='Optional path to a simple-shear experimental dataset (column 2 stores gamma >= 0).',
	)
	parser.add_argument(
		'--sn-dataset',
		type=Path,
		default=None,
		help='Optional path to a negative simple-shear experimental dataset (column 2 stores gamma <= 0).',
	)
	parser.add_argument(
		'--enable-biaxial',
		action='store_true',
		help='Include biaxial datasets B1..B9 from the selected biaxial folder.',
	)
	parser.add_argument(
		'--biaxial-prefix',
		choices=('BS', 'BL'),
		default='BS',
		help='Biaxial mode prefix: BS for BT_small, BL for BT_large.',
	)
	parser.add_argument(
		'--biaxial-base-dir',
		type=Path,
		default=DEFAULT_BIAXIAL_BASE_DIR,
		help='Base directory containing B1..B9 biaxial folders.',
	)
	if include_skip:
		parser.add_argument(
			'--skip-experiment-prediction',
			action='store_true',
			help='Do not generate symbolic predictions on the experimental datasets after fitting.',
		)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Generate strain energy from the trained invariant NN and fit a symbolic PySR expression.'
	)
	subparsers = parser.add_subparsers(dest='command', required=True)

	export_parser = subparsers.add_parser(
		'export',
		help='Export NN-predicted strain energy and stress targets from either a grid or UT/PS/ET loadcases.',
	)
	configure_sampling_arguments(export_parser)
	export_parser.add_argument('--batch-size', type=int, default=4096, help='Batch size used during NN forward evaluation.')
	export_parser.add_argument(
		'--output-csv',
		type=Path,
		default=DEFAULT_DATASET_PATH,
		help='CSV file that stores the generated energy dataset.',
	)
	export_parser.add_argument(
		'--metadata-json',
		type=Path,
		default=None,
		help='Optional metadata JSON output. Defaults to <output-csv stem>_metadata.json.',
	)

	fit_parser = subparsers.add_parser(
		'fit',
		help='Fit a symbolic strain-energy expression with PySR from an exported CSV dataset.',
	)
	fit_parser.add_argument(
		'--input-csv',
		type=Path,
		default=DEFAULT_DATASET_PATH,
		help='CSV dataset generated by the export step.',
	)
	add_fit_arguments(fit_parser)
	add_experiment_prediction_arguments(fit_parser, include_skip=True)

	pipeline_parser = subparsers.add_parser(
		'pipeline',
		help='Run export and fit in sequence in the current Python environment.',
	)
	configure_sampling_arguments(pipeline_parser)
	pipeline_parser.add_argument('--batch-size', type=int, default=4096, help='Batch size used during NN forward evaluation.')
	pipeline_parser.add_argument(
		'--output-csv',
		type=Path,
		default=DEFAULT_DATASET_PATH,
		help='CSV file that stores the generated energy dataset.',
	)
	pipeline_parser.add_argument(
		'--metadata-json',
		type=Path,
		default=None,
		help='Optional metadata JSON output. Defaults to <output-csv stem>_metadata.json.',
	)
	add_fit_arguments(pipeline_parser)
	add_experiment_prediction_arguments(pipeline_parser, include_skip=True)
	pipeline_parser.add_argument(
		'--export-python',
		type=Path,
		default=None,
		help='Optional Python interpreter for the export step when the current environment lacks torch.',
	)
	pipeline_parser.add_argument(
		'--fit-python',
		type=Path,
		default=None,
		help='Optional Python interpreter for the fit step when the current environment lacks pysr.',
	)

	predict_parser = subparsers.add_parser(
		'predict-experiment',
		help='Use the fitted symbolic expression to predict the experimental stress-stretch datasets.',
	)
	predict_parser.add_argument(
		'--feature-space',
		choices=('invariants', 'stretches'),
		default='invariants',
		help='Feature space used by the symbolic expression.',
	)
	predict_parser.add_argument(
		'--output-dir',
		type=Path,
		default=DEFAULT_FIT_OUTPUT_DIR,
		help='Directory where experiment-prediction outputs are written.',
	)
	add_experiment_prediction_arguments(predict_parser, include_equation_file=True)

	return parser.parse_args()
