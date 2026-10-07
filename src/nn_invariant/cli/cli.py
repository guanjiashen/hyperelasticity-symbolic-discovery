import argparse
from pathlib import Path


DEFAULT_TRAIN_MODE = 'stress'  # supported: energy, stress
DEFAULT_USE_SYNTHETIC_DATA = False
DEFAULT_SYNTHETIC_MODEL = 'ogden'  # supported: ogden, arruda_boyce, mooney_rivlin
DEFAULT_SYNTHETIC_MODE_RANGES = {
	'UT': (1.0, 3.0),
	'PS': (1.0, 3.0),
	'ET': (1.0, 3.0),
}
DEFAULT_SYNTHETIC_POINT_COUNT = 30
DEFAULT_MOONEY_RIVLIN_C10 = 0.18
DEFAULT_MOONEY_RIVLIN_C01 = 0.02
DEFAULT_ARRUDA_BOYCE_MU = 0.24
DEFAULT_ARRUDA_BOYCE_LAMBDA_M = 3.5
DEFAULT_SYNTHETIC_NOISE_STD = 0.0
DEFAULT_OGDEN_MU_TERMS = (2.0, -2.0, 0.0)
DEFAULT_OGDEN_ALPHA_TERMS = (2.0, -2.0, 0.0)


def _default_dataset_path(mode: str) -> Path:
	workspace_root = Path(__file__).resolve().parents[3]
	return workspace_root / 'data' / 'experimental' / 'Treloar_1944' / mode / 'stress_stretch.txt'


def _default_biaxial_base_dir(size: str) -> Path:
	workspace_root = Path(__file__).resolve().parents[3]
	return workspace_root / 'data' / 'experimental' / 'Kawabata_1981' / size


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--train-mode',
		choices=('energy', 'stress'),
		default=DEFAULT_TRAIN_MODE,
		help='Training objective to use.',
	)
	parser.add_argument('--hidden-neurons', type=int, default=3, help='Number of neurons in the hidden layer.')
	parser.add_argument(
		'--hidden-layers',
		nargs='+',
		type=int,
		default=None,
		metavar='N',
		help='Widths of all hidden layers, e.g. --hidden-layers 16 16 8. Overrides --hidden-neurons.',
	)
	parser.add_argument(
		'--activation',
		choices=('sigmoid', 'relu', 'tanh', 'gelu', 'selu', 'leaky_relu', 'elu', 'softplus'),
		default='softplus',
		help='Hidden-layer activation function.',
	)
	parser.add_argument('--optimizer', choices=('adam', 'lbfgs'), default='lbfgs', help='Training optimizer.')
	parser.add_argument('--learning-rate', type=float, default=0.2, help='Optimizer learning rate.')
	parser.add_argument('--epochs', type=int, default=200, help='Number of outer training epochs.')
	parser.add_argument('--print-every', type=int, default=20, help='Print training metrics every N epochs.')
	parser.add_argument('--lbfgs-max-iter', type=int, default=30, help='Maximum LBFGS iterations per optimizer step.')
	parser.add_argument('--lbfgs-history-size', type=int, default=50, help='LBFGS history size.')
	parser.add_argument('--random-seed', type=int, default=42, help='Random seed for NumPy and PyTorch.')
	parser.add_argument(
		'--init-seed',
		type=int,
		default=None,
		help='Optional separate PyTorch seed for the network initialisation; defaults to --random-seed.',
	)
	parser.add_argument(
		'--output-suffix',
		default='',
		help='Optional suffix for the NN output directory, e.g. 15 writes to NN_output-15.',
	)
	parser.add_argument(
		'--use-synthetic-data',
		action='store_true',
		default=DEFAULT_USE_SYNTHETIC_DATA,
		help='Use synthetic datasets generated from the analytical model.',
	)
	parser.add_argument(
		'--synthetic-model',
		choices=('ogden', 'arruda_boyce', 'mooney_rivlin'),
		default=DEFAULT_SYNTHETIC_MODEL,
		help='Synthetic dataset model to generate.',
	)
	parser.add_argument(
		'--mooney-rivlin-c10',
		type=float,
		default=DEFAULT_MOONEY_RIVLIN_C10,
		help='Mooney-Rivlin C10 coefficient used when --synthetic-model mooney_rivlin.',
	)
	parser.add_argument(
		'--mooney-rivlin-c01',
		type=float,
		default=DEFAULT_MOONEY_RIVLIN_C01,
		help='Mooney-Rivlin C01 coefficient used when --synthetic-model mooney_rivlin.',
	)
	parser.add_argument(
		'--arruda-boyce-mu',
		type=float,
		default=DEFAULT_ARRUDA_BOYCE_MU,
		help='Arruda-Boyce shear modulus mu used when --synthetic-model arruda_boyce.',
	)
	parser.add_argument(
		'--arruda-boyce-lambda-m',
		type=float,
		default=DEFAULT_ARRUDA_BOYCE_LAMBDA_M,
		help='Arruda-Boyce locking stretch lambda_m used when --synthetic-model arruda_boyce.',
	)
	parser.add_argument(
		'--ogden-mu',
		nargs='+',
		type=float,
		default=DEFAULT_OGDEN_MU_TERMS,
		help='Ogden mu terms used when --synthetic-model ogden. Provide one value per term.',
	)
	parser.add_argument(
		'--ogden-alpha',
		nargs='+',
		type=float,
		default=DEFAULT_OGDEN_ALPHA_TERMS,
		help='Ogden alpha terms used when --synthetic-model ogden. Provide one value per term.',
	)
	parser.add_argument(
		'--synthetic-point-count',
		type=int,
		default=DEFAULT_SYNTHETIC_POINT_COUNT,
		help='Number of synthetic samples per mode.',
	)
	parser.add_argument(
		'--synthetic-noise-std',
		type=float,
		default=DEFAULT_SYNTHETIC_NOISE_STD,
		help='Standard deviation of additive Gaussian noise applied to synthetic stresses.',
	)
	parser.add_argument(
		'--synthetic-ut-range',
		nargs=2,
		type=float,
		metavar=('MIN', 'MAX'),
		default=DEFAULT_SYNTHETIC_MODE_RANGES['UT'],
		help='Synthetic UT stretch range.',
	)
	parser.add_argument(
		'--synthetic-ps-range',
		nargs=2,
		type=float,
		metavar=('MIN', 'MAX'),
		default=DEFAULT_SYNTHETIC_MODE_RANGES['PS'],
		help='Synthetic PS stretch range.',
	)
	parser.add_argument(
		'--synthetic-et-range',
		nargs=2,
		type=float,
		metavar=('MIN', 'MAX'),
		default=DEFAULT_SYNTHETIC_MODE_RANGES['ET'],
		help='Synthetic ET stretch range.',
	)
	parser.add_argument(
		'--ut-dataset',
		type=Path,
		default=_default_dataset_path('UT'),
		help='Path to the UT dataset file.',
	)
	parser.add_argument(
		'--ps-dataset',
		type=Path,
		default=_default_dataset_path('PS'),
		help='Path to the PS dataset file.',
	)
	parser.add_argument(
		'--et-dataset',
		type=Path,
		default=_default_dataset_path('ET'),
		help='Path to the ET dataset file.',
	)
	parser.add_argument(
		'--uc-dataset',
		type=Path,
		default=None,
		help='Optional path to a uniaxial-compression dataset file.',
	)
	parser.add_argument(
		'--ss-dataset',
		type=Path,
		default=None,
		help='Optional path to a simple-shear dataset file (column 2 stores gamma >= 0).',
	)
	parser.add_argument(
		'--sn-dataset',
		type=Path,
		default=None,
		help='Optional path to a negative simple-shear dataset file (column 2 stores gamma <= 0).',
	)
	parser.add_argument(
		'--enable-biaxial',
		action='store_true',
		help='Include biaxial datasets (BS1..BS9 or BL1..BL9).',
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
		default=_default_biaxial_base_dir('BT_small'),
		help='Base directory containing B1..B9 biaxial folders.',
	)
	return parser.parse_args()
