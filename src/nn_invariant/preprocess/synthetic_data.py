from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

# OGDEN_MU_TERMS = (0.45, 0.02, -0.08)
# OGDEN_ALPHA_TERMS = (1.3, 5.0, -2.0)
# OGDEN_MU_TERMS = (0.24, -0.06, 0.0)
# OGDEN_ALPHA_TERMS = (2.0, -2.0, 0.0)
OGDEN_MU_TERMS = (2.0, -2.0, 0.0)
OGDEN_ALPHA_TERMS = (2.0, -2.0, 0.0)
ARRUDA_BOYCE_MU = 0.24
ARRUDA_BOYCE_LAMBDA_M = 3.5
MOONEY_RIVLIN_C10 = 0.18
MOONEY_RIVLIN_C01 = 0.02
SYNTHETIC_NOISE_STD = 0.0


def _get_mode_stretches(stretch: np.ndarray, mode_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if mode_name == 'UT':
		lambda_1 = stretch
		lambda_2 = stretch ** -0.5
		lambda_3 = stretch ** -0.5
	elif mode_name == 'PS':
		lambda_1 = stretch
		lambda_2 = np.ones_like(stretch)
		lambda_3 = stretch ** -1.0
	elif mode_name == 'ET':
		lambda_1 = stretch
		lambda_2 = stretch
		lambda_3 = stretch ** -2.0
	else:
		raise ValueError(f'Unsupported mode: {mode_name}')

	return lambda_1, lambda_2, lambda_3


def _validate_sampling(stretch_min: float, stretch_max: float, point_count: int) -> None:
	if point_count < 2:
		raise ValueError('point_count must be at least 2')
	if stretch_min <= 0.0:
		raise ValueError('stretch_min must be positive')
	if stretch_max <= stretch_min:
		raise ValueError('stretch_max must be larger than stretch_min')


def _with_optional_noise(
	stress: np.ndarray,
	noise_std: float,
	rng: np.random.Generator | None,
) -> np.ndarray:
	if noise_std <= 0.0:
		return stress

	if rng is None:
		rng = np.random.default_rng(0)
	return stress + rng.normal(0.0, noise_std, size=stress.shape).astype(np.float32)


def _validate_ogden_terms(mu_terms: Sequence[float], alpha_terms: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
	if len(mu_terms) != len(alpha_terms):
		raise ValueError('Ogden requires mu_terms and alpha_terms to have the same length.')
	if len(mu_terms) == 0:
		raise ValueError('Ogden requires at least one term.')

	mu_array = np.asarray(mu_terms, dtype=np.float32)
	alpha_array = np.asarray(alpha_terms, dtype=np.float32)

	inactive_mask = np.isclose(mu_array, 0.0) & np.isclose(alpha_array, 0.0)
	active_mask = ~inactive_mask

	if not np.any(active_mask):
		raise ValueError('At least one active Ogden term is required.')

	if np.any(np.isclose(mu_array[active_mask], 0.0)):
		raise ValueError('For active Ogden terms, mu_i must be non-zero.')
	if np.any(np.isclose(alpha_array[active_mask], 0.0)):
		raise ValueError('For active Ogden terms, alpha_i must be non-zero.')
	if np.any(mu_array[active_mask] * alpha_array[active_mask] <= 0.0):
		raise ValueError('For active Ogden terms, mu_i and alpha_i must have the same sign.')

	return mu_array, alpha_array


def _validate_arruda_boyce_params(mu: float, lambda_m: float) -> tuple[float, float]:
	mu_value = float(mu)
	lambda_m_value = float(lambda_m)

	if mu_value <= 0.0:
		raise ValueError('Arruda-Boyce requires mu > 0.')
	if lambda_m_value <= 1.0:
		raise ValueError('Arruda-Boyce requires lambda_m > 1.')

	return mu_value, lambda_m_value


def mooney_rivlin_to_ogden_terms(c10: float, c01: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
	# Mooney-Rivlin is the N=2 Ogden special case with alpha=(2, -2) and mu=(2C10, -2C01).
	mu_terms = [2.0 * c10, -2.0 * c01]
	alpha_terms = [2.0, -2.0]

	filtered_terms = [
		(mu_value, alpha_value)
		for mu_value, alpha_value in zip(mu_terms, alpha_terms)
		if not np.isclose(mu_value, 0.0)
	]

	if not filtered_terms:
		raise ValueError('At least one non-zero Mooney-Rivlin coefficient is required.')

	mu_filtered, alpha_filtered = zip(*filtered_terms)
	return tuple(float(value) for value in mu_filtered), tuple(float(value) for value in alpha_filtered)


def ogden_nominal_stress(
	stretch: np.ndarray,
	mode_name: str,
	mu_terms: Sequence[float],
	alpha_terms: Sequence[float],
) -> np.ndarray:
	mu_array, alpha_array = _validate_ogden_terms(mu_terms, alpha_terms)
	stress = np.zeros_like(stretch, dtype=np.float32)

	for mu, alpha in zip(mu_array, alpha_array):
		if np.isclose(mu, 0.0) and np.isclose(alpha, 0.0):
			continue
		if mode_name == 'UT':
			stress = stress + mu * (stretch ** (alpha - 1.0) - stretch ** (-0.5 * alpha - 1.0))
		elif mode_name == 'PS':
			stress = stress + mu * (stretch ** (alpha - 1.0) - stretch ** (-alpha - 1.0))
		elif mode_name == 'ET':
			stress = stress + mu * (stretch ** (alpha - 1.0) - stretch ** (-2.0 * alpha - 1.0))
		else:
			raise ValueError(f'Unsupported mode: {mode_name}')

	return stress


def arruda_boyce_nominal_stress(
	stretch: np.ndarray,
	mode_name: str,
	mu: float,
	lambda_m: float,
) -> np.ndarray:
	mu_value, lambda_m_value = _validate_arruda_boyce_params(mu, lambda_m)
	lambda_1, lambda_2, lambda_3 = _get_mode_stretches(stretch, mode_name)
	i1 = lambda_1 ** 2 + lambda_2 ** 2 + lambda_3 ** 2

	inverse_lambda_m2 = 1.0 / (lambda_m_value ** 2)
	inverse_lambda_m4 = inverse_lambda_m2 ** 2
	inverse_lambda_m6 = inverse_lambda_m2 ** 3
	inverse_lambda_m8 = inverse_lambda_m2 ** 4

	dwd_i1 = mu_value * (
		0.5
		+ 0.1 * i1 * inverse_lambda_m2
		+ (11.0 / 350.0) * (i1 ** 2) * inverse_lambda_m4
		+ (19.0 / 1750.0) * (i1 ** 3) * inverse_lambda_m6
		+ (519.0 / 134750.0) * (i1 ** 4) * inverse_lambda_m8
	)

	if mode_name == 'UT':
		return 2.0 * (stretch - stretch ** -2.0) * dwd_i1
	if mode_name == 'PS':
		return 2.0 * (stretch - stretch ** -3.0) * dwd_i1
	if mode_name == 'ET':
		return 2.0 * (stretch - stretch ** -5.0) * dwd_i1
	raise ValueError(f'Unsupported mode: {mode_name}')

def generate_ogden_mode_data(
	mode_name: str,
	output_file: Path,
	stretch_min: float,
	stretch_max: float,
	point_count: int,
	mu_terms: Sequence[float],
	alpha_terms: Sequence[float],
	noise_std: float = 0.0,
	rng: np.random.Generator | None = None,
) -> None:
	_validate_sampling(stretch_min, stretch_max, point_count)

	stretch = np.linspace(stretch_min, stretch_max, point_count, dtype=np.float32)
	stress = ogden_nominal_stress(stretch, mode_name, mu_terms, alpha_terms).astype(np.float32)
	stress = _with_optional_noise(stress, noise_std, rng)

	output_file.parent.mkdir(parents=True, exist_ok=True)
	np.savetxt(output_file, np.column_stack([stress, stretch]), fmt='%.8f')


def generate_arruda_boyce_mode_data(
	mode_name: str,
	output_file: Path,
	stretch_min: float,
	stretch_max: float,
	point_count: int,
	mu: float,
	lambda_m: float,
	noise_std: float = 0.0,
	rng: np.random.Generator | None = None,
) -> None:
	_validate_sampling(stretch_min, stretch_max, point_count)

	stretch = np.linspace(stretch_min, stretch_max, point_count, dtype=np.float32)
	stress = arruda_boyce_nominal_stress(stretch, mode_name, mu, lambda_m).astype(np.float32)
	stress = _with_optional_noise(stress, noise_std, rng)

	output_file.parent.mkdir(parents=True, exist_ok=True)
	np.savetxt(output_file, np.column_stack([stress, stretch]), fmt='%.8f')

def generate_ogden_datasets(
	base_dir: Path,
	mode_ranges: Mapping[str, tuple[float, float]],
	point_count: int,
	mu_terms: Sequence[float],
	alpha_terms: Sequence[float],
	noise_std: float = 0.0,
	random_state: int = 42,
) -> dict[str, Path]:
	rng = np.random.default_rng(random_state)
	output_root = base_dir / 'synthetic_data' / 'ogden'
	datasets: dict[str, Path] = {}

	for mode_name, (stretch_min, stretch_max) in mode_ranges.items():
		output_file = output_root / mode_name / 'stress_stretch.txt'
		generate_ogden_mode_data(
			mode_name=mode_name,
			output_file=output_file,
			stretch_min=stretch_min,
			stretch_max=stretch_max,
			point_count=point_count,
			mu_terms=mu_terms,
			alpha_terms=alpha_terms,
			noise_std=noise_std,
			rng=rng,
		)
		datasets[mode_name] = output_file

	return datasets


def generate_arruda_boyce_datasets(
	base_dir: Path,
	mode_ranges: Mapping[str, tuple[float, float]],
	point_count: int,
	mu: float,
	lambda_m: float,
	noise_std: float = 0.0,
	random_state: int = 42,
) -> dict[str, Path]:
	rng = np.random.default_rng(random_state)
	output_root = base_dir / 'synthetic_data' / 'arruda_boyce'
	datasets: dict[str, Path] = {}

	for mode_name, (stretch_min, stretch_max) in mode_ranges.items():
		output_file = output_root / mode_name / 'stress_stretch.txt'
		generate_arruda_boyce_mode_data(
			mode_name=mode_name,
			output_file=output_file,
			stretch_min=stretch_min,
			stretch_max=stretch_max,
			point_count=point_count,
			mu=mu,
			lambda_m=lambda_m,
			noise_std=noise_std,
			rng=rng,
		)
		datasets[mode_name] = output_file

	return datasets


def generate_mooney_rivlin_datasets(
	base_dir: Path,
	mode_ranges: Mapping[str, tuple[float, float]],
	point_count: int,
	c10: float,
	c01: float,
	noise_std: float = 0.0,
	random_state: int = 42,
) -> dict[str, Path]:
	mu_terms, alpha_terms = mooney_rivlin_to_ogden_terms(c10, c01)
	rng = np.random.default_rng(random_state)
	output_root = base_dir / 'synthetic_data' / 'mooney_rivlin'
	datasets: dict[str, Path] = {}

	for mode_name, (stretch_min, stretch_max) in mode_ranges.items():
		output_file = output_root / mode_name / 'stress_stretch.txt'
		generate_ogden_mode_data(
			mode_name=mode_name,
			output_file=output_file,
			stretch_min=stretch_min,
			stretch_max=stretch_max,
			point_count=point_count,
			mu_terms=mu_terms,
			alpha_terms=alpha_terms,
			noise_std=noise_std,
			rng=rng,
		)
		datasets[mode_name] = output_file

	return datasets