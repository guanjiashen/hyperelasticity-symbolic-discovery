from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import sympy

from .experimental_prediction import _EXPRESSION_LOCALS, evaluate_initial_state_energy


def _range(values) -> list[float]:
	array = np.asarray(values, dtype=np.float64).reshape(-1)
	return [float(np.min(array)), float(np.max(array))]


def _sampling_mode(data: np.ndarray) -> str:
	column_names = set(data.dtype.names or ())
	if 'stretch' in column_names:
		return 'loadcases'
	return 'grid'


def build_material_package(
	*,
	expression_text: str,
	feature_space: str,
	data: np.ndarray,
	args,
	rmse: float,
	r2: float,
	initial_state_check: dict[str, float | bool],
) -> dict[str, object]:
	if feature_space != 'invariants':
		raise NotImplementedError('OpenRadioss material-package export currently supports only feature_space=invariants.')

	i1_symbol, i2_symbol = sympy.symbols('I1 I2')
	expression = sympy.sympify(expression_text, locals=_EXPRESSION_LOCALS)
	dwd_i1 = sympy.simplify(sympy.diff(expression, i1_symbol))
	dwd_i2 = sympy.simplify(sympy.diff(expression, i2_symbol))
	_, denominator = sympy.fraction(sympy.together(expression))
	denominator_text = str(sympy.simplify(denominator))
	denominator_fn = sympy.lambdify((i1_symbol, i2_symbol), denominator, modules='numpy')

	i1_values = np.asarray(data['I1'], dtype=np.float64)
	i2_values = np.asarray(data['I2'], dtype=np.float64)
	denominator_values = np.asarray(denominator_fn(i1_values, i2_values), dtype=np.float64)
	finite_mask = np.isfinite(denominator_values)
	finite_denominator_values = denominator_values[finite_mask]
	if finite_denominator_values.size == 0:
		singularity_margin = float('nan')
		denominator_range = [float('nan'), float('nan')]
	else:
		singularity_margin = float(np.min(np.abs(finite_denominator_values)))
		denominator_range = _range(finite_denominator_values)

	warnings: list[str] = []
	if not finite_mask.all():
		warnings.append('Expression denominator is non-finite on some exported samples.')
	if np.isfinite(singularity_margin) and singularity_margin < float(args.singularity_threshold):
		warnings.append(
			'Expression denominator approaches zero inside the exported domain; '
			'protect the runtime evaluator or refit the symbolic expression.'
		)
	if float(args.bulk_kappa) <= 0.0:
		warnings.append('bulk_kappa is non-positive; volumetric response is not physically configured yet.')

	package = {
		'format_version': 1,
		'material_model': 'openradioss_invariant_hyperelastic',
		'feature_space': feature_space,
		'sampling_mode': _sampling_mode(data),
		'source': {
			'input_csv': str(args.input_csv.resolve()),
			'output_dir': str(args.output_dir.resolve()),
			'seed': int(args.seed),
			'energy_loss_weight': float(args.energy_loss_weight),
			'stress_loss_weight': float(args.stress_loss_weight),
			'niterations': int(args.niterations),
			'population_size': int(args.population_size),
			'populations': int(args.populations),
			'maxsize': int(args.maxsize),
		},
		'reference_state': {
			'I1': 3.0,
			'I2': 3.0,
			'J': 1.0,
			'energy': float(initial_state_check['initial_energy']),
			'is_zero': bool(initial_state_check['is_initial_energy_zero']),
		},
		'constitutive': {
			'isochoric_energy': expression_text,
			'dW_dI1': str(dwd_i1),
			'dW_dI2': str(dwd_i2),
			'bulk_model': str(args.bulk_model),
			'bulk_kappa': float(args.bulk_kappa),
		},
		'valid_domain': {
			'lambda1_range': _range(data['lambda1']),
			'lambda2_range': _range(data['lambda2']),
			'lambda3_range': _range(data['lambda3']),
			'I1_range': _range(i1_values),
			'I2_range': _range(i2_values),
		},
		'diagnostics': {
			'rmse': float(rmse),
			'r2': float(r2),
			'expression_denominator': denominator_text,
			'denominator_range': denominator_range,
			'singularity_margin': singularity_margin,
			'nonfinite_denominator_count': int(np.size(denominator_values) - np.count_nonzero(finite_mask)),
		},
		'warnings': warnings,
	}
	package_payload = json.dumps(package, sort_keys=True, ensure_ascii=True)
	package['model_hash'] = hashlib.sha256(package_payload.encode('utf-8')).hexdigest()
	return package


def write_material_package(
	*,
	expression_text: str,
	feature_space: str,
	data: np.ndarray,
	args,
	rmse: float,
	r2: float,
	initial_state_check: dict[str, float | bool],
	output_path: Path,
) -> Path:
	package = build_material_package(
		expression_text=expression_text,
		feature_space=feature_space,
		data=data,
		args=args,
		rmse=rmse,
		r2=r2,
		initial_state_check=initial_state_check,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(package, indent=2) + '\n', encoding='utf-8')
	return output_path
