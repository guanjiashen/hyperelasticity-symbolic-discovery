from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy

from .config import add_local_import_paths


_EXPRESSION_LOCALS = {
	'square': lambda value: value ** 2,
	'cube': lambda value: value ** 3,
	'sqrt': sympy.sqrt,
	'exp': sympy.exp,
	'log': sympy.log,
	'abs': sympy.Abs,
	'sin': sympy.sin,
	'cos': sympy.cos,
	'tan': sympy.tan,
}

INITIAL_STATE_TOLERANCE = 1e-6


def _load_preprocess_helpers():
	add_local_import_paths()
	from preprocess.prepare_mode_dataset_invariant import (  # pylint: disable=import-outside-toplevel
		build_invariants_invariant,
		load_mode_data,
	)

	return build_invariants_invariant, load_mode_data


def _extract_expression_text(equation_file: Path) -> str:
	for line in equation_file.read_text(encoding='utf-8').splitlines():
		if line.startswith('expression:'):
			return line.split(':', 1)[1].strip()
	raise ValueError(f'Could not find an expression line in {equation_file}.')


def _build_dataset_map(args) -> dict[str, Path]:
	dataset_candidates = (
		('UT', args.ut_dataset),
		('UC', getattr(args, 'uc_dataset', None)),
		('PS', args.ps_dataset),
		('ET', args.et_dataset),
		('SS', getattr(args, 'ss_dataset', None)),
		('SN', getattr(args, 'sn_dataset', None)),
	)
	datasets = {
		mode_name: dataset_path
		for mode_name, dataset_path in dataset_candidates
		if dataset_path is not None and str(dataset_path).lower() != 'none'
	}
	if args.enable_biaxial:
		biaxial_candidates = sorted(
			[
				path for path in args.biaxial_base_dir.glob('B*')
				if path.is_dir() and (path / 'stress_stretch.txt').is_file()
			],
			key=lambda path: int(path.name[1:]) if path.name[1:].isdigit() else float('inf'),
		)
		for folder in biaxial_candidates:
			index_token = folder.name[1:]
			if index_token.isdigit():
				datasets[f'{args.biaxial_prefix}{int(index_token)}'] = folder / 'stress_stretch.txt'
	return datasets


def _compile_invariant_derivatives(expression_text: str):
	i1_symbol, i2_symbol = sympy.symbols('I1 I2')
	expression = sympy.sympify(expression_text, locals=_EXPRESSION_LOCALS)
	dwd_i1 = sympy.diff(expression, i1_symbol)
	dwd_i2 = sympy.diff(expression, i2_symbol)
	return (
		sympy.lambdify((i1_symbol, i2_symbol), dwd_i1, modules='numpy'),
		sympy.lambdify((i1_symbol, i2_symbol), dwd_i2, modules='numpy'),
	)


def _as_vector(values, reference: np.ndarray) -> np.ndarray:
	array = np.asarray(values, dtype=np.float64)
	if array.ndim == 0:
		return np.full_like(reference, float(array), dtype=np.float64)
	return array.reshape(-1)


def _get_biaxial_ratios():
	add_local_import_paths()
	from preprocess.prepare_mode_dataset_invariant import BIAXIAL_STRETCH_RATIOS  # pylint: disable=import-outside-toplevel

	return BIAXIAL_STRETCH_RATIOS


def _get_mode_stretches(stretch: np.ndarray, mode_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	lambda1 = np.asarray(stretch, dtype=np.float64)
	if mode_name in ('UT', 'UC'):
		lambda2 = lambda1 ** -0.5
		lambda3 = lambda1 ** -0.5
	elif mode_name == 'PS':
		lambda2 = np.ones_like(lambda1)
		lambda3 = lambda1 ** -1.0
	elif mode_name == 'ET':
		lambda2 = lambda1.copy()
		lambda3 = lambda1 ** -2.0
	elif mode_name in ('SS', 'SN'):
		gamma = lambda1
		root = np.sqrt(4.0 + gamma ** 2)
		lambda1 = np.sqrt(1.0 + 0.5 * gamma ** 2 + 0.5 * gamma * root)
		lambda2 = np.sqrt(1.0 + 0.5 * gamma ** 2 - 0.5 * gamma * root)
		lambda3 = np.ones_like(gamma)
	elif mode_name.startswith('B'):
		lambda2 = np.full_like(lambda1, _get_biaxial_ratios()[mode_name])
		lambda3 = 1.0 / (lambda1 * lambda2)
	else:
		raise ValueError(f'Unsupported mode: {mode_name}')
	return lambda1, lambda2, lambda3


def evaluate_initial_state_energy(expression_text: str, feature_space: str) -> dict[str, float | bool]:
	if feature_space == 'invariants':
		variables = sympy.symbols('I1 I2')
		values = (3.0, 3.0)
	else:
		variables = sympy.symbols('lambda1 lambda2 lambda3')
		values = (1.0, 1.0, 1.0)

	expression = sympy.sympify(expression_text, locals=_EXPRESSION_LOCALS)
	energy_fn = sympy.lambdify(variables, expression, modules='numpy')
	initial_energy = float(np.asarray(energy_fn(*values), dtype=np.float64))
	return {
		'initial_energy': initial_energy,
		'abs_initial_energy': abs(initial_energy),
		'is_initial_energy_zero': abs(initial_energy) <= INITIAL_STATE_TOLERANCE,
	}


def _compute_invariant_stress_terms(
	stretch: np.ndarray,
	mode_name: str,
	build_invariants_invariant,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	invariants = build_invariants_invariant(stretch.astype(np.float32), mode_name).astype(np.float64)
	stretch64 = stretch.astype(np.float64)

	if mode_name in ('UT', 'UC'):
		stress_factor = 2.0 * (stretch64 - stretch64 ** -2.0)
		dwd_i2_factor = 1.0 / stretch64
	elif mode_name in ('SS', 'SN'):
		stress_factor = 2.0 * stretch64
		dwd_i2_factor = np.ones_like(stretch64)
	elif mode_name == 'PS':
		stress_factor = 2.0 * (stretch64 - stretch64 ** -3.0)
		dwd_i2_factor = np.ones_like(stretch64)
	elif mode_name == 'ET':
		stress_factor = 2.0 * (stretch64 - stretch64 ** -5.0)
		dwd_i2_factor = stretch64 ** 2.0
	elif mode_name.startswith('B'):
		biaxial_stretch_ratios = _get_biaxial_ratios()
		lambda_2 = np.full_like(stretch64, biaxial_stretch_ratios[mode_name], dtype=np.float64)
		lambda_3 = 1.0 / (stretch64 * lambda_2)
		stress_factor = 2.0 * (stretch64 - lambda_3 ** 2.0 / stretch64)
		dwd_i2_factor = lambda_2 ** 2.0
	else:
		raise ValueError(f'Unsupported mode: {mode_name}')

	return invariants, stress_factor, dwd_i2_factor


def evaluate_expression_on_features(
	expression_text: str,
	feature_space: str,
	features: np.ndarray,
) -> np.ndarray:
	if feature_space == 'invariants':
		variables = sympy.symbols('I1 I2')
	else:
		variables = sympy.symbols('lambda1 lambda2 lambda3')

	expression = sympy.sympify(expression_text, locals=_EXPRESSION_LOCALS)
	energy_fn = sympy.lambdify(variables, expression, modules='numpy')
	values = energy_fn(*[features[:, index] for index in range(features.shape[1])])
	return _as_vector(values, features[:, 0])


def normalize_expression_to_zero_initial_state(expression_text: str, feature_space: str) -> dict[str, str | float | bool]:
	raw_check = evaluate_initial_state_energy(expression_text, feature_space)
	expression = sympy.sympify(expression_text, locals=_EXPRESSION_LOCALS)
	normalized_expression = sympy.simplify(expression - raw_check['initial_energy'])
	normalized_expression_text = str(normalized_expression)
	normalized_check = evaluate_initial_state_energy(normalized_expression_text, feature_space)
	return {
		'raw_expression': expression_text,
		'raw_initial_energy': raw_check['initial_energy'],
		'normalized_expression': normalized_expression_text,
		'normalized_initial_energy': normalized_check['initial_energy'],
		'abs_normalized_initial_energy': normalized_check['abs_initial_energy'],
		'is_normalized_initial_energy_zero': normalized_check['is_initial_energy_zero'],
	}


def _predict_stress_from_invariants(
	stretch: np.ndarray,
	mode_name: str,
	dwd_i1_fn,
	dwd_i2_fn,
	build_invariants_invariant,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	invariants, stress_factor, dwd_i2_factor = _compute_invariant_stress_terms(
		stretch,
		mode_name,
		build_invariants_invariant,
	)
	i1 = invariants[:, 0]
	i2 = invariants[:, 1]
	dwd_i1 = _as_vector(dwd_i1_fn(i1, i2), i1)
	dwd_i2 = _as_vector(dwd_i2_fn(i1, i2), i1)
	prediction = stress_factor * (dwd_i1 + dwd_i2_factor * dwd_i2)
	return invariants, dwd_i1, dwd_i2, prediction


def _predict_stress_from_mode(
	stretch: np.ndarray,
	mode_name: str,
	dwd_i1_fn,
	dwd_i2_fn,
	build_invariants_invariant,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	return _predict_stress_from_invariants(stretch, mode_name, dwd_i1_fn, dwd_i2_fn, build_invariants_invariant)


def _compile_stretch_derivatives(expression_text: str):
	lambda1, lambda2, lambda3 = sympy.symbols('lambda1 lambda2 lambda3')
	expression = sympy.sympify(expression_text, locals=_EXPRESSION_LOCALS)
	return tuple(
		sympy.lambdify((lambda1, lambda2, lambda3), sympy.diff(expression, variable), modules='numpy')
		for variable in (lambda1, lambda2, lambda3)
	)


def _predict_stress_from_stretches(stretch: np.ndarray, mode_name: str, derivative_functions):
	lambda1, lambda2, lambda3 = _get_mode_stretches(stretch, mode_name)
	dwd_lambda1 = _as_vector(derivative_functions[0](lambda1, lambda2, lambda3), lambda1)
	dwd_lambda2 = _as_vector(derivative_functions[1](lambda1, lambda2, lambda3), lambda1)
	dwd_lambda3 = _as_vector(derivative_functions[2](lambda1, lambda2, lambda3), lambda1)
	if mode_name == 'UT':
		prediction = dwd_lambda1 - 0.5 * stretch ** -1.5 * (dwd_lambda2 + dwd_lambda3)
	elif mode_name == 'PS':
		prediction = dwd_lambda1 - stretch ** -2.0 * dwd_lambda3
	elif mode_name == 'ET':
		prediction = 0.5 * (dwd_lambda1 + dwd_lambda2) - stretch ** -3.0 * dwd_lambda3
	elif mode_name.startswith('B'):
		prediction = dwd_lambda1 - (lambda3 / lambda1) * dwd_lambda3
	else:
		raise ValueError(f'Unsupported mode: {mode_name}')
	return lambda1, lambda2, lambda3, dwd_lambda1, dwd_lambda2, dwd_lambda3, prediction


def build_experiment_stress_dataset(args) -> dict[str, np.ndarray]:
	build_invariants_invariant, load_mode_data = _load_preprocess_helpers()
	datasets = _build_dataset_map(args)
	all_invariants: list[np.ndarray] = []
	all_stress_factor: list[np.ndarray] = []
	all_dwd_i2_factor: list[np.ndarray] = []
	all_stress_target: list[np.ndarray] = []

	for mode_name, file_path in datasets.items():
		if not file_path.is_file():
			raise FileNotFoundError(f'Dataset for {mode_name} does not exist: {file_path}')
		mode_data = load_mode_data(mode_name, file_path)
		stretch = mode_data['stretch'].astype(np.float64)
		invariants, stress_factor, dwd_i2_factor = _compute_invariant_stress_terms(
			stretch,
			mode_name,
			build_invariants_invariant,
		)
		all_invariants.append(invariants)
		all_stress_factor.append(stress_factor)
		all_dwd_i2_factor.append(dwd_i2_factor)
		all_stress_target.append(mode_data['stress'].astype(np.float64))

	return {
		'invariants': np.concatenate(all_invariants, axis=0),
		'stress_factor': np.concatenate(all_stress_factor, axis=0),
		'dwd_i2_factor': np.concatenate(all_dwd_i2_factor, axis=0),
		'stress_target': np.concatenate(all_stress_target, axis=0),
	}


def _write_predictions_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
	with output_path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.writer(handle)
		writer.writerow(['mode', 'stretch', 'I1', 'I2', 'dW_dI1', 'dW_dI2', 'stress_target', 'stress_prediction'])
		for row in rows:
			writer.writerow([
				row['mode'],
				f'{float(row["stretch"]):.17g}',
				f'{float(row["I1"]):.17g}',
				f'{float(row["I2"]):.17g}',
				f'{float(row["dW_dI1"]):.17g}',
				f'{float(row["dW_dI2"]):.17g}',
				f'{float(row["stress_target"]):.17g}',
				f'{float(row["stress_prediction"]):.17g}',
			])


def _save_prediction_plot(per_mode: Mapping[str, dict[str, np.ndarray]], output_path: Path) -> None:
	plt.figure(figsize=(8, 6))
	color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
	colors = {
		mode_name: color_cycle[index % len(color_cycle)]
		for index, mode_name in enumerate(per_mode.keys())
	}
	for mode_name, mode_values in per_mode.items():
		stretch = mode_values['stretch']
		target = mode_values['stress_target']
		prediction = mode_values['stress_prediction']
		reference_mask = np.isclose(stretch, 1.0) & np.isclose(target, 0.0)
		plot_mask = ~reference_mask if mode_name.startswith('B') else np.ones_like(reference_mask, dtype=bool)
		stretch = stretch[plot_mask]
		target = target[plot_mask]
		prediction = prediction[plot_mask]
		sort_idx = np.argsort(stretch)
		plt.scatter(stretch, target, label=f'{mode_name} data', s=18, color=colors[mode_name], alpha=0.75)
		plt.plot(stretch[sort_idx], prediction[sort_idx], label=f'{mode_name} symbolic', linewidth=2.0, color=colors[mode_name])

	plt.xlabel(r'$\lambda_1$', fontsize=25)
	plt.ylabel(r'$P_{11}\, (\mathrm{MPa})$', fontsize=25)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def generate_experiment_predictions(args, expression_text: str | None = None) -> dict[str, Path | dict[str, dict[str, float]]]:
	build_invariants_invariant, load_mode_data = _load_preprocess_helpers()
	resolved_expression = expression_text or _extract_expression_text(args.equation_file)
	if args.feature_space == 'invariants':
		dwd_i1_fn, dwd_i2_fn = _compile_invariant_derivatives(resolved_expression)
	else:
		stretch_derivative_functions = _compile_stretch_derivatives(resolved_expression)
	datasets = _build_dataset_map(args)
	for mode_name, file_path in datasets.items():
		if not file_path.is_file():
			raise FileNotFoundError(f'Dataset for {mode_name} does not exist: {file_path}')

	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	csv_path = output_dir / 'experiment_predictions.csv'
	plot_path = output_dir / 'experiment_predictions.png'
	summary_path = output_dir / 'experiment_prediction_summary.json'

	rows: list[dict[str, float | str]] = []
	per_mode: dict[str, dict[str, np.ndarray]] = {}
	summary: dict[str, dict[str, float]] = {}
	total_squared_error = 0.0
	total_sample_count = 0

	for mode_name, file_path in datasets.items():
		mode_data = load_mode_data(mode_name, file_path)
		stretch = mode_data['stretch'].astype(np.float64)
		target = mode_data['stress'].astype(np.float64)
		if args.feature_space == 'invariants':
			invariants, dwd_i1, dwd_i2, prediction = _predict_stress_from_mode(
				stretch, mode_name, dwd_i1_fn, dwd_i2_fn, build_invariants_invariant,
			)
		else:
			lambda1, lambda2, lambda3, dwd_i1, dwd_i2, dwd_i3, prediction = _predict_stress_from_stretches(
				stretch, mode_name, stretch_derivative_functions,
			)
			invariants = np.column_stack([
				lambda1 ** 2 + lambda2 ** 2 + lambda3 ** 2,
				lambda1 ** 2 * lambda2 ** 2 + lambda2 ** 2 * lambda3 ** 2 + lambda3 ** 2 * lambda1 ** 2,
			])
		squared_error = (prediction - target) ** 2
		stress_loss = float(np.mean(squared_error))
		rmse = float(np.sqrt(stress_loss))
		denominator = float(np.sum((target - target.mean()) ** 2))
		r2 = float(1.0 - np.sum((prediction - target) ** 2) / denominator) if denominator > 0.0 else float('nan')
		total_squared_error += float(np.sum(squared_error))
		total_sample_count += len(stretch)
		summary[mode_name] = {
			'stress_loss': stress_loss,
			'rmse': rmse,
			'r2': r2,
			'sample_count': float(len(stretch)),
		}
		per_mode[mode_name] = {
			'stretch': stretch,
			'stress_target': target,
			'stress_prediction': prediction,
		}
		for index in range(len(stretch)):
			rows.append(
				{
					'mode': mode_name,
					'stretch': stretch[index],
					'I1': invariants[index, 0],
					'I2': invariants[index, 1],
					'dW_dI1': dwd_i1[index],
					'dW_dI2': dwd_i2[index],
					'stress_target': target[index],
					'stress_prediction': prediction[index],
				}
			)

	_write_predictions_csv(rows, csv_path)
	_save_prediction_plot(per_mode, plot_path)
	summary['overall'] = {
		'stress_loss': total_squared_error / max(total_sample_count, 1),
		'rmse': float(np.sqrt(total_squared_error / max(total_sample_count, 1))),
		'sample_count': float(total_sample_count),
	}
	summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
	print(f'Symbolic stress loss (overall MSE): {summary["overall"]["stress_loss"]:.10g}')
	return {
		'csv_path': csv_path,
		'plot_path': plot_path,
		'summary_path': summary_path,
		'metrics': summary,
	}


def run_predict_experiment(args) -> None:
	outputs = generate_experiment_predictions(args)
	print(f'Reference-data predictions written to {outputs["csv_path"]}.')
	print(f'Reference-data prediction plot written to {outputs["plot_path"]}.')
