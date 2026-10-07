from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy

from .experimental_prediction import build_experiment_stress_dataset, evaluate_expression_on_features


@dataclass
class _LinearTerm:
	coefficient: float
	basis: sympy.Expr


def _split_linear_additive_terms(expression_text: str) -> list[_LinearTerm]:
	expression = sympy.expand(sympy.sympify(expression_text))
	terms: list[_LinearTerm] = []
	for term in sympy.Add.make_args(expression):
		coefficient, basis = term.as_coeff_Mul()
		if basis == 1:
			continue
		if not coefficient.is_number or not basis.free_symbols:
			raise ValueError(f'Cannot isolate a numeric coefficient from term: {term}')
		terms.append(_LinearTerm(float(coefficient), basis))
	if len(terms) < 2:
		raise ValueError('At least two non-constant additive terms are required for pruning.')
	return terms


def _as_column(value, size: int) -> np.ndarray:
	array = np.asarray(value, dtype=np.float64)
	if array.ndim == 0:
		return np.full(size, float(array), dtype=np.float64)
	return array.reshape(-1)


def _build_energy_matrix(terms: list[_LinearTerm], features: np.ndarray) -> np.ndarray:
	i1, i2 = sympy.symbols('I1 I2')
	columns = []
	for term in terms:
		function = sympy.lambdify((i1, i2), term.basis, modules='numpy')
		values = _as_column(function(features[:, 0], features[:, 1]), len(features))
		reference_value = float(function(3.0, 3.0))
		columns.append(values - reference_value)
	return np.column_stack(columns)


def _build_stress_matrix(terms: list[_LinearTerm], dataset: dict[str, np.ndarray]) -> np.ndarray:
	i1, i2 = sympy.symbols('I1 I2')
	features = dataset['invariants']
	columns = []
	for term in terms:
		d1 = sympy.lambdify((i1, i2), sympy.diff(term.basis, i1), modules='numpy')
		d2 = sympy.lambdify((i1, i2), sympy.diff(term.basis, i2), modules='numpy')
		d1_values = _as_column(d1(features[:, 0], features[:, 1]), len(features))
		d2_values = _as_column(d2(features[:, 0], features[:, 1]), len(features))
		columns.append(
			dataset['stress_factor']
			* (d1_values + dataset['dwd_i2_factor'] * d2_values)
		)
	return np.column_stack(columns)


def _normalized_block(matrix: np.ndarray, target: np.ndarray, weight: float) -> tuple[np.ndarray, np.ndarray]:
	scale = max(float(np.sqrt(np.mean(target ** 2))), np.finfo(np.float64).eps)
	factor = np.sqrt(weight) / scale
	return matrix * factor, target * factor


def _fit_coefficients(
	active: list[int],
	energy_matrix: np.ndarray,
	energy_target: np.ndarray,
	stress_matrix: np.ndarray,
	stress_target: np.ndarray,
	energy_weight: float,
	stress_weight: float,
) -> np.ndarray:
	blocks, targets = [], []
	if energy_weight > 0.0:
		block, target = _normalized_block(energy_matrix[:, active], energy_target, energy_weight)
		blocks.append(block)
		targets.append(target)
	if stress_weight > 0.0:
		block, target = _normalized_block(stress_matrix[:, active], stress_target, stress_weight)
		blocks.append(block)
		targets.append(target)
	return np.linalg.lstsq(np.vstack(blocks), np.concatenate(targets), rcond=None)[0]


def _metrics(
	active: list[int],
	coefficients: np.ndarray,
	energy_matrix: np.ndarray,
	energy_target: np.ndarray,
	stress_matrix: np.ndarray,
	stress_target: np.ndarray,
	energy_weight: float,
	stress_weight: float,
) -> dict[str, float]:
	energy_error = energy_matrix[:, active] @ coefficients - energy_target
	stress_error = stress_matrix[:, active] @ coefficients - stress_target
	energy_scale = max(float(np.sqrt(np.mean(energy_target ** 2))), np.finfo(np.float64).eps)
	stress_scale = max(float(np.sqrt(np.mean(stress_target ** 2))), np.finfo(np.float64).eps)
	weighted_error = (
		energy_weight * float(np.mean(energy_error ** 2)) / energy_scale ** 2
		+ stress_weight * float(np.mean(stress_error ** 2)) / stress_scale ** 2
	)
	total_weight = energy_weight + stress_weight
	return {
		'energy_rmse': float(np.sqrt(np.mean(energy_error ** 2))),
		'stress_rmse': float(np.sqrt(np.mean(stress_error ** 2))),
		'selection_score': float(np.sqrt(weighted_error / total_weight)),
	}


def _expression(terms: list[_LinearTerm], active: list[int], coefficients: np.ndarray) -> str:
	i1, i2 = sympy.symbols('I1 I2')
	reference = {i1: 3.0, i2: 3.0}
	expression = sympy.Integer(0)
	for index, coefficient in zip(active, coefficients):
		basis = terms[index].basis
		expression += sympy.Float(float(coefficient), 12) * (basis - basis.subs(reference))
	return str(sympy.expand(expression))


def simplify_and_refit_expression(
	expression_text: str,
	data: np.ndarray,
	features: np.ndarray,
	args,
) -> dict[str, object]:
	"""Greedily prune additive invariant terms and refit their linear coefficients."""
	if args.feature_space != 'invariants':
		raise ValueError('Automatic pruning currently supports only invariant expressions.')
	if args.simplify_energy_weight < 0.0 or args.simplify_stress_weight < 0.0:
		raise ValueError('Simplification loss weights must be non-negative.')
	if args.simplify_energy_weight == 0.0 and args.simplify_stress_weight == 0.0:
		raise ValueError('At least one simplification loss weight must be positive.')
	if args.simplify_rmse_tolerance < 0.0:
		raise ValueError('--simplify-rmse-tolerance must be non-negative.')

	terms = _split_linear_additive_terms(expression_text)
	energy_target = np.asarray(data['energy'], dtype=np.float64)
	energy_matrix = _build_energy_matrix(terms, features)
	stress_dataset = build_experiment_stress_dataset(args)
	stress_target = stress_dataset['stress_target']
	stress_matrix = _build_stress_matrix(terms, stress_dataset)
	original_coefficients = np.asarray([term.coefficient for term in terms])
	all_indices = list(range(len(terms)))
	original_metrics = _metrics(
		all_indices, original_coefficients, energy_matrix, energy_target,
		stress_matrix, stress_target, args.simplify_energy_weight, args.simplify_stress_weight,
	)
	criterion = 'selection_score'
	limit = original_metrics[criterion] * (1.0 + args.simplify_rmse_tolerance)

	contributions = []
	energy_scale = max(float(np.sqrt(np.mean(energy_target ** 2))), np.finfo(np.float64).eps)
	stress_scale = max(float(np.sqrt(np.mean(stress_target ** 2))), np.finfo(np.float64).eps)
	for index, term in enumerate(terms):
		energy_rms = float(np.sqrt(np.mean((term.coefficient * energy_matrix[:, index]) ** 2)))
		stress_rms = float(np.sqrt(np.mean((term.coefficient * stress_matrix[:, index]) ** 2)))
		contributions.append({
			'term': str(term.coefficient * term.basis),
			'coefficient': term.coefficient,
			'energy_rms': energy_rms,
			'stress_rms': stress_rms,
			'normalized_energy_contribution': energy_rms / energy_scale,
			'normalized_stress_contribution': stress_rms / stress_scale,
		})

	active = all_indices
	pareto = [{
		'term_count': len(active),
		'expression': expression_text,
		**original_metrics,
		'accepted': True,
	}]
	removed_terms: list[str] = []
	final_coefficients = original_coefficients
	while len(active) > 1:
		candidates = []
		for removed_index in active:
			candidate_active = [index for index in active if index != removed_index]
			coefficients = _fit_coefficients(
				candidate_active, energy_matrix, energy_target, stress_matrix, stress_target,
				args.simplify_energy_weight, args.simplify_stress_weight,
			)
			candidate_metrics = _metrics(
				candidate_active, coefficients, energy_matrix, energy_target,
				stress_matrix, stress_target, args.simplify_energy_weight, args.simplify_stress_weight,
			)
			candidates.append((candidate_metrics[criterion], removed_index, candidate_active, coefficients, candidate_metrics))
		candidates.sort(key=lambda candidate: candidate[0])
		_, removed_index, candidate_active, coefficients, candidate_metrics = candidates[0]
		accepted = candidate_metrics[criterion] <= limit
		candidate_expression = _expression(terms, candidate_active, coefficients)
		pareto.append({
			'term_count': len(candidate_active),
			'expression': candidate_expression,
			'removed_term': str(terms[removed_index].coefficient * terms[removed_index].basis),
			**candidate_metrics,
			'accepted': accepted,
		})
		if not accepted:
			break
		removed_terms.append(str(terms[removed_index].coefficient * terms[removed_index].basis))
		active = candidate_active
		final_coefficients = coefficients

	final_expression = expression_text if not removed_terms else _expression(terms, active, final_coefficients)
	final_metrics = original_metrics if not removed_terms else _metrics(
		active, final_coefficients, energy_matrix, energy_target, stress_matrix, stress_target,
		args.simplify_energy_weight, args.simplify_stress_weight,
	)
	return {
		'original_expression': expression_text,
		'simplified_expression': final_expression,
		'changed': bool(removed_terms),
		'criterion': criterion,
		'rmse_tolerance': args.simplify_rmse_tolerance,
		'criterion_limit': limit,
		'energy_weight': args.simplify_energy_weight,
		'stress_weight': args.simplify_stress_weight,
		'original_metrics': original_metrics,
		'final_metrics': final_metrics,
		'removed_terms': removed_terms,
		'term_contributions': contributions,
		'pareto': pareto,
	}
