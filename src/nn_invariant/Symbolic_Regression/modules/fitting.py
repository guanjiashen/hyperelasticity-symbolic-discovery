from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import sympy

from .config import add_local_import_paths
from .experimental_prediction import (
	evaluate_expression_on_features,
	evaluate_initial_state_energy,
	generate_experiment_predictions,
	normalize_expression_to_zero_initial_state,
)
from .material_package import write_material_package
from .plotting import save_energy_comparison_plot, save_symbolic_energy_contour_plot
from .simplification import simplify_and_refit_expression


def load_dataset(input_csv) -> np.ndarray:
	return np.genfromtxt(input_csv, delimiter=',', names=True, dtype=np.float64)


def _format_julia_vector(values: np.ndarray) -> str:
	flat_values = np.asarray(values, dtype=np.float64).reshape(-1)
	return ', '.join(f'{float(value):.17g}' for value in flat_values)


def _format_julia_matrix(values: np.ndarray) -> tuple[str, int, int]:
	matrix = np.asarray(values, dtype=np.float64)
	rows, cols = matrix.shape
	flat_values = matrix.flatten(order='F')
	return ', '.join(f'{float(value):.17g}' for value in flat_values), rows, cols


def build_grid_stress_dataset(data: np.ndarray, feature_space: str) -> dict[str, np.ndarray]:
	required_columns = {'lambda1', 'lambda2', 'lambda3', 'stress_lambda1', 'stress_lambda2'}
	missing_columns = sorted(required_columns.difference(data.dtype.names or ()))
	if missing_columns:
		raise ValueError(
			'Stress-aware fitting now expects NN-derived stress targets in the exported CSV. '
			f'Missing columns: {", ".join(missing_columns)}. Re-run the export step to regenerate the dataset.'
		)

	lambda1 = np.asarray(data['lambda1'], dtype=np.float64)
	lambda2 = np.asarray(data['lambda2'], dtype=np.float64)
	lambda3 = 1.0 / (lambda1 * lambda2)

	if feature_space == 'invariants':
		return {
			'features': np.column_stack([data['I1'], data['I2']]).astype(np.float64),
			'stress_factor_lambda1': 2.0 * (lambda1 - lambda3 ** 2.0 / lambda1),
			'stress_factor_lambda2': 2.0 * (lambda2 - lambda3 ** 2.0 / lambda2),
			'dwd_i2_factor_lambda1': lambda2 ** 2.0,
			'dwd_i2_factor_lambda2': lambda1 ** 2.0,
			'stress_target_lambda1': np.asarray(data['stress_lambda1'], dtype=np.float64),
			'stress_target_lambda2': np.asarray(data['stress_lambda2'], dtype=np.float64),
		}

	return {
		# Each grid point contributes two constrained principal-stretch derivatives:
		# dW/dlambda1 - (lambda3/lambda1)dW/dlambda3 and the analogous lambda2 term.
		'kind': np.asarray(['stretches']),
		'features': np.vstack([
			np.column_stack([lambda1, lambda2, lambda3]),
			np.column_stack([lambda1, lambda2, lambda3]),
		]).astype(np.float64),
		'stress_coefficients': np.vstack([
			np.column_stack([np.ones_like(lambda1), np.zeros_like(lambda1), -lambda3 / lambda1]),
			np.column_stack([np.zeros_like(lambda2), np.ones_like(lambda2), -lambda3 / lambda2]),
		]).astype(np.float64),
		'stress_target': np.concatenate([
			np.asarray(data['stress_lambda1'], dtype=np.float64),
			np.asarray(data['stress_lambda2'], dtype=np.float64),
		]),
	}


def build_loadcase_stress_dataset(data: np.ndarray, feature_space: str) -> dict[str, np.ndarray]:
	required_columns = {
		'stretch', 'lambda1', 'lambda2', 'lambda3', 'mode_id',
		'nominal_stress', 'stress_factor', 'dwd_i2_factor', 'I1', 'I2',
	}
	missing_columns = sorted(required_columns.difference(data.dtype.names or ()))
	if missing_columns:
		raise ValueError(
			'Loadcase stress-aware fitting expects NN-derived nominal-stress columns in the exported CSV. '
			f'Missing columns: {", ".join(missing_columns)}. Re-run the export step to regenerate the dataset.'
		)
	if feature_space == 'stretches':
		lambda1 = np.asarray(data['lambda1'], dtype=np.float64)
		lambda2 = np.asarray(data['lambda2'], dtype=np.float64)
		lambda3 = np.asarray(data['lambda3'], dtype=np.float64)
		mode_id = np.asarray(data['mode_id'], dtype=np.int64)
		coefficients = np.zeros((len(lambda1), 3), dtype=np.float64)
		ut_mask, ps_mask, et_mask = mode_id == 1, mode_id == 2, mode_id == 3
		if not np.all(ut_mask | ps_mask | et_mask):
			raise ValueError('Unsupported loadcase mode_id for principal-stretch stress fitting.')
		coefficients[ut_mask] = np.column_stack([
			np.ones(np.count_nonzero(ut_mask)),
			-0.5 * lambda1[ut_mask] ** -1.5,
			-0.5 * lambda1[ut_mask] ** -1.5,
		])
		coefficients[ps_mask] = np.column_stack([
			np.ones(np.count_nonzero(ps_mask)),
			np.zeros(np.count_nonzero(ps_mask)),
			-lambda1[ps_mask] ** -2.0,
		])
		coefficients[et_mask] = np.column_stack([
			0.5 * np.ones(np.count_nonzero(et_mask)),
			0.5 * np.ones(np.count_nonzero(et_mask)),
			-lambda1[et_mask] ** -3.0,
		])
		return {
			'kind': np.asarray(['stretches']),
			'features': np.column_stack([lambda1, lambda2, lambda3]).astype(np.float64),
			'stress_coefficients': coefficients,
			'stress_target': np.asarray(data['nominal_stress'], dtype=np.float64),
		}

	return {
		'kind': np.asarray(['loadcases']),
		'features': np.column_stack([data['I1'], data['I2']]).astype(np.float64),
		'stress_factor': np.asarray(data['stress_factor'], dtype=np.float64),
		'dwd_i2_factor': np.asarray(data['dwd_i2_factor'], dtype=np.float64),
		'stress_target': np.asarray(data['nominal_stress'], dtype=np.float64),
	}


def build_stress_dataset(data: np.ndarray, feature_space: str) -> dict[str, np.ndarray]:
	column_names = set(data.dtype.names or ())
	if {'stress_lambda1', 'stress_lambda2'}.issubset(column_names):
		return build_grid_stress_dataset(data, feature_space)
	if {'nominal_stress', 'stress_factor', 'dwd_i2_factor'}.issubset(column_names):
		return build_loadcase_stress_dataset(data, feature_space)
	raise ValueError(
		'Stress-aware fitting requires either grid stress columns '
		'("stress_lambda1", "stress_lambda2") or loadcase stress columns '
		'("nominal_stress", "stress_factor", "dwd_i2_factor").'
	)


def _build_combined_loss_function(args, stress_dataset: dict[str, np.ndarray]) -> str:
	stress_dataset_kind = str(np.asarray(stress_dataset['kind']).reshape(-1)[0])
	stress_x_literal, stress_rows, stress_cols = _format_julia_matrix(stress_dataset['features'].T)
	if stress_dataset_kind == 'stretches':
		stress_target_literal = _format_julia_vector(stress_dataset['stress_target'])
		coefficients_literal, coefficient_rows, coefficient_cols = _format_julia_matrix(stress_dataset['stress_coefficients'].T)
		return f"""
using Zygote

const SYMBOLIC_STRESS_X = reshape(Float64[{stress_x_literal}], {stress_rows}, {stress_cols})
const SYMBOLIC_STRESS_COEFFICIENTS = reshape(Float64[{coefficients_literal}], {coefficient_rows}, {coefficient_cols})
const SYMBOLIC_STRESS_TARGET = Float64[{stress_target_literal}]
const SYMBOLIC_REFERENCE_POINT = reshape(Float64[1.0, 1.0, 1.0], {stress_rows}, 1)
const SYMBOLIC_ENERGY_LOSS_WEIGHT = {float(args.energy_loss_weight):.17g}
const SYMBOLIC_STRESS_LOSS_WEIGHT = {float(args.stress_loss_weight):.17g}

function combined_energy_stress_loss(tree, dataset, options)
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return Inf
    end
    T = eltype(dataset.y)
    reference_prediction, reference_flag = eval_tree_array(tree, T.(SYMBOLIC_REFERENCE_POINT), options)
    if !reference_flag
        return Inf
    end
    energy_loss = sum(((prediction .- reference_prediction[1]) .- dataset.y) .^ 2) / dataset.n
    stress_X = T.(SYMBOLIC_STRESS_X)
    coefficients = T.(SYMBOLIC_STRESS_COEFFICIENTS)
    stress_target = T.(SYMBOLIC_STRESS_TARGET)
    _, dwd_lambda1, flag1 = eval_diff_tree_array(tree, stress_X, options, 1)
    _, dwd_lambda2, flag2 = eval_diff_tree_array(tree, stress_X, options, 2)
    _, dwd_lambda3, flag3 = eval_diff_tree_array(tree, stress_X, options, 3)
    if !flag1 || !flag2 || !flag3
        return Inf
    end
    stress_prediction = coefficients[1, :] .* dwd_lambda1 .+ coefficients[2, :] .* dwd_lambda2 .+ coefficients[3, :] .* dwd_lambda3
    stress_loss = sum((stress_prediction .- stress_target) .^ 2) / length(stress_target)
    return SYMBOLIC_ENERGY_LOSS_WEIGHT * energy_loss + SYMBOLIC_STRESS_LOSS_WEIGHT * stress_loss
end
"""
	if stress_dataset_kind == 'loadcases':
		stress_target_literal = _format_julia_vector(stress_dataset['stress_target'])
		stress_factor_literal = _format_julia_vector(stress_dataset['stress_factor'])
		dwd_i2_factor_literal = _format_julia_vector(stress_dataset['dwd_i2_factor'])
		return f"""
using Zygote

const SYMBOLIC_STRESS_X = reshape(Float64[{stress_x_literal}], {stress_rows}, {stress_cols})
const SYMBOLIC_STRESS_TARGET = Float64[{stress_target_literal}]
const SYMBOLIC_STRESS_FACTOR = Float64[{stress_factor_literal}]
const SYMBOLIC_DWDI2_FACTOR = Float64[{dwd_i2_factor_literal}]
const SYMBOLIC_REFERENCE_POINT = reshape(Float64[3.0, 3.0], {stress_rows}, 1)
const SYMBOLIC_ENERGY_LOSS_WEIGHT = {float(args.energy_loss_weight):.17g}
const SYMBOLIC_STRESS_LOSS_WEIGHT = {float(args.stress_loss_weight):.17g}

function combined_energy_stress_loss(tree, dataset, options)
	prediction, flag = eval_tree_array(tree, dataset.X, options)
	if !flag
		return Inf
	end

	T = eltype(dataset.y)
	stress_X = T.(SYMBOLIC_STRESS_X)
	stress_target = T.(SYMBOLIC_STRESS_TARGET)
	stress_factor = T.(SYMBOLIC_STRESS_FACTOR)
	dwd_i2_factor = T.(SYMBOLIC_DWDI2_FACTOR)
	reference_point = T.(SYMBOLIC_REFERENCE_POINT)

	reference_prediction, reference_flag = eval_tree_array(tree, reference_point, options)
	if !reference_flag
		return Inf
	end
	reference_energy = reference_prediction[1]
	shifted_prediction = prediction .- reference_energy
	energy_loss = sum((shifted_prediction .- dataset.y) .^ 2) / dataset.n

	_, dwd_i1, flag_i1 = eval_diff_tree_array(tree, stress_X, options, 1)
	if !flag_i1
		return Inf
	end
	_, dwd_i2, flag_i2 = eval_diff_tree_array(tree, stress_X, options, 2)
	if !flag_i2
		return Inf
	end

	symbolic_stress = stress_factor .* (dwd_i1 .+ dwd_i2_factor .* dwd_i2)
	stress_loss = sum((symbolic_stress .- stress_target) .^ 2) / length(stress_target)
	return SYMBOLIC_ENERGY_LOSS_WEIGHT * energy_loss + SYMBOLIC_STRESS_LOSS_WEIGHT * stress_loss
end
"""

	stress_target_lambda1_literal = _format_julia_vector(stress_dataset['stress_target_lambda1'])
	stress_target_lambda2_literal = _format_julia_vector(stress_dataset['stress_target_lambda2'])
	if args.feature_space == 'invariants':
		stress_factor_lambda1_literal = _format_julia_vector(stress_dataset['stress_factor_lambda1'])
		stress_factor_lambda2_literal = _format_julia_vector(stress_dataset['stress_factor_lambda2'])
		dwd_i2_factor_lambda1_literal = _format_julia_vector(stress_dataset['dwd_i2_factor_lambda1'])
		dwd_i2_factor_lambda2_literal = _format_julia_vector(stress_dataset['dwd_i2_factor_lambda2'])
		reference_point_literal = 'Float64[3.0, 3.0]'
	else:
		reference_point_literal = 'Float64[1.0, 1.0]'

	if args.feature_space == 'invariants':
		return f"""
using Zygote

const SYMBOLIC_STRESS_X = reshape(Float64[{stress_x_literal}], {stress_rows}, {stress_cols})
const SYMBOLIC_STRESS_TARGET_LAMBDA1 = Float64[{stress_target_lambda1_literal}]
const SYMBOLIC_STRESS_TARGET_LAMBDA2 = Float64[{stress_target_lambda2_literal}]
const SYMBOLIC_STRESS_FACTOR_LAMBDA1 = Float64[{stress_factor_lambda1_literal}]
const SYMBOLIC_STRESS_FACTOR_LAMBDA2 = Float64[{stress_factor_lambda2_literal}]
const SYMBOLIC_DWDI2_FACTOR_LAMBDA1 = Float64[{dwd_i2_factor_lambda1_literal}]
const SYMBOLIC_DWDI2_FACTOR_LAMBDA2 = Float64[{dwd_i2_factor_lambda2_literal}]
const SYMBOLIC_REFERENCE_POINT = reshape({reference_point_literal}, {stress_rows}, 1)
const SYMBOLIC_ENERGY_LOSS_WEIGHT = {float(args.energy_loss_weight):.17g}
const SYMBOLIC_STRESS_LOSS_WEIGHT = {float(args.stress_loss_weight):.17g}

function combined_energy_stress_loss(tree, dataset, options)
	prediction, flag = eval_tree_array(tree, dataset.X, options)
	if !flag
		return Inf
	end

	T = eltype(dataset.y)
	stress_X = T.(SYMBOLIC_STRESS_X)
	stress_target_lambda1 = T.(SYMBOLIC_STRESS_TARGET_LAMBDA1)
	stress_target_lambda2 = T.(SYMBOLIC_STRESS_TARGET_LAMBDA2)
	stress_factor_lambda1 = T.(SYMBOLIC_STRESS_FACTOR_LAMBDA1)
	stress_factor_lambda2 = T.(SYMBOLIC_STRESS_FACTOR_LAMBDA2)
	dwd_i2_factor_lambda1 = T.(SYMBOLIC_DWDI2_FACTOR_LAMBDA1)
	dwd_i2_factor_lambda2 = T.(SYMBOLIC_DWDI2_FACTOR_LAMBDA2)
	reference_point = T.(SYMBOLIC_REFERENCE_POINT)

	reference_prediction, reference_flag = eval_tree_array(tree, reference_point, options)
	if !reference_flag
		return Inf
	end
	reference_energy = reference_prediction[1]
	shifted_prediction = prediction .- reference_energy
	energy_loss = sum((shifted_prediction .- dataset.y) .^ 2) / dataset.n

	stress_prediction, dwd_i1, flag_i1 = eval_diff_tree_array(tree, stress_X, options, 1)
	if !flag_i1
		return Inf
	end
	_, dwd_i2, flag_i2 = eval_diff_tree_array(tree, stress_X, options, 2)
	if !flag_i2
		return Inf
	end

	symbolic_stress_lambda1 = stress_factor_lambda1 .* (dwd_i1 .+ dwd_i2_factor_lambda1 .* dwd_i2)
	symbolic_stress_lambda2 = stress_factor_lambda2 .* (dwd_i1 .+ dwd_i2_factor_lambda2 .* dwd_i2)
	stress_loss = (
		sum((symbolic_stress_lambda1 .- stress_target_lambda1) .^ 2)
		+ sum((symbolic_stress_lambda2 .- stress_target_lambda2) .^ 2)
	) / (2 * length(stress_target_lambda1))
	return SYMBOLIC_ENERGY_LOSS_WEIGHT * energy_loss + SYMBOLIC_STRESS_LOSS_WEIGHT * stress_loss
end
"""

	return f"""
using Zygote

const SYMBOLIC_STRESS_X = reshape(Float64[{stress_x_literal}], {stress_rows}, {stress_cols})
const SYMBOLIC_STRESS_TARGET_LAMBDA1 = Float64[{stress_target_lambda1_literal}]
const SYMBOLIC_STRESS_TARGET_LAMBDA2 = Float64[{stress_target_lambda2_literal}]
const SYMBOLIC_REFERENCE_POINT = reshape({reference_point_literal}, {stress_rows}, 1)
const SYMBOLIC_ENERGY_LOSS_WEIGHT = {float(args.energy_loss_weight):.17g}
const SYMBOLIC_STRESS_LOSS_WEIGHT = {float(args.stress_loss_weight):.17g}

function combined_energy_stress_loss(tree, dataset, options)
	prediction, flag = eval_tree_array(tree, dataset.X, options)
	if !flag
		return Inf
	end

	T = eltype(dataset.y)
	stress_X = T.(SYMBOLIC_STRESS_X)
	stress_target_lambda1 = T.(SYMBOLIC_STRESS_TARGET_LAMBDA1)
	stress_target_lambda2 = T.(SYMBOLIC_STRESS_TARGET_LAMBDA2)
	reference_point = T.(SYMBOLIC_REFERENCE_POINT)

	reference_prediction, reference_flag = eval_tree_array(tree, reference_point, options)
	if !reference_flag
		return Inf
	end
	reference_energy = reference_prediction[1]
	shifted_prediction = prediction .- reference_energy
	energy_loss = sum((shifted_prediction .- dataset.y) .^ 2) / dataset.n

	_, symbolic_stress_lambda1, flag_lambda1 = eval_diff_tree_array(tree, stress_X, options, 1)
	if !flag_lambda1
		return Inf
	end
	_, symbolic_stress_lambda2, flag_lambda2 = eval_diff_tree_array(tree, stress_X, options, 2)
	if !flag_lambda2
		return Inf
	end

	stress_loss = (
		sum((symbolic_stress_lambda1 .- stress_target_lambda1) .^ 2)
		+ sum((symbolic_stress_lambda2 .- stress_target_lambda2) .^ 2)
	) / (2 * length(stress_target_lambda1))
	return SYMBOLIC_ENERGY_LOSS_WEIGHT * energy_loss + SYMBOLIC_STRESS_LOSS_WEIGHT * stress_loss
end
"""


def _build_valanis_landel_loss_function(args, data: np.ndarray) -> str:
	if not {'stretch', 'lambda1', 'lambda2', 'lambda3', 'nominal_stress', 'mode_id'}.issubset(data.dtype.names or ()):
		raise ValueError('--valanis-landel currently requires a loadcase export with UT/PS/ET nominal stresses.')
	stress_dataset = build_loadcase_stress_dataset(data, 'stretches')
	features = np.asarray(stress_dataset['features'], dtype=np.float64)
	coefficients = np.asarray(stress_dataset['stress_coefficients'], dtype=np.float64)
	energy_targets = np.asarray(data['energy'], dtype=np.float64)
	stress_targets = np.asarray(stress_dataset['stress_target'], dtype=np.float64)
	lambda_literals = [_format_julia_vector(features[:, index]) for index in range(3)]
	coefficient_literals = [_format_julia_vector(coefficients[:, index]) for index in range(3)]
	energy_literal = _format_julia_vector(energy_targets)
	stress_literal = _format_julia_vector(stress_targets)
	return f"""
begin
    using Zygote

    _lambda1_full = [{lambda_literals[0]}]
    _lambda2_full = [{lambda_literals[1]}]
    _lambda3_full = [{lambda_literals[2]}]
    _coeff1_full = [{coefficient_literals[0]}]
    _coeff2_full = [{coefficient_literals[1]}]
    _coeff3_full = [{coefficient_literals[2]}]
    _energy_full = [{energy_literal}]
    _stress_full = [{stress_literal}]
    _energy_weight = {float(args.energy_loss_weight):.17g}
    _stress_weight = {float(args.stress_loss_weight):.17g}

    function valanis_landel_loss(tree, dataset::Dataset{{T,L}}, options, idx=nothing)::L where {{T,L}}
        use_idx = isnothing(idx) ? (1:dataset.n) : idx
        lambda1 = T.(isnothing(idx) ? _lambda1_full : _lambda1_full[use_idx])
        lambda2 = T.(isnothing(idx) ? _lambda2_full : _lambda2_full[use_idx])
        lambda3 = T.(isnothing(idx) ? _lambda3_full : _lambda3_full[use_idx])
        coeff1 = T.(isnothing(idx) ? _coeff1_full : _coeff1_full[use_idx])
        coeff2 = T.(isnothing(idx) ? _coeff2_full : _coeff2_full[use_idx])
        coeff3 = T.(isnothing(idx) ? _coeff3_full : _coeff3_full[use_idx])
        energy_target = T.(isnothing(idx) ? _energy_full : _energy_full[use_idx])
        stress_target = T.(isnothing(idx) ? _stress_full : _stress_full[use_idx])
        x1, x2, x3 = reshape(lambda1, 1, :), reshape(lambda2, 1, :), reshape(lambda3, 1, :)
        w1, ok1 = eval_tree_array(tree, x1, options)
        w2, ok2 = eval_tree_array(tree, x2, options)
        w3, ok3 = eval_tree_array(tree, x3, options)
        if !(ok1 && ok2 && ok3)
            return L(Inf)
        end
        energy_prediction = w1 .+ w2 .+ w3
        if any(x -> !isfinite(x), energy_prediction)
            return L(Inf)
        end
        energy_mse = sum((energy_prediction .- energy_target) .^ 2) / length(energy_target)
        if _stress_weight == 0.0
            return T(_energy_weight) * energy_mse
        end
        _, dw1, okd1 = eval_diff_tree_array(tree, x1, options, 1)
        _, dw2, okd2 = eval_diff_tree_array(tree, x2, options, 1)
        _, dw3, okd3 = eval_diff_tree_array(tree, x3, options, 1)
        if !(okd1 && okd2 && okd3)
            return L(Inf)
        end
        stress_prediction = coeff1 .* dw1 .+ coeff2 .* dw2 .+ coeff3 .* dw3
        if any(x -> !isfinite(x), stress_prediction)
            return L(Inf)
        end
        stress_mse = sum((stress_prediction .- stress_target) .^ 2) / length(stress_target)
        return T(_energy_weight) * energy_mse + T(_stress_weight) * stress_mse
    end
    valanis_landel_loss
end
"""


def _reconstruct_valanis_landel_expression(single_stretch_expression: str) -> str:
	lam, lambda1, lambda2, lambda3 = sympy.symbols('lam lambda1 lambda2 lambda3')
	expression = sympy.sympify(single_stretch_expression, locals={'square': lambda value: value ** 2})
	return str(sympy.simplify(
		expression.subs(lam, lambda1) + expression.subs(lam, lambda2) + expression.subs(lam, lambda3)
	))


def fit_symbolic_expression(args) -> None:
	if args.energy_loss_weight < 0.0 or args.stress_loss_weight < 0.0:
		raise ValueError('--energy-loss-weight and --stress-loss-weight must be non-negative.')
	if args.energy_loss_weight == 0.0 and args.stress_loss_weight == 0.0:
		raise ValueError('At least one symbolic-regression loss weight must be positive.')
	add_local_import_paths()
	try:
		PySRRegressor = importlib.import_module('pysr').PySRRegressor
	except ImportError as error:
		raise ImportError(
			'Failed to import pysr. Use an environment where PySR is installed, '
			'or activate SymbolicRegression/PySR-master/.venv after installing it.'
		) from error

	data = load_dataset(args.input_csv)
	if args.valanis_landel and args.feature_space != 'stretches':
		raise ValueError('--valanis-landel requires --feature-space stretches.')
	if args.feature_space == 'invariants':
		feature_names = ['I1', 'I2']
		features = np.column_stack([data['I1'], data['I2']])
	elif args.valanis_landel:
		feature_names = ['lam']
		features = np.asarray(data['lambda1'], dtype=np.float64).reshape(-1, 1)
	else:
		feature_names = ['lambda1', 'lambda2', 'lambda3']
		features = np.column_stack([data['lambda1'], data['lambda2'], data['lambda3']])
	target = np.asarray(data['energy'], dtype=np.float64)

	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	use_stress_loss = args.stress_loss_weight > 0.0
	stress_dataset: dict[str, np.ndarray] | None = None
	custom_loss_function: str | None = None
	if args.valanis_landel:
		custom_loss_function = _build_valanis_landel_loss_function(args, data)
	elif use_stress_loss:
		stress_dataset = build_stress_dataset(data, args.feature_space)
		custom_loss_function = _build_combined_loss_function(args, stress_dataset)

	filtered_unary_operators = [
		operator for operator in args.unary_operators
		if str(operator).strip().lower() != 'log'
	]
	binary_operators = list(dict.fromkeys(str(operator) for operator in args.binary_operators))
	if args.feature_space == 'stretches':
		if '^' not in binary_operators:
			binary_operators.append('^')

	model_kwargs = dict(
		model_selection='best',
		niterations=args.niterations,
		population_size=args.population_size,
		populations=args.populations,
		tournament_selection_n=max(2, min(10, args.population_size - 1)),
		maxsize=args.maxsize,
		binary_operators=binary_operators,
		unary_operators=filtered_unary_operators,
		random_state=args.seed,
		progress=True,
		verbosity=1,
		print_precision=5,
		output_directory=str(output_dir.parent),
		run_id=output_dir.name,
	)
	if args.feature_space == 'stretches':
		model_kwargs.update(constraints={'^': (1, 1)})
	if custom_loss_function is not None:
		model_kwargs.update(loss_function=custom_loss_function)
	else:
		model_kwargs.update(elementwise_loss='loss(prediction, target) = (prediction - target)^2')
	if args.deterministic:
		model_kwargs.update(deterministic=True, parallelism='serial')

	model = PySRRegressor(**model_kwargs)
	model.fit(features, target, variable_names=feature_names)
	contour_plot_path: Path | None = None
	symbolic_contour_plot_path: Path | None = None
	experiment_prediction_outputs: dict[str, Path | dict[str, dict[str, float]]] | None = None
	simplification_summary: dict[str, object] | None = None
	material_package_path = args.material_package_json or (output_dir / 'material_package.json')

	best_row = model.get_best()
	best_equation = str(best_row.get('equation', best_row.get('sympy_format', ''))) or str(best_row)
	if hasattr(model, 'sympy'):
		try:
			best_equation = str(model.sympy())
		except Exception:
			pass
	if args.valanis_landel:
		best_equation = _reconstruct_valanis_landel_expression(best_equation)
	normalization = normalize_expression_to_zero_initial_state(best_equation, args.feature_space)
	best_equation = str(normalization['normalized_expression'])
	prediction_features = (
		np.column_stack([data['lambda1'], data['lambda2'], data['lambda3']])
		if args.valanis_landel else features
	)
	if args.simplify_expression:
		(output_dir / 'raw_best_equation.txt').write_text(
			'Raw normalized PySR expression before post-fit simplification\n'
			f'expression: {best_equation}\n',
			encoding='utf-8',
		)
		try:
			simplification_summary = simplify_and_refit_expression(
				best_equation, data, prediction_features, args,
			)
			(output_dir / 'simplification_summary.json').write_text(
				json.dumps(simplification_summary, indent=2) + '\n',
				encoding='utf-8',
			)
			best_equation = str(simplification_summary['simplified_expression'])
		except (FileNotFoundError, ValueError, np.linalg.LinAlgError) as error:
			simplification_summary = {'changed': False, 'skipped_reason': str(error)}
			(output_dir / 'simplification_summary.json').write_text(
				json.dumps(simplification_summary, indent=2) + '\n',
				encoding='utf-8',
			)
			print(f'Skipped post-fit expression simplification: {error}')
	predictions = evaluate_expression_on_features(best_equation, args.feature_space, prediction_features)
	rmse = float(np.sqrt(np.mean((predictions - target) ** 2)))
	denominator = float(np.sum((target - target.mean()) ** 2))
	r2 = float(1.0 - np.sum((predictions - target) ** 2) / denominator) if denominator > 0.0 else float('nan')
	initial_state_check = evaluate_initial_state_energy(best_equation, args.feature_space)

	(output_dir / 'best_equation.txt').write_text(
		'Best symbolic strain-energy expression\n'
		f'feature_space: {args.feature_space}\n'
		f'rmse: {rmse:.10g}\n'
		f'r2: {r2:.10g}\n'
		f'raw_initial_state_energy: {float(normalization["raw_initial_energy"]):.10g}\n'
		f'initial_state_energy: {initial_state_check["initial_energy"]:.10g}\n'
		f'initial_state_energy_is_zero: {initial_state_check["is_initial_energy_zero"]}\n'
		f'expression: {best_equation}\n',
		encoding='utf-8',
	)

	with (output_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
		writer = csv.writer(handle)
		if {'stretch', 'nominal_stress'}.issubset(set(data.dtype.names or ())):
			writer.writerow([
				'stretch', 'lambda1', 'lambda2', 'lambda3', 'I1', 'I2', *feature_names,
				'energy_target', 'energy_prediction', 'nominal_stress_target',
			])
			for index in range(len(target)):
				writer.writerow([
					f'{float(data["stretch"][index]):.17g}',
					f'{float(data["lambda1"][index]):.17g}',
					f'{float(data["lambda2"][index]):.17g}',
					f'{float(data["lambda3"][index]):.17g}',
					f'{float(data["I1"][index]):.17g}',
					f'{float(data["I2"][index]):.17g}',
					*(f'{float(value):.17g}' for value in features[index]),
					f'{float(target[index]):.17g}',
					f'{float(predictions[index]):.17g}',
					f'{float(data["nominal_stress"][index]):.17g}',
				])
		else:
			writer.writerow(['lambda1', 'lambda2', 'lambda3', 'I1', 'I2', *feature_names, 'energy_target', 'energy_prediction'])
			for index in range(len(target)):
				writer.writerow([
					f'{float(data["lambda1"][index]):.17g}',
					f'{float(data["lambda2"][index]):.17g}',
					f'{float(data["lambda3"][index]):.17g}',
					f'{float(data["I1"][index]):.17g}',
					f'{float(data["I2"][index]):.17g}',
					*(f'{float(value):.17g}' for value in features[index]),
					f'{float(target[index]):.17g}',
					f'{float(predictions[index]):.17g}',
				])

	try:
		contour_plot_path = save_energy_comparison_plot(data, predictions, output_dir)
	except ValueError as error:
		print(f'Skipped contour plot generation: {error}')

	try:
		symbolic_contour_plot_path = save_symbolic_energy_contour_plot(
			best_equation,
			args.feature_space,
			output_dir,
		)
	except ValueError as error:
		print(f'Skipped symbolic energy contour generation: {error}')

	if not args.skip_experiment_prediction:
		try:
			experiment_prediction_outputs = generate_experiment_predictions(args, expression_text=best_equation)
		except (FileNotFoundError, NotImplementedError, ValueError) as error:
			print(f'Skipped reference-data prediction: {error}')

	if args.feature_space == 'invariants':
		material_package_path = write_material_package(
			expression_text=best_equation,
			feature_space=args.feature_space,
			data=data,
			args=args,
			rmse=rmse,
			r2=r2,
			initial_state_check=initial_state_check,
			output_path=material_package_path,
		)
	else:
		print('Skipped OpenRadioss material-package export: it currently requires invariant expressions.')
		material_package_path = None

	(output_dir / 'fit_summary.json').write_text(
		json.dumps(
			{
				'input_csv': str(args.input_csv.resolve()),
				'feature_space': args.feature_space,
				'rmse': rmse,
				'r2': r2,
				'niterations': args.niterations,
				'population_size': args.population_size,
				'populations': args.populations,
				'maxsize': args.maxsize,
				'unary_operators': filtered_unary_operators,
				'binary_operators': binary_operators,
				'energy_loss_weight': args.energy_loss_weight,
				'stress_loss_weight': args.stress_loss_weight,
				'simplify_expression': args.simplify_expression,
				'simplification': simplification_summary,
				'seed': args.seed,
				'best_equation': best_equation,
				'raw_initial_state_energy': normalization['raw_initial_energy'],
				'initial_state_energy': initial_state_check['initial_energy'],
				'abs_initial_state_energy': initial_state_check['abs_initial_energy'],
				'initial_state_energy_is_zero': initial_state_check['is_initial_energy_zero'],
				'energy_contour_plot': str(contour_plot_path.resolve()) if contour_plot_path is not None else None,
				'symbolic_energy_contour_plot': (
					str(symbolic_contour_plot_path.resolve())
					if symbolic_contour_plot_path is not None else None
				),
				'experiment_prediction_csv': (
					str(experiment_prediction_outputs['csv_path'].resolve())
					if experiment_prediction_outputs is not None else None
				),
				'experiment_prediction_plot': (
					str(experiment_prediction_outputs['plot_path'].resolve())
					if experiment_prediction_outputs is not None else None
				),
				'experiment_prediction_summary': (
					str(experiment_prediction_outputs['summary_path'].resolve())
					if experiment_prediction_outputs is not None else None
				),
				'experiment_stress_loss': (
					float(experiment_prediction_outputs['metrics']['overall']['stress_loss'])
					if experiment_prediction_outputs is not None else None
				),
				'experiment_stress_rmse': (
					float(experiment_prediction_outputs['metrics']['overall']['rmse'])
					if experiment_prediction_outputs is not None else None
				),
				'material_package_json': str(material_package_path.resolve()) if material_package_path is not None else None,
			},
			indent=2,
		) + '\n',
		encoding='utf-8',
	)

	print(f'PySR fit complete. Best equation written to {output_dir / "best_equation.txt"}.')
	if simplification_summary is not None and simplification_summary.get('changed'):
		print(
			'Post-fit simplification accepted: removed '
			f'{len(simplification_summary["removed_terms"])} term(s) and refitted the remaining coefficients.'
		)
	print(
		'Normalized final symbolic expression to enforce zero reference energy: '
		f'raw W(reference state) = {float(normalization["raw_initial_energy"]):.10g}.'
	)
	if initial_state_check['is_initial_energy_zero']:
		print('Initial-state energy check passed: W(reference state) is approximately zero.')
	else:
		print(
			'Initial-state energy check failed: '
			f'W(reference state) = {initial_state_check["initial_energy"]:.10g}. '
			'The symbolic expression does not preserve zero reference energy.'
		)
	if contour_plot_path is not None:
		print(f'Energy contour comparison written to {contour_plot_path}.')
	if symbolic_contour_plot_path is not None:
		print(f'Symbolic energy contour written to {symbolic_contour_plot_path}.')
	if experiment_prediction_outputs is not None:
		print(f'Reference-data predictions written to {experiment_prediction_outputs["csv_path"]}.')
	if material_package_path is not None:
		print(f'Material package written to {material_package_path}.')
