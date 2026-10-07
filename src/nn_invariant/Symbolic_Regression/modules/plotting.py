from __future__ import annotations

from pathlib import Path

import numpy as np
import sympy


_EXPRESSION_LOCALS = {
	'square': lambda value: value ** 2,
	'cube': lambda value: value ** 3,
	'sqrt': sympy.sqrt,
	'exp': sympy.exp,
	'log': sympy.log,
	'abs': sympy.Abs,
}


def infer_lambda_grid(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	lambda1_values = np.asarray(data['lambda1'], dtype=np.float64)
	lambda2_values = np.asarray(data['lambda2'], dtype=np.float64)
	unique_lambda1 = np.unique(lambda1_values)
	unique_lambda2 = np.unique(lambda2_values)
	grid_size = unique_lambda1.size * unique_lambda2.size
	if grid_size != len(data):
		raise ValueError(
			'Input CSV does not form a complete lambda1-lambda2 grid, so contour plots cannot be generated.'
		)

	lambda1_grid, lambda2_grid = np.meshgrid(unique_lambda1, unique_lambda2, indexing='ij')
	return lambda1_values, lambda2_values, lambda1_grid, lambda2_grid


def reshape_on_grid(
	data: np.ndarray,
	values: np.ndarray,
	unique_lambda1: np.ndarray,
	unique_lambda2: np.ndarray,
) -> np.ndarray:
	grid = np.full((unique_lambda1.size, unique_lambda2.size), np.nan, dtype=np.float64)
	lambda1_indices = {float(value): index for index, value in enumerate(unique_lambda1)}
	lambda2_indices = {float(value): index for index, value in enumerate(unique_lambda2)}

	for lambda1_value, lambda2_value, field_value in zip(
		data['lambda1'],
		data['lambda2'],
		values,
		strict=True,
	):
		row_index = lambda1_indices[float(lambda1_value)]
		column_index = lambda2_indices[float(lambda2_value)]
		grid[row_index, column_index] = float(field_value)

	if np.isnan(grid).any():
		raise ValueError('Failed to reconstruct a dense lambda1-lambda2 grid for contour plotting.')

	return grid


def save_energy_comparison_plot(
	data: np.ndarray,
	energy_predictions: np.ndarray,
	output_dir: Path,
) -> Path:
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
	except ImportError as error:
		raise ImportError(
			'matplotlib is required to generate contour plots for the symbolic regression fit.'
		) from error

	unique_lambda1 = np.unique(np.asarray(data['lambda1'], dtype=np.float64))
	unique_lambda2 = np.unique(np.asarray(data['lambda2'], dtype=np.float64))
	_, _, lambda1_grid, lambda2_grid = infer_lambda_grid(data)
	energy_target_grid = reshape_on_grid(data, np.asarray(data['energy'], dtype=np.float64), unique_lambda1, unique_lambda2)
	energy_prediction_grid = reshape_on_grid(data, np.asarray(energy_predictions, dtype=np.float64), unique_lambda1, unique_lambda2)
	error_grid = np.abs(energy_prediction_grid - energy_target_grid)

	figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
	plot_specs = [
		(axes[0], energy_target_grid, 'NN Energy', 'viridis'),
		(axes[1], energy_prediction_grid, 'Symbolic Energy', 'viridis'),
		(axes[2], error_grid, 'Absolute Error', 'magma'),
	]

	for axis, field_grid, title, colormap in plot_specs:
		filled = axis.contourf(lambda1_grid, lambda2_grid, field_grid, levels=20, cmap=colormap)
		contours = axis.contour(lambda1_grid, lambda2_grid, field_grid, levels=10, colors='white', linewidths=0.4, alpha=0.7)
		axis.clabel(contours, inline=True, fontsize=7, fmt='%.2f')
		axis.set_title(title)
		axis.set_xlabel('lambda1')
		axis.set_ylabel('lambda2')
		figure.colorbar(filled, ax=axis)

	output_dir.mkdir(parents=True, exist_ok=True)
	plot_path = output_dir / 'energy_comparison_contours.png'
	figure.savefig(plot_path, dpi=200, bbox_inches='tight')
	plt.close(figure)
	return plot_path


def save_symbolic_energy_contour_plot(
	expression_text: str,
	feature_space: str,
	output_dir: Path,
	lambda1_range: tuple[float, float] = (0.5, 3.0),
	lambda2_range: tuple[float, float] = (0.5, 3.0),
	grid_points: int = 120,
	contour_levels: int = 20,
) -> Path:
	"""Plot symbolic energy on an incompressible principal-stretch grid."""
	if grid_points < 2:
		raise ValueError('grid_points must be at least 2.')
	if feature_space not in ('invariants', 'stretches'):
		raise ValueError(f'Unsupported feature space: {feature_space}.')

	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
	except ImportError as error:
		raise ImportError('matplotlib is required to generate the symbolic energy contour.') from error

	lambda1_values = np.linspace(*lambda1_range, grid_points, dtype=np.float64)
	lambda2_values = np.linspace(*lambda2_range, grid_points, dtype=np.float64)
	lambda1_grid, lambda2_grid = np.meshgrid(lambda1_values, lambda2_values, indexing='xy')
	lambda3_grid = 1.0 / (lambda1_grid * lambda2_grid)

	expression = sympy.sympify(expression_text, locals=_EXPRESSION_LOCALS)
	if feature_space == 'invariants':
		i1_grid = lambda1_grid ** 2 + lambda2_grid ** 2 + lambda3_grid ** 2
		i2_grid = (
			lambda1_grid ** 2 * lambda2_grid ** 2
			+ lambda2_grid ** 2 * lambda3_grid ** 2
			+ lambda3_grid ** 2 * lambda1_grid ** 2
		)
		i1_symbol, i2_symbol = sympy.symbols('I1 I2')
		energy_function = sympy.lambdify((i1_symbol, i2_symbol), expression, modules='numpy')
		energy_grid = energy_function(i1_grid, i2_grid)
	else:
		lambda1_symbol, lambda2_symbol, lambda3_symbol = sympy.symbols('lambda1 lambda2 lambda3')
		energy_function = sympy.lambdify(
			(lambda1_symbol, lambda2_symbol, lambda3_symbol), expression, modules='numpy'
		)
		energy_grid = energy_function(lambda1_grid, lambda2_grid, lambda3_grid)

	energy_grid = np.asarray(energy_grid, dtype=np.float64)
	if energy_grid.ndim == 0:
		energy_grid = np.full_like(lambda1_grid, float(energy_grid))
	else:
		energy_grid = np.broadcast_to(energy_grid, lambda1_grid.shape).copy()
	energy_grid[~np.isfinite(energy_grid)] = np.nan
	if np.isnan(energy_grid).all():
		raise ValueError('The symbolic energy is non-finite over the entire contour grid.')

	figure, axis = plt.subplots(figsize=(8, 6))
	filled = axis.contourf(
		lambda1_grid, lambda2_grid, energy_grid, levels=contour_levels, cmap='viridis'
	)
	lines = axis.contour(
		lambda1_grid, lambda2_grid, energy_grid, levels=10, colors='white', linewidths=0.4
	)
	axis.clabel(lines, inline=True, fontsize=9, fmt='%.3g')
	axis.set_xlabel(r'$\lambda_1$', fontsize=25)
	axis.set_ylabel(r'$\lambda_2$', fontsize=25)
	axis.tick_params(axis='both', labelsize=20)
	colorbar = figure.colorbar(filled, ax=axis)
	colorbar.set_label(r'$W$', fontsize=20)
	colorbar.ax.tick_params(labelsize=16)
	figure.tight_layout()

	output_dir.mkdir(parents=True, exist_ok=True)
	plot_path = output_dir / 'symbolic_energy_contour.png'
	figure.savefig(plot_path, dpi=220, bbox_inches='tight')
	plt.close(figure)
	return plot_path
