#!/usr/bin/env python3
"""Re-evaluate an invariant SR expression on an existing prediction CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument('--baseline-csv', type=Path, required=True)
	parser.add_argument('--equation-file', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	return parser.parse_args()


def load_expression(path: Path) -> str:
	for line in path.read_text(encoding='utf-8').splitlines():
		if line.startswith('expression:'):
			return line.split(':', 1)[1].strip()
	raise ValueError(f'No expression line found in {path}')


def main() -> None:
	args = parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)
	with args.baseline_csv.open(newline='', encoding='utf-8') as handle:
		rows = list(csv.DictReader(handle))

	i1, i2 = sympy.symbols('I1 I2')
	expression = sympy.sympify(load_expression(args.equation_file))
	d1_expr, d2_expr = sympy.diff(expression, i1), sympy.diff(expression, i2)
	d1_fn = sympy.lambdify((i1, i2), d1_expr, modules='numpy')
	d2_fn = sympy.lambdify((i1, i2), d2_expr, modules='numpy')

	for row in rows:
		stretch = float(row['stretch'])
		d1 = float(d1_fn(float(row['I1']), float(row['I2'])))
		d2 = float(d2_fn(float(row['I1']), float(row['I2'])))
		if row['mode'] == 'UT':
			factor, d2_factor = 2 * (stretch - stretch ** -2), stretch ** -1
		elif row['mode'] == 'PS':
			factor, d2_factor = 2 * (stretch - stretch ** -3), 1.0
		elif row['mode'] == 'ET':
			factor, d2_factor = 2 * (stretch - stretch ** -5), stretch ** 2
		else:
			raise ValueError(f"Unsupported mode {row['mode']}")
		row['dW_dI1'] = f'{d1:.17g}'
		row['dW_dI2'] = f'{d2:.17g}'
		row['stress_prediction'] = f'{factor * (d1 + d2_factor * d2):.17g}'

	fieldnames = list(rows[0])
	with (args.output_dir / 'experiment_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	metrics: dict[str, dict[str, float]] = {}
	all_errors = []
	fig, axis = plt.subplots(figsize=(7, 5))
	for mode in ('UT', 'PS', 'ET'):
		mode_rows = [row for row in rows if row['mode'] == mode]
		x = np.array([float(row['stretch']) for row in mode_rows])
		y = np.array([float(row['stress_target']) for row in mode_rows])
		prediction = np.array([float(row['stress_prediction']) for row in mode_rows])
		error = prediction - y
		all_errors.extend(error)
		mse = float(np.mean(error ** 2))
		denominator = float(np.sum((y - np.mean(y)) ** 2))
		metrics[mode] = {
			'stress_loss': mse,
			'rmse': float(np.sqrt(mse)),
			'r2': float(1 - np.sum(error ** 2) / denominator),
			'sample_count': float(len(y)),
		}
		axis.scatter(x, y, s=13, alpha=0.55, label=f'{mode} target')
		order = np.argsort(x)
		axis.plot(x[order], prediction[order], linewidth=1.8, label=f'{mode} pruned SR')

	all_errors_array = np.asarray(all_errors)
	overall_mse = float(np.mean(all_errors_array ** 2))
	metrics['overall'] = {
		'stress_loss': overall_mse,
		'rmse': float(np.sqrt(overall_mse)),
		'sample_count': float(len(all_errors_array)),
	}
	(args.output_dir / 'experiment_prediction_summary.json').write_text(
		json.dumps(metrics, indent=2) + '\n', encoding='utf-8'
	)
	axis.set_xlabel('Stretch')
	axis.set_ylabel('Nominal stress')
	axis.grid(alpha=0.25)
	axis.legend(ncol=2, fontsize=8)
	fig.tight_layout()
	fig.savefig(args.output_dir / 'experiment_predictions.png', dpi=200)
	plt.close(fig)


if __name__ == '__main__':
	main()
