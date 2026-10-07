#!/usr/bin/env python
"""Evaluate every hall-of-fame candidate on the experimental UT/PS/ET data.

For each PySR Pareto-front expression this script reports the experimental
nominal-stress RMSE (i) with the raw PySR coefficients, (ii) after refitting the
linear coefficients on the experimental stresses, and (iii) after the greedy
contribution-aware pruning used by the main pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.experimental_prediction import (  # noqa: E402
	_EXPRESSION_LOCALS,
	build_experiment_stress_dataset,
	normalize_expression_to_zero_initial_state,
)
from modules.fitting import load_dataset  # noqa: E402
from modules.simplification import (  # noqa: E402
	_build_energy_matrix,
	_build_stress_matrix,
	_expression,
	_fit_coefficients,
	_split_linear_additive_terms,
	simplify_and_refit_expression,
)

DATA_ROOT = Path('/home/guanjs/NN-constitutive/FFNN/fitting-data-PK')


def _mode_slices(stress_dataset, args):
	from modules.experimental_prediction import _load_preprocess_helpers
	_, load_mode_data = _load_preprocess_helpers()
	slices, start = {}, 0
	for mode, path in (('UT', args.ut_dataset), ('PS', args.ps_dataset), ('ET', args.et_dataset)):
		n = len(load_mode_data(mode, path)['stretch'])
		slices[mode] = slice(start, start + n)
		start += n
	return slices


def _stress_from_expression(expression_text, dataset):
	i1, i2 = sympy.symbols('I1 I2')
	expression = sympy.sympify(expression_text, locals=_EXPRESSION_LOCALS)
	d1 = sympy.lambdify((i1, i2), sympy.diff(expression, i1), modules='numpy')
	d2 = sympy.lambdify((i1, i2), sympy.diff(expression, i2), modules='numpy')
	inv = dataset['invariants']
	v1 = np.broadcast_to(np.asarray(d1(inv[:, 0], inv[:, 1]), dtype=np.float64), (len(inv),))
	v2 = np.broadcast_to(np.asarray(d2(inv[:, 0], inv[:, 1]), dtype=np.float64), (len(inv),))
	return dataset['stress_factor'] * (v1 + dataset['dwd_i2_factor'] * v2)


def _rmse_by_mode(prediction, dataset, slices):
	target = dataset['stress_target']
	out = {m: float(np.sqrt(np.mean((prediction[s] - target[s]) ** 2))) for m, s in slices.items()}
	out['overall'] = float(np.sqrt(np.mean((prediction - target) ** 2)))
	return out


def _refit_expression(expression_text, data, features, dataset, args):
	terms = _split_linear_additive_terms(expression_text)
	energy_matrix = _build_energy_matrix(terms, features)
	stress_matrix = _build_stress_matrix(terms, dataset)
	active = list(range(len(terms)))
	coefficients = _fit_coefficients(
		active, energy_matrix, np.asarray(data['energy'], dtype=np.float64),
		stress_matrix, dataset['stress_target'], args.simplify_energy_weight, args.simplify_stress_weight,
	)
	return _expression(terms, active, coefficients), len(terms)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True, help='Treloar_1944 or Yohsuke_2011')
	parser.add_argument('--hall-of-fame', type=Path, nargs='+', required=True)
	parser.add_argument('--input-csv', type=Path, required=True, help='NN loadcase samples used for the search')
	parser.add_argument('--tolerance', type=float, default=0.02)
	parser.add_argument('--output-json', type=Path, default=None)
	cli = parser.parse_args()

	args = SimpleNamespace(
		feature_space='invariants',
		simplify_energy_weight=0.0,
		simplify_stress_weight=1.0,
		simplify_rmse_tolerance=cli.tolerance,
		ut_dataset=DATA_ROOT / cli.dataset / 'UT' / 'stress_stretch.txt',
		ps_dataset=DATA_ROOT / cli.dataset / 'PS' / 'stress_stretch.txt',
		et_dataset=DATA_ROOT / cli.dataset / 'ET' / 'stress_stretch.txt',
		enable_biaxial=False,
	)
	dataset = build_experiment_stress_dataset(args)
	slices = _mode_slices(dataset, args)
	data = load_dataset(cli.input_csv)
	features = np.column_stack([data['I1'], data['I2']]).astype(np.float64)

	rows = []
	for hof in cli.hall_of_fame:
		with hof.open(newline='') as handle:
			for record in csv.DictReader(handle):
				raw = record['Equation']
				try:
					norm = normalize_expression_to_zero_initial_state(raw, 'invariants')
					expr = str(norm['normalized_expression'])
					raw_rmse = _rmse_by_mode(_stress_from_expression(expr, dataset), dataset, slices)
				except Exception as error:  # noqa: BLE001
					print(f'skip {hof.parent.name} c={record["Complexity"]}: {error}')
					continue
				entry = {
					'source': hof.parent.name,
					'complexity': int(record['Complexity']),
					'search_loss': float(record['Loss']),
					'expression': expr,
					'raw_rmse': raw_rmse,
				}
				try:
					refit_expr, n_terms = _refit_expression(expr, data, features, dataset, args)
					entry['refit_expression'] = refit_expr
					entry['n_terms'] = n_terms
					entry['refit_rmse'] = _rmse_by_mode(_stress_from_expression(refit_expr, dataset), dataset, slices)
					summary = simplify_and_refit_expression(expr, data, features, args)
					pruned = str(summary['simplified_expression'])
					entry['pruned_expression'] = pruned
					entry['pruned_terms'] = n_terms - len(summary['removed_terms'])
					entry['pruned_rmse'] = _rmse_by_mode(_stress_from_expression(pruned, dataset), dataset, slices)
				except Exception as error:  # noqa: BLE001
					entry['refit_error'] = str(error)
				rows.append(entry)

	rows.sort(key=lambda r: (r['complexity'], r['search_loss']))
	print(f'\n=== {cli.dataset}: experimental nominal-stress RMSE (MPa) ===')
	print(f'{"src":>22} {"cx":>3} {"raw":>8} {"refit":>8} {"pruned":>8} {"nt":>2}  expression (pruned/refit)')
	for r in rows:
		refit = r.get('refit_rmse', {}).get('overall', float('nan'))
		pruned = r.get('pruned_rmse', {}).get('overall', float('nan'))
		expr = r.get('pruned_expression') or r.get('refit_expression') or r['expression']
		print(f'{r["source"]:>22} {r["complexity"]:>3} {r["raw_rmse"]["overall"]:8.4f} {refit:8.4f} {pruned:8.4f} {r.get("pruned_terms", r.get("n_terms", "-")):>2}  {expr}')
	if cli.output_json:
		cli.output_json.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
		print(f'written {cli.output_json}')


if __name__ == '__main__':
	main()