#!/usr/bin/env python3
"""Summarize and plot several archived Mooney--Rivlin noise experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from preprocess.synthetic_data import mooney_rivlin_to_ogden_terms, ogden_nominal_stress


def expression_from_file(path: Path) -> str:
	for line in path.read_text(encoding='utf-8').splitlines():
		if line.startswith('expression:'):
			return line.split(':', 1)[1].strip()
	raise ValueError(f'No expression found in {path}')


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--study-dir', type=Path, required=True)
	parser.add_argument('--noise-levels', nargs='+', default=['0', '0.05', '0.1'])
	args = parser.parse_args()
	mu_terms, alpha_terms = mooney_rivlin_to_ogden_terms(0.18, 0.02)
	colors = {'UT': 'tab:blue', 'PS': 'tab:orange', 'ET': 'tab:green'}
	figure, axes = plt.subplots(1, len(args.noise_levels), figsize=(5.2 * len(args.noise_levels), 4.6), sharey=True)
	summary = {}

	for axis, noise in zip(np.atleast_1d(axes), args.noise_levels):
		case_dir = args.study_dir / f'noise_{noise}'
		data = np.genfromtxt(
			case_dir / 'SR_output' / 'experiment_predictions.csv',
			delimiter=',', names=True, dtype=None, encoding='utf-8',
		)
		clean = np.empty(len(data), dtype=np.float64)
		per_mode = {}
		for mode in ('UT', 'PS', 'ET'):
			mask = data['mode'] == mode
			stretch = np.asarray(data['stretch'][mask], dtype=np.float64)
			noisy = np.asarray(data['stress_target'][mask], dtype=np.float64)
			prediction = np.asarray(data['stress_prediction'][mask], dtype=np.float64)
			truth = ogden_nominal_stress(stretch.astype(np.float32), mode, mu_terms, alpha_terms).astype(np.float64)
			clean[mask] = truth
			order = np.argsort(stretch)
			axis.scatter(stretch, noisy, s=17, facecolors='none', edgecolors=colors[mode], alpha=0.55)
			axis.plot(stretch[order], prediction[order], color=colors[mode], linewidth=2, label=mode)
			per_mode[mode] = {
				'noisy_rmse': float(np.sqrt(np.mean((prediction - noisy) ** 2))),
				'clean_rmse': float(np.sqrt(np.mean((prediction - truth) ** 2))),
			}
		prediction = np.asarray(data['stress_prediction'], dtype=np.float64)
		noisy = np.asarray(data['stress_target'], dtype=np.float64)
		summary[noise] = {
			'expression': expression_from_file(case_dir / 'SR_output' / 'best_equation.txt'),
			'noisy_rmse': float(np.sqrt(np.mean((prediction - noisy) ** 2))),
			'clean_rmse': float(np.sqrt(np.mean((prediction - clean) ** 2))),
			'per_mode': per_mode,
		}
		axis.set_title(f'Noise std = {noise}')
		axis.set_xlabel(r'$\lambda_1$')
		axis.grid(alpha=0.25)

	axes = np.atleast_1d(axes)
	axes[0].set_ylabel(r'$P_{11}$ (MPa)')
	axes[-1].legend(loc='upper left')
	figure.tight_layout()
	figure.savefig(args.study_dir / 'stress_strain_noise_comparison.png', dpi=220)
	figure.savefig(args.study_dir / 'stress_strain_noise_comparison.pdf')
	plt.close(figure)
	(args.study_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
	main()
