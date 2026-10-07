#!/usr/bin/env python3
"""Plot noisy Mooney--Rivlin samples against clean truth and CSR predictions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from preprocess.synthetic_data import mooney_rivlin_to_ogden_terms, ogden_nominal_stress


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--prediction-csv', type=Path, required=True)
	parser.add_argument('--output-pdf', type=Path, required=True)
	parser.add_argument('--summary-json', type=Path, required=True)
	parser.add_argument('--legend-output-pdf', type=Path, default=None)
	parser.add_argument('--c10', type=float, default=0.18)
	parser.add_argument('--c01', type=float, default=0.02)
	args = parser.parse_args()

	data = np.genfromtxt(args.prediction_csv, delimiter=',', names=True, dtype=None, encoding='utf-8')
	mu_terms, alpha_terms = mooney_rivlin_to_ogden_terms(args.c10, args.c01)
	colors = {'UT': 'tab:blue', 'PS': 'tab:orange', 'ET': 'tab:green'}
	clean_targets = np.empty(data.shape[0], dtype=np.float64)

	figure, axis = plt.subplots(figsize=(8, 6))
	for mode in ('UT', 'PS', 'ET'):
		mask = data['mode'] == mode
		stretch = np.asarray(data['stretch'][mask], dtype=np.float64)
		noisy_stress = np.asarray(data['stress_target'][mask], dtype=np.float64)
		prediction = np.asarray(data['stress_prediction'][mask], dtype=np.float64)
		clean_stress = ogden_nominal_stress(
			stretch.astype(np.float32), mode, mu_terms, alpha_terms
		).astype(np.float64)
		clean_targets[mask] = clean_stress
		order = np.argsort(stretch)

		axis.scatter(
			stretch,
			noisy_stress,
			s=28,
			facecolors='none',
			edgecolors=colors[mode],
			alpha=0.75,
			label=f'{mode} data',
		)
		axis.plot(
			stretch[order], prediction[order], '-', color=colors[mode], linewidth=2.2,
			label=f'{mode} prediction',
		)

	rmse_clean = float(np.sqrt(np.mean((data['stress_prediction'] - clean_targets) ** 2)))
	ss_res = float(np.sum((data['stress_prediction'] - clean_targets) ** 2))
	ss_tot = float(np.sum((clean_targets - np.mean(clean_targets)) ** 2))
	r2_clean = 1.0 - ss_res / ss_tot

	axis.set_xlabel(r'$\lambda_1$', fontsize=25)
	axis.set_ylabel(r'$P_{11}$ (MPa)', fontsize=25)
	axis.tick_params(labelsize=20)
	axis.grid(alpha=0.25)
	figure.tight_layout()
	args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(args.output_pdf, bbox_inches='tight')
	plt.close(figure)

	if args.legend_output_pdf is not None:
		handles, labels = axis.get_legend_handles_labels()
		legend_figure = plt.figure(figsize=(8, 0.6))
		legend_figure.legend(
			handles,
			labels,
			loc='lower center',
			ncol=6,
			frameon=True,
			fontsize=9,
			bbox_to_anchor=(0.5, 0.0),
		)
		args.legend_output_pdf.parent.mkdir(parents=True, exist_ok=True)
		legend_figure.savefig(args.legend_output_pdf, bbox_inches='tight', pad_inches=0.02)
		plt.close(legend_figure)

	args.summary_json.parent.mkdir(parents=True, exist_ok=True)
	args.summary_json.write_text(
		json.dumps({'clean_truth_stress_rmse': rmse_clean, 'clean_truth_stress_r2': r2_clean}, indent=2)
		+ '\n'
	)


if __name__ == '__main__':
	main()
