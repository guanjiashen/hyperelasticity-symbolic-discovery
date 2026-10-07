from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def loss_plot(
	base_dir: Path,
	train_losses: Sequence[float] | None = None,
	val_losses: Sequence[float] | None = None,
	file_name: str = 'ffbp_loss_curve.pdf',
) -> None:
	base_dir.mkdir(parents=True, exist_ok=True)
	if train_losses is not None and val_losses is not None:
		plt.figure(figsize=(8, 6))
		plt.semilogy(train_losses, label='train loss')
		plt.semilogy(val_losses, label='validation loss')
		plt.xlabel('Epoch')
		plt.ylabel(
			r'Stress loss',
			fontsize=14,
		)
		plt.grid(True, which='both', alpha=0.25)
		plt.legend(fontsize=16)
		plt.tight_layout()
		plt.savefig(base_dir / file_name, dpi=200)
		plt.close()

def stress_strain_plot(
	base_dir: Path,
	datasets: Mapping[str, Path],
	mode_tensors: Mapping[str, Mapping[str, torch.Tensor]],
	model: nn.Module,
	compute_stress: Callable[[nn.Module, torch.Tensor, str], torch.Tensor],
	show_legend: bool = True,
	file_name: str = 'ffbp_predictions.pdf',
) -> None:
	base_dir.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(8, 6))
	color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
	colors = {
		mode_name: color_cycle[index % len(color_cycle)]
		for index, mode_name in enumerate(datasets.keys())
	}
	for mode_name in datasets.keys():
		tensors = mode_tensors[mode_name]
		prediction = compute_stress(model, tensors['all_stretch'], mode_name)
		stretch_np = tensors['all_stretch'].detach().cpu().numpy().flatten()
		prediction_np = prediction.detach().cpu().numpy().flatten()
		is_biaxial_mode = mode_name.startswith('B')
		all_stress_np = tensors['all_stress'].detach().cpu().numpy().flatten()
		reference_mask = np.isclose(stretch_np, 1.0) & np.isclose(all_stress_np, 0.0)
		plot_mask = ~reference_mask if is_biaxial_mode else np.ones_like(reference_mask, dtype=bool)
		stretch_np = stretch_np[plot_mask]
		prediction_np = prediction_np[plot_mask]
		sort_idx = np.argsort(stretch_np)

		for split_name, marker, marker_size in (
			('train', 'o', 52),
			('test', 'x', 70),
		):
			split_stretch = tensors[f'{split_name}_stretch'].detach().cpu().numpy().flatten()
			split_stress = tensors[f'{split_name}_stress'].detach().cpu().numpy().flatten()
			split_reference_mask = np.isclose(split_stretch, 1.0) & np.isclose(split_stress, 0.0)
			split_plot_mask = (
				~split_reference_mask
				if is_biaxial_mode else np.ones_like(split_reference_mask, dtype=bool)
			)
			plt.scatter(
				split_stretch[split_plot_mask],
				split_stress[split_plot_mask],
				label=f'{mode_name} {split_name}',
				s=marker_size,
				marker=marker,
				color=colors[mode_name],
				alpha=0.8,
			)
		plt.plot(
			stretch_np[sort_idx],
			prediction_np[sort_idx],
			label=f'{mode_name} prediction',
			linewidth=2.0,
			color=colors[mode_name],
		)

	plt.xlabel(r'$\lambda_1$', fontsize=16)
	plt.ylabel(r'$P_{11}$(MPa)', fontsize=16)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	# plt.title(', '.join(datasets.keys()) + ' combined prediction')
	plt.grid(alpha=0.25)
	if show_legend:
		plt.legend(loc='upper left', fontsize=13, ncol=2)
	plt.tight_layout()
	plt.savefig(base_dir / file_name, dpi=200)
	plt.close()


def energy_contour_plot(
	base_dir: Path,
	model: nn.Module,
	lambda_1_range: tuple[float, float] = (0.5, 3.0),
	lambda_2_range: tuple[float, float] = (0.5, 3.0),
	grid_points: int = 120,
	contour_levels: int = 10,
	file_name: str = 'ffbp_energy_contour.png',
) -> None:
	base_dir.mkdir(parents=True, exist_ok=True)
	if grid_points < 2:
		raise ValueError('grid_points must be at least 2')

	lambda_1_values = np.linspace(lambda_1_range[0], lambda_1_range[1], grid_points, dtype=np.float32)
	lambda_2_values = np.linspace(lambda_2_range[0], lambda_2_range[1], grid_points, dtype=np.float32)
	mesh_lambda_1, mesh_lambda_2 = np.meshgrid(lambda_1_values, lambda_2_values)

	flat_lambda_1 = mesh_lambda_1.reshape(-1, 1)
	flat_lambda_2 = mesh_lambda_2.reshape(-1, 1)
	lambda_3 = 1.0 / (flat_lambda_1 * flat_lambda_2)
	i1 = flat_lambda_1 ** 2 + flat_lambda_2 ** 2 + lambda_3 ** 2
	i2 = (
		flat_lambda_1 ** 2 * flat_lambda_2 ** 2
		+ flat_lambda_2 ** 2 * lambda_3 ** 2
		+ lambda_3 ** 2 * flat_lambda_1 ** 2
	)
	model_inputs_np = np.concatenate([i1, i2], axis=1).astype(np.float32)

	device = next(model.parameters()).device
	model_inputs = torch.from_numpy(model_inputs_np).to(device)

	with torch.no_grad():
		energy = model(model_inputs)
	energy_np = energy.detach().cpu().numpy().reshape(mesh_lambda_1.shape)

	plt.figure(figsize=(8, 6))
	contour = plt.contourf(mesh_lambda_1, mesh_lambda_2, energy_np, levels=contour_levels, cmap='viridis')
	colorbar = plt.colorbar(contour, label='Strain energy density')
	colorbar.ax.tick_params(labelsize=12)
	colorbar.set_label('Strain energy density', fontsize=14)
	line_contour = plt.contour(mesh_lambda_1, mesh_lambda_2, energy_np, levels=contour_levels, colors='white', linewidths=0.35)
	plt.clabel(line_contour, inline=True, fontsize=10, fmt='%.3f')
	plt.xlabel(r'$\lambda_1$', fontsize=16)
	plt.ylabel(r'$\lambda_2$', fontsize=16)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	# plt.title('Energy contour in lambda_1-lambda_2 space')
	plt.tight_layout()
	plt.savefig(base_dir / file_name, dpi=220)
	plt.close()
