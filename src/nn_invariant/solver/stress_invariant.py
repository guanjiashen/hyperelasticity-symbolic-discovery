import torch
import torch.nn as nn


BIAXIAL_STRETCH_RATIOS = {
	'BS1': 1.04,
	'BS2': 1.06,
	'BS3': 1.08,
	'BS4': 1.10,
	'BS5': 1.12,
	'BS6': 1.14,
	'BS7': 1.16,
	'BS8': 1.20,
	'BS9': 1.24,
	'BL1': 1.3,
	'BL2': 1.6,
	'BL3': 1.9,
	'BL4': 2.2,
	'BL5': 2.5,
	'BL6': 2.8,
	'BL7': 3.1,
	'BL8': 3.4,
	'BL9': 3.7,
}


SIMPLE_SHEAR_MODES = ('SS', 'SN')


def compute_norminal_stress_invariant(model: nn.Module, stretch: torch.Tensor, mode_name: str) -> torch.Tensor:
	stretch = stretch.reshape(-1, 1)

	if mode_name in ('UT', 'UC'):
		lambda_1 = stretch
		lambda_2 = stretch.pow(-0.5)
		lambda_3 = stretch.pow(-0.5)
	elif mode_name == 'PS':
		lambda_1 = stretch
		lambda_2 = torch.ones_like(stretch)
		lambda_3 = stretch.pow(-1.0)
	elif mode_name == 'ET':
		lambda_1 = stretch
		lambda_2 = stretch
		lambda_3 = stretch.pow(-2.0)
	elif mode_name in SIMPLE_SHEAR_MODES:
		gamma = stretch
		root = torch.sqrt(4.0 + gamma.pow(2))
		lambda_1 = torch.sqrt(1.0 + 0.5 * gamma.pow(2) + 0.5 * gamma * root)
		lambda_2 = torch.sqrt(1.0 + 0.5 * gamma.pow(2) - 0.5 * gamma * root)
		lambda_3 = torch.ones_like(stretch)
	elif mode_name in BIAXIAL_STRETCH_RATIOS:
		lambda_1 = stretch
		lambda_2 = torch.full_like(stretch, BIAXIAL_STRETCH_RATIOS[mode_name])
		lambda_3 = 1.0 / (lambda_1 * lambda_2)
	else:
		raise ValueError(f'Unsupported mode: {mode_name}')

	i1 = lambda_1.pow(2) + lambda_2.pow(2) + lambda_3.pow(2)
	i2 = (
		lambda_1.pow(2) * lambda_2.pow(2)
		+ lambda_2.pow(2) * lambda_3.pow(2)
		+ lambda_3.pow(2) * lambda_1.pow(2)
	)

	invariants = torch.cat([i1, i2], dim=1)
	invariants.requires_grad_(True)
	energy = model(invariants)
	gradients = torch.autograd.grad(
		outputs=energy,
		inputs=invariants,
		grad_outputs=torch.ones_like(energy),
		create_graph=True,
		retain_graph=True,
	)[0]

	dwd_i1 = gradients[:, 0:1]
	dwd_i2 = gradients[:, 1:2]

	if mode_name in ('UT', 'UC'):
		return 2.0 * (stretch - stretch.pow(-2.0)) * (dwd_i1 + dwd_i2 / stretch)
	if mode_name in SIMPLE_SHEAR_MODES:
		return 2.0 * stretch * (dwd_i1 + dwd_i2)
	if mode_name == 'PS':
		return 2.0 * (stretch - stretch.pow(-3.0)) * (dwd_i1 + dwd_i2)
	if mode_name in BIAXIAL_STRETCH_RATIOS:
		return 2.0 * (lambda_1 - lambda_3.pow(2.0) / lambda_1) * (dwd_i1 + dwd_i2 * lambda_2.pow(2.0))
	return 2.0 * (stretch - stretch.pow(-5.0)) * (dwd_i1 + dwd_i2 * stretch.pow(2))