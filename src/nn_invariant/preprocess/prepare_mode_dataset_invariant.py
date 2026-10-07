from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

if TYPE_CHECKING:
	import torch


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


def load_mode_data(mode_name: str, file_path: Path) -> dict[str, np.ndarray]:
	data = np.loadtxt(file_path, dtype=np.float32)
	stress = data[:, 0]
	stretch = data[:, 1]

	reference_stretch = 0.0 if mode_name in SIMPLE_SHEAR_MODES else 1.0
	stretch = np.insert(stretch, 0, reference_stretch)
	stress = np.insert(stress, 0, 0.0)

	return {
		'mode': np.array([mode_name] * len(stretch)),
		'stretch': stretch,
		'stress': stress,
	}


def get_mode_stretches(stretch: np.ndarray, mode_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if mode_name in ('UT', 'UC'):
		lambda_1 = stretch
		lambda_2 = stretch ** -0.5
		lambda_3 = stretch ** -0.5
	elif mode_name == 'PS':
		lambda_1 = stretch
		lambda_2 = np.ones_like(stretch)
		lambda_3 = stretch ** -1.0
	elif mode_name == 'ET':
		lambda_1 = stretch
		lambda_2 = stretch
		lambda_3 = stretch ** -2.0
	elif mode_name in SIMPLE_SHEAR_MODES:
		gamma = stretch
		root = np.sqrt(4.0 + gamma ** 2)
		lambda_1 = np.sqrt(1.0 + 0.5 * gamma ** 2 + 0.5 * gamma * root)
		lambda_2 = np.sqrt(1.0 + 0.5 * gamma ** 2 - 0.5 * gamma * root)
		lambda_3 = np.ones_like(stretch)
	elif mode_name in BIAXIAL_STRETCH_RATIOS:
		lambda_1 = stretch
		lambda_2 = np.full_like(stretch, BIAXIAL_STRETCH_RATIOS[mode_name])
		lambda_3 = 1.0 / (lambda_1 * lambda_2)
	else:
		raise ValueError(f'Unsupported mode: {mode_name}')

	return lambda_1, lambda_2, lambda_3


def build_invariants_invariant(stretch: np.ndarray, mode_name: str) -> np.ndarray:
	lambda_1, lambda_2, lambda_3 = get_mode_stretches(stretch, mode_name)
	i1 = lambda_1 ** 2 + lambda_2 ** 2 + lambda_3 ** 2
	i2 = (
		lambda_1 ** 2 * lambda_2 ** 2
		+ lambda_2 ** 2 * lambda_3 ** 2
		+ lambda_3 ** 2 * lambda_1 ** 2
	)
	return np.stack([i1, i2], axis=1).astype(np.float32)


def integrate_energy(stretch: np.ndarray, stress: np.ndarray, mode_name: str) -> np.ndarray:
	strain = stretch if mode_name in SIMPLE_SHEAR_MODES else stretch - 1.0
	energy = np.zeros_like(stretch, dtype=np.float32)
	multiplier = 2.0 if mode_name == 'ET' else 1.0

	for index in range(1, len(stretch)):
		delta_strain = strain[index] - strain[index - 1]
		avg_stress = 0.5 * (stress[index] + stress[index - 1])
		energy[index] = energy[index - 1] + multiplier * avg_stress * delta_strain

	return energy.reshape(-1, 1).astype(np.float32)


def prepare_mode_dataset_invariant(mode_name: str, file_path: Path) -> dict[str, np.ndarray]:
	mode_data = load_mode_data(mode_name, file_path)
	invariants = build_invariants_invariant(mode_data['stretch'], mode_name)
	energy = integrate_energy(mode_data['stretch'], mode_data['stress'], mode_name)

	return {
		'mode': mode_data['mode'],
		'stretch': mode_data['stretch'].astype(np.float32),
		'stress': mode_data['stress'].astype(np.float32),
		'invariants': invariants,
		'energy': energy,
	}


def prepare_mode_tensors_invariant(
	datasets: Mapping[str, Path],
	to_tensor: Callable[[np.ndarray], torch.Tensor],
	split_mode_indices: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, torch.Tensor]]]:
	mode_data = {
		mode_name: prepare_mode_dataset_invariant(mode_name, file_path)
		for mode_name, file_path in datasets.items()
	}

	mode_tensors: dict[str, dict[str, torch.Tensor]] = {}
	for mode_name, values in mode_data.items():
		train_idx, val_idx, test_idx = split_mode_indices(len(values['stretch']))
		mode_tensors[mode_name] = {
			'train_invariants': to_tensor(values['invariants'][train_idx]),
			'train_energy': to_tensor(values['energy'][train_idx]),
			'train_stretch': to_tensor(values['stretch'][train_idx]).reshape(-1, 1).requires_grad_(True),
			'train_stress': to_tensor(values['stress'][train_idx]).reshape(-1, 1),
			'val_invariants': to_tensor(values['invariants'][val_idx]),
			'val_energy': to_tensor(values['energy'][val_idx]),
			'val_stretch': to_tensor(values['stretch'][val_idx]).reshape(-1, 1).requires_grad_(True),
			'val_stress': to_tensor(values['stress'][val_idx]).reshape(-1, 1),
			'test_invariants': to_tensor(values['invariants'][test_idx]),
			'test_energy': to_tensor(values['energy'][test_idx]),
			'test_stretch': to_tensor(values['stretch'][test_idx]).reshape(-1, 1).requires_grad_(True),
			'test_stress': to_tensor(values['stress'][test_idx]).reshape(-1, 1),
			'all_invariants': to_tensor(values['invariants']),
			'all_stretch': to_tensor(values['stretch']).reshape(-1, 1).requires_grad_(True),
			'all_stress': to_tensor(values['stress']).reshape(-1, 1),
			'all_energy': to_tensor(values['energy']),
		}

	return mode_data, mode_tensors