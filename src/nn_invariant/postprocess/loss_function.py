
from collections.abc import Callable, Mapping

import torch
import torch.nn as nn


def energy_loss(
	model: nn.Module,
	split_data: dict[str, dict[str, torch.Tensor]],
	key: str,
	model_input_key: str = 'invariants',
) -> float:
	total_loss = 0.0
	total_count = 0

	with torch.no_grad():
		for tensors in split_data.values():
			prediction = model(tensors[f'{key}_{model_input_key}'])
			target = tensors[f'{key}_energy']
			loss = torch.mean((prediction - target) ** 2)
			total_loss += loss.item() * target.shape[0]
			total_count += target.shape[0]

	return total_loss / max(total_count, 1)


def stress_loss(
	model: nn.Module,
	split_data: Mapping[str, Mapping[str, torch.Tensor]],
	key: str,
	compute_stress: Callable[[nn.Module, torch.Tensor, str], torch.Tensor],
) -> float:
	total_loss = 0.0
	total_count = 0

	for mode_name, tensors in split_data.items():
		stretch = tensors[f'{key}_stretch'].clone().detach().requires_grad_(True)
		prediction = compute_stress(model, stretch, mode_name)
		target = tensors[f'{key}_stress']
		loss = torch.mean((prediction - target) ** 2)
		total_loss += loss.item() * target.shape[0]
		total_count += target.shape[0]

	return total_loss / max(total_count, 1)