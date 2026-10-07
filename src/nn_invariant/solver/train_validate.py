from collections.abc import Callable, Mapping

import torch
import torch.nn as nn
import torch.optim as optim

from postprocess.loss_function import energy_loss, stress_loss


DEFAULT_EPOCHS = 200
DEFAULT_PRINT_EVERY = 20


def train_validate_energy(
	model: nn.Module,
	optimizer: optim.Optimizer,
	mode_tensors: Mapping[str, Mapping[str, torch.Tensor]],
	device: torch.device,
	model_input_key: str = 'invariants',
	epochs: int = DEFAULT_EPOCHS,
	print_every: int = DEFAULT_PRINT_EVERY,
) -> tuple[list[float], list[float], float]:
	train_losses: list[float] = []
	val_losses: list[float] = []
	best_val_loss = float('inf')
	best_state = None

	for epoch in range(epochs):
		model.train()

		def closure() -> torch.Tensor:
			optimizer.zero_grad()
			weighted_loss = 0.0
			weighted_count = 0
			for tensors in mode_tensors.values():
				prediction = model(tensors[f'train_{model_input_key}'])
				target = tensors['train_energy']
				mode_loss = torch.mean((prediction - target) ** 2)
				weighted_loss = weighted_loss + mode_loss * target.shape[0]
				weighted_count += target.shape[0]
			total_loss = weighted_loss / weighted_count
			total_loss.backward()
			return total_loss

		train_loss = optimizer.step(closure)
		train_loss_value = float(train_loss.item())
		val_loss_value = energy_loss(model, mode_tensors, 'val', model_input_key=model_input_key)
		train_losses.append(train_loss_value)
		val_losses.append(val_loss_value)

		if val_loss_value < best_val_loss:
			best_val_loss = val_loss_value
			best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

		if (epoch + 1) % print_every == 0:
			print(
				f'Epoch [{epoch + 1}/{epochs}] '
				f'train_energy_loss={train_loss_value:.6f} '
				f'val_energy_loss={val_loss_value:.6f}'
			)

	if best_state is not None:
		model.load_state_dict(best_state)
		model.to(device)

	return train_losses, val_losses, best_val_loss


def train_validate_stress(
	model: nn.Module,
	optimizer: optim.Optimizer,
	mode_tensors: Mapping[str, Mapping[str, torch.Tensor]],
	device: torch.device,
	compute_stress: Callable[[nn.Module, torch.Tensor, str], torch.Tensor],
	epochs: int = DEFAULT_EPOCHS,
	print_every: int = DEFAULT_PRINT_EVERY,
) -> tuple[list[float], list[float], float]:
	train_losses: list[float] = []
	val_losses: list[float] = []
	best_val_loss = float('inf')
	best_state = None

	for epoch in range(epochs):
		model.train()

		def closure() -> torch.Tensor:
			optimizer.zero_grad()
			weighted_loss = 0.0
			weighted_count = 0
			for mode_name, tensors in mode_tensors.items():
				stretch = tensors['train_stretch'].clone().detach().requires_grad_(True)
				prediction = compute_stress(model, stretch, mode_name)
				target = tensors['train_stress']
				mode_loss = torch.mean((prediction - target) ** 2)
				weighted_loss = weighted_loss + mode_loss * target.shape[0]
				weighted_count += target.shape[0]
			total_loss = weighted_loss / weighted_count
			total_loss.backward()
			return total_loss

		train_loss = optimizer.step(closure)
		train_loss_value = float(train_loss.item())
		val_loss_value = stress_loss(model, mode_tensors, 'val', compute_stress)
		train_losses.append(train_loss_value)
		val_losses.append(val_loss_value)

		if val_loss_value < best_val_loss:
			best_val_loss = val_loss_value
			best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

		if (epoch + 1) % print_every == 0:
			print(
				f'Epoch [{epoch + 1}/{epochs}] '
				f'train_stress_loss={train_loss_value:.6f} '
				f'val_stress_loss={val_loss_value:.6f}'
			)

	if best_state is not None:
		model.load_state_dict(best_state)
		model.to(device)

	return train_losses, val_losses, best_val_loss