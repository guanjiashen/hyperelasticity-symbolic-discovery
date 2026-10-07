import torch
import torch.nn.functional as torch_f
import torch.nn as nn
import torch.optim as optim
from collections.abc import Callable, Sequence


HIDDEN_NEURONS = 3

# lbfgs
# adam
OPTIMIZER_NAME = 'lbfgs'
LEARNING_RATE = 0.2

ACTIVATION = 'softplus'
# ACTIVATION = CustomActivation(slope=0.1)


# Activation can be configured by name, a PyTorch module, or a callable tensor-to-tensor function.
ActivationLike = str | nn.Module | Callable[[torch.Tensor], torch.Tensor]


class CustomActivation(nn.Module):
	def __init__(self, slope: float = 0.1) -> None:
		super().__init__()
		self.slope = slope

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.tanh(x) + self.slope * x


def resolve_activation(activation: ActivationLike) -> Callable[[torch.Tensor], torch.Tensor]:
	if isinstance(activation, nn.Module):
		return activation

	if callable(activation) and not isinstance(activation, str):
		return activation

	if isinstance(activation, str):
		activation_map: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
			'sigmoid': torch.sigmoid,
			'relu': torch.relu,
			'tanh': torch.tanh,
			'gelu': torch_f.gelu,
			'selu': torch_f.selu,
			'leaky_relu': torch_f.leaky_relu,
			'elu': torch_f.elu,
			'softplus': torch_f.softplus,
		}
		try:
			return activation_map[activation]
		except KeyError as error:
			available = ', '.join(sorted(activation_map))
			raise ValueError(f'Unsupported activation {activation!r}. Available: {available}') from error

	raise TypeError('activation must be a string, nn.Module, or callable')


class FFBPNetworkInvariant(nn.Module):
	def __init__(
		self,
		hidden_neurons: int = HIDDEN_NEURONS,
		hidden_layers: Sequence[int] | None = None,
		activation: ActivationLike = ACTIVATION,
	) -> None:
		super().__init__()
		layer_widths = tuple(hidden_layers) if hidden_layers is not None else (hidden_neurons,)
		if not layer_widths or any(width < 1 for width in layer_widths):
			raise ValueError('hidden_layers must contain one or more positive integers')
		self.hidden_layer_sizes = layer_widths
		self.hidden = nn.Linear(2, layer_widths[0])
		self.additional_hidden = nn.ModuleList(
			nn.Linear(input_width, output_width)
			for input_width, output_width in zip(layer_widths, layer_widths[1:])
		)
		self.output = nn.Linear(layer_widths[-1], 1)
		self.activation_name = activation if isinstance(activation, str) else type(activation).__name__
		self.activation = resolve_activation(activation)

	def forward(self, invariants: torch.Tensor) -> torch.Tensor:
		hidden_values = self.activation(self.hidden(invariants))
		for layer in self.additional_hidden:
			hidden_values = self.activation(layer(hidden_values))
		return self.output(hidden_values)


ShenFFBPNetworkInvariant = FFBPNetworkInvariant


MAX_ITER = 30
HISTORY_SIZE = 50
LINE_SEARCH_FN = 'strong_wolfe'


def set_optimizer(
	model: nn.Module,
	optimizer_name: str = OPTIMIZER_NAME,
	learning_rate: float = LEARNING_RATE,
	lbfgs_max_iter: int = MAX_ITER,
	lbfgs_history_size: int = HISTORY_SIZE,
) -> optim.Optimizer:
	if optimizer_name == 'adam':
		return optim.Adam(model.parameters(), lr=learning_rate)

	if optimizer_name == 'lbfgs':
		return optim.LBFGS(
			model.parameters(),
			lr=learning_rate,
			max_iter=lbfgs_max_iter,
			history_size=lbfgs_history_size,
			line_search_fn=LINE_SEARCH_FN,
		)

	raise ValueError(f'Unsupported optimizer {optimizer_name!r}. Expected "adam" or "lbfgs".')
