from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
HAS_LATEX = shutil.which('latex') is not None or shutil.which('pdflatex') is not None
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['text.usetex'] = HAS_LATEX
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as torch_f


OUTPUT_PATH = Path(__file__).resolve().parent / 'torch_activation_and_derivative.png'
X_MIN = -4.0
X_MAX = 4.0
NUM_POINTS = 1001


def get_activation_functions():
	return {
		'sigmoid': torch.sigmoid,
		'relu': torch.relu,
		'tanh': torch.tanh,
		'gelu': torch_f.gelu,
		'selu': torch_f.selu,
		'leaky_relu': torch_f.leaky_relu,
		'elu': torch_f.elu,
		'softplus': torch_f.softplus,
	}


def evaluate_activation_and_derivative(activation, x_values: torch.Tensor):
	x = x_values.clone().detach().requires_grad_(True)
	y = activation(x)
	derivative = torch.autograd.grad(
		outputs=y,
		inputs=x,
		grad_outputs=torch.ones_like(y),
		create_graph=False,
		retain_graph=False,
	)[0]
	return y.detach(), derivative.detach()


def main() -> None:
	activation_functions = get_activation_functions()
	x_values = torch.linspace(X_MIN, X_MAX, NUM_POINTS)
	line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2))]

	figure, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
	activation_axis, derivative_axis = axes

	for index, (name, activation) in enumerate(activation_functions.items()):
		y_values, derivative_values = evaluate_activation_and_derivative(activation, x_values)
		line_style = line_styles[index % len(line_styles)]
		activation_axis.plot(x_values.numpy(), y_values.numpy(), label=name, linewidth=3.0, linestyle=line_style)
		derivative_axis.plot(x_values.numpy(), derivative_values.numpy(), label=name, linewidth=3.0, linestyle=line_style)

	activation_axis.set_ylabel(r'$f(x)$', fontsize=25)
	activation_axis.set_xlim([-4, 4])
	activation_axis.set_ylim([-2, 2])
	activation_axis.tick_params(axis='both', labelsize=25)
	activation_axis.grid(alpha=0.25)

	derivative_axis.set_xlabel(r'$x$', fontsize=25)
	derivative_axis.set_ylabel(r"$f'(x)$", fontsize=25)
	derivative_axis.set_xlim([-4, 4])
	derivative_axis.set_ylim([-2, 2])
	derivative_axis.tick_params(axis='both', labelsize=25)
	derivative_axis.grid(alpha=0.25)
	derivative_axis.legend(
		loc='upper center',
		bbox_to_anchor=(0.5, -0.22),
		ncol=4,
		fontsize=20,
		frameon=False,
	)
	# derivative_axis.set_title(r'$\mathrm{First\ Derivatives}$')

	figure.tight_layout()
	figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
	plt.close(figure)
	print(f'Saved activation plot to {OUTPUT_PATH}')


if __name__ == '__main__':
	main()
