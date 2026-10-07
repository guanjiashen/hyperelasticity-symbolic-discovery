from pathlib import Path
import time

import numpy as np
import torch

from preprocess.prepare_mode_dataset_invariant import prepare_mode_tensors_invariant
from preprocess.synthetic_data import (
	MOONEY_RIVLIN_C01,
	MOONEY_RIVLIN_C10,
	SYNTHETIC_NOISE_STD,
	generate_arruda_boyce_datasets,
	generate_mooney_rivlin_datasets,
	generate_ogden_datasets,
)
from preprocess.tools import split_mode_indices, to_tensor

from solver.network_invariant import FFBPNetworkInvariant, set_optimizer
from solver.stress_invariant import compute_norminal_stress_invariant
from solver.train_validate import train_validate_energy, train_validate_stress

from postprocess.loss_function import energy_loss, stress_loss
from postprocess.plots import stress_strain_plot
from postprocess.plots import energy_contour_plot
from postprocess.plots import loss_plot
from postprocess.save_weights import save_weights

from cli import (
	parse_args,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'output'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main() -> None:
	args = parse_args()
	nn_output_dir = OUTPUT_DIR / ('NN_output' if not args.output_suffix else f'NN_output-{args.output_suffix}')
	random_state = args.random_seed
	init_seed = args.init_seed if args.init_seed is not None else random_state
	torch.manual_seed(init_seed)
	np.random.seed(random_state)
	train_mode = args.train_mode
	use_synthetic_data = args.use_synthetic_data
	synthetic_model = args.synthetic_model
	synthetic_point_count = args.synthetic_point_count
	synthetic_noise_std = args.synthetic_noise_std
	synthetic_mode_ranges = {
		'UT': tuple(args.synthetic_ut_range),
		'PS': tuple(args.synthetic_ps_range),
		'ET': tuple(args.synthetic_et_range),
	}
	enable_biaxial = args.enable_biaxial
	biaxial_prefix = args.biaxial_prefix
	biaxial_base_dir = args.biaxial_base_dir
	dataset_candidates = (
		('UT', args.ut_dataset),
		('UC', args.uc_dataset),
		('PS', args.ps_dataset),
		('ET', args.et_dataset),
		('SS', args.ss_dataset),
		('SN', args.sn_dataset),
	)
	datasets = {
		mode_name: dataset_path
		for mode_name, dataset_path in dataset_candidates
		if dataset_path is not None and str(dataset_path).lower() != 'none'
	}
	if enable_biaxial:
		biaxial_candidates = sorted(
			[
				path for path in biaxial_base_dir.glob('B*')
				if path.is_dir() and (path / 'stress_stretch.txt').is_file()
			],
			key=lambda path: int(path.name[1:]) if path.name[1:].isdigit() else float('inf'),
		)
		if not biaxial_candidates:
			raise ValueError(
				f'No biaxial files found under {biaxial_base_dir!s}. '
				'Expected folders like B1, B2, ... each containing stress_stretch.txt.'
			)
		biaxial_datasets = {}
		for folder in biaxial_candidates:
			index_token = folder.name[1:]
			if not index_token.isdigit():
				continue
			mode_name = f'{biaxial_prefix}{int(index_token)}'
			biaxial_datasets[mode_name] = folder / 'stress_stretch.txt'
		if not biaxial_datasets:
			raise ValueError(
				f'No numbered biaxial folders found under {biaxial_base_dir!s}. '
				'Expected folder names like B1, B2, ... with stress_stretch.txt.'
			)
		datasets.update(biaxial_datasets)

	print(f'Using device: {DEVICE}')

	if use_synthetic_data:
		if synthetic_model == 'ogden':
			datasets = generate_ogden_datasets(
				base_dir=BASE_DIR,
				mode_ranges=synthetic_mode_ranges,
				point_count=synthetic_point_count,
				mu_terms=args.ogden_mu,
				alpha_terms=args.ogden_alpha,
				noise_std=synthetic_noise_std,
				random_state=random_state,
			)
			print(
				'Using synthetic Ogden datasets for UT, PS, ET '
				f'(mu={tuple(args.ogden_mu)}, alpha={tuple(args.ogden_alpha)}, noise_std={synthetic_noise_std}).'
			)
		elif synthetic_model == 'arruda_boyce':
			datasets = generate_arruda_boyce_datasets(
				base_dir=BASE_DIR,
				mode_ranges=synthetic_mode_ranges,
				point_count=synthetic_point_count,
				mu=args.arruda_boyce_mu,
				lambda_m=args.arruda_boyce_lambda_m,
				noise_std=synthetic_noise_std,
				random_state=random_state,
			)
			print(
				'Using synthetic Arruda-Boyce datasets for UT, PS, ET '
				f'(mu={args.arruda_boyce_mu}, lambda_m={args.arruda_boyce_lambda_m}, noise_std={synthetic_noise_std}).'
			)
		elif synthetic_model == 'mooney_rivlin':
			datasets = generate_mooney_rivlin_datasets(
				base_dir=BASE_DIR,
				mode_ranges=synthetic_mode_ranges,
				point_count=synthetic_point_count,
				c10=args.mooney_rivlin_c10,
				c01=args.mooney_rivlin_c01,
				noise_std=synthetic_noise_std,
				random_state=random_state,
			)
			print(
				'Using synthetic Mooney-Rivlin datasets for UT, PS, ET '
					f'(C10={args.mooney_rivlin_c10}, C01={args.mooney_rivlin_c01}, '
					f'noise_std={synthetic_noise_std}).'
			)
		else:
			raise ValueError(
				f'Unsupported synthetic model {synthetic_model!r}. '
				'Expected "ogden", "arruda_boyce", or "mooney_rivlin".'
			)

	prepare_mode_tensors = prepare_mode_tensors_invariant
	model = FFBPNetworkInvariant(
		hidden_neurons=args.hidden_neurons,
		hidden_layers=args.hidden_layers,
		activation=args.activation,
	).to(DEVICE)
	compute_stress = compute_norminal_stress_invariant
	model_input_key = 'invariants'
	print('Model input type: invariants')
	print(f'Train mode: {train_mode}')
	if enable_biaxial:
		biaxial_modes = [mode_name for mode_name in datasets if mode_name.startswith(biaxial_prefix)]
		print(f'Biaxial modes enabled ({len(biaxial_modes)}): {", ".join(biaxial_modes)} from {biaxial_base_dir}')

    # preprocess-data
	# mode_data stores the per-mode NumPy preprocessing results before any train/val/test split.
	# mode_tensors stores the split-specific PyTorch tensors used directly by training, evaluation, and plotting.
	mode_data, mode_tensors = prepare_mode_tensors(
		datasets,
		lambda array: to_tensor(array).to(DEVICE),
		lambda sample_count: split_mode_indices(sample_count, random_state),
	)
	total_samples = sum(len(values['stretch']) for values in mode_data.values())
	print(f'Total samples: {total_samples}')

    # solver-model
	print(f'Total parameters: {sum(parameter.numel() for parameter in model.parameters())}')

	optimizer = set_optimizer(
		model,
		optimizer_name=args.optimizer,
		learning_rate=args.learning_rate,
		lbfgs_max_iter=args.lbfgs_max_iter,
		lbfgs_history_size=args.lbfgs_history_size,
	)

	# solver-train_validate
	train_start_time = time.perf_counter()

	if train_mode == 'energy':
		train_losses, val_losses, best_val_loss = train_validate_energy(
			model,
			optimizer,
			mode_tensors,
			DEVICE,
			model_input_key=model_input_key,
			epochs=args.epochs,
			print_every=args.print_every,
		)
	elif train_mode == 'stress':
		train_losses, val_losses, best_val_loss = train_validate_stress(
			model,
			optimizer,
			mode_tensors,
			DEVICE,
			compute_stress,
			epochs=args.epochs,
			print_every=args.print_every,
		)
	else:
		raise ValueError(f'Unsupported train mode {train_mode!r}. Expected "energy" or "stress".')


	train_elapsed_time = time.perf_counter() - train_start_time


    # postprocess-loss
	test_energy_loss = energy_loss(model, mode_tensors, 'test', model_input_key=model_input_key)
	test_stress_loss = stress_loss(model, mode_tensors, 'test', compute_stress)
	
	print(f'Best validation energy loss: {best_val_loss:.6f}')
	print(f'Test energy loss: {test_energy_loss:.6f}')
	print(f'Test stress loss: {test_stress_loss:.6f}')
	print(f'Training time: {train_elapsed_time:.2f} s')

    # postprpocess-plot
	stress_strain_plot(
		base_dir=nn_output_dir,
		datasets=datasets,
		mode_tensors=mode_tensors,
		model=model,
		compute_stress=compute_stress,
		show_legend=True,
	)

	energy_contour_plot(
		base_dir=nn_output_dir,
		model=model,
	)

    # postprpocess-plot
	loss_plot(
		base_dir=nn_output_dir,
		train_losses=train_losses,
		val_losses=val_losses,
	)

    # postprocess-weights
	save_weights(
		model,
		nn_output_dir,
	)

if __name__ == '__main__':
	main()
