from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np

from .config import add_local_import_paths


def build_lambda_grid(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	lambda1_values = np.linspace(args.lambda1_min, args.lambda1_max, args.lambda1_points, dtype=np.float64)
	lambda2_values = np.linspace(args.lambda2_min, args.lambda2_max, args.lambda2_points, dtype=np.float64)
	lambda1_grid, lambda2_grid = np.meshgrid(lambda1_values, lambda2_values, indexing='ij')
	lambda1 = lambda1_grid.reshape(-1)
	lambda2 = lambda2_grid.reshape(-1)
	lambda3 = 1.0 / (lambda1 * lambda2)
	return lambda1, lambda2, lambda3


def build_loadcase_samples(args) -> dict[str, np.ndarray]:
	mode_names: list[str] = []
	stretches: list[np.ndarray] = []
	for mode_name, stretch_range, point_count in (
		('UT', args.ut_stretch_range, args.ut_points),
		('UC', getattr(args, 'uc_stretch_range', (0.9, 1.0)), getattr(args, 'uc_points', 0)),
		('PS', args.ps_stretch_range, args.ps_points),
		('ET', args.et_stretch_range, args.et_points),
		('SS', getattr(args, 'ss_shear_range', (0.0, 0.2)), getattr(args, 'ss_points', 0)),
	):
		stretch_min, stretch_max = (float(stretch_range[0]), float(stretch_range[1]))
		if point_count == 0:
			continue
		if point_count < 2:
			raise ValueError(f'{mode_name} point count must be at least 2.')
		if stretch_min <= 0.0 and mode_name != 'SS':
			raise ValueError(f'{mode_name} stretch minimum must be positive.')
		if stretch_max <= stretch_min:
			raise ValueError(f'{mode_name} stretch maximum must be larger than the minimum.')
		mode_stretch = np.linspace(stretch_min, stretch_max, point_count, dtype=np.float64)
		mode_names.extend([mode_name] * point_count)
		stretches.append(mode_stretch)

	return {
		'mode_name': np.asarray(mode_names, dtype='U8'),
		'stretch': np.concatenate(stretches).astype(np.float64),
	}


def compute_invariants(lambda1: np.ndarray, lambda2: np.ndarray, lambda3: np.ndarray) -> np.ndarray:
	i1 = lambda1 ** 2 + lambda2 ** 2 + lambda3 ** 2
	i2 = (
		lambda1 ** 2 * lambda2 ** 2
		+ lambda2 ** 2 * lambda3 ** 2
		+ lambda3 ** 2 * lambda1 ** 2
	)
	return np.column_stack([i1, i2]).astype(np.float32)


def _load_mode_stretches_helper():
	add_local_import_paths()
	try:
		return importlib.import_module('preprocess.prepare_mode_dataset_invariant').get_mode_stretches
	except ImportError as error:
		raise ImportError(
			'Failed to import mode-stretch helpers from preprocess.prepare_mode_dataset_invariant.'
		) from error


def load_nn_model(model_path: Path):
	add_local_import_paths()
	try:
		torch = importlib.import_module('torch')
		FFBPNetworkInvariant = importlib.import_module('solver.network_invariant').FFBPNetworkInvariant
	except ImportError as error:
		raise ImportError(
			'Failed to import torch or the invariant NN model. '
			'Run the export step in an environment that contains the FFNN dependencies.'
		) from error

	state_dict = torch.load(model_path, map_location='cpu')
	hidden_neurons = int(state_dict['hidden.weight'].shape[0])
	hidden_layers = [hidden_neurons]
	index = 0
	while f'additional_hidden.{index}.weight' in state_dict:
		hidden_layers.append(int(state_dict[f'additional_hidden.{index}.weight'].shape[0]))
		index += 1
	activation = 'softplus'
	metadata_path = model_path.with_suffix('.json')
	if metadata_path.is_file():
		metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
		activation = str(metadata.get('activation', activation))
		hidden_layers = [int(width) for width in metadata.get('hidden_layers', hidden_layers)]
	model = FFBPNetworkInvariant(hidden_layers=hidden_layers, activation=activation)
	model.load_state_dict(state_dict)
	model.eval()
	return torch, model


def _evaluate_model_invariant_response(torch, model, invariants: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
	energy_batches: list[np.ndarray] = []
	dwd_i1_batches: list[np.ndarray] = []
	dwd_i2_batches: list[np.ndarray] = []

	for start in range(0, len(invariants), batch_size):
		batch = torch.tensor(invariants[start:start + batch_size], dtype=torch.float32, requires_grad=True)
		predictions = model(batch)
		gradients = torch.autograd.grad(
			outputs=predictions,
			inputs=batch,
			grad_outputs=torch.ones_like(predictions),
			create_graph=False,
			retain_graph=False,
		)[0]
		energy_batches.append(predictions.detach().cpu().numpy().reshape(-1))
		dwd_i1_batches.append(gradients[:, 0].detach().cpu().numpy().reshape(-1))
		dwd_i2_batches.append(gradients[:, 1].detach().cpu().numpy().reshape(-1))

	return {
		'energy': np.concatenate(energy_batches).astype(np.float64),
		'dwd_i1': np.concatenate(dwd_i1_batches).astype(np.float64),
		'dwd_i2': np.concatenate(dwd_i2_batches).astype(np.float64),
	}


def evaluate_energy_grid(
	model_path: Path,
	lambda1: np.ndarray,
	lambda2: np.ndarray,
	lambda3: np.ndarray,
	batch_size: int,
	apply_reference_shift: bool,
) -> dict[str, np.ndarray | float]:
	torch, model = load_nn_model(model_path)
	invariants = compute_invariants(lambda1, lambda2, lambda3)
	response = _evaluate_model_invariant_response(torch, model, invariants, batch_size)
	energy = response['energy']
	dwd_i1 = response['dwd_i1']
	dwd_i2 = response['dwd_i2']
	stress_lambda1 = 2.0 * (lambda1 - lambda3 ** 2.0 / lambda1) * (dwd_i1 + lambda2 ** 2.0 * dwd_i2)
	stress_lambda2 = 2.0 * (lambda2 - lambda3 ** 2.0 / lambda2) * (dwd_i1 + lambda1 ** 2.0 * dwd_i2)
	reference_energy = 0.0
	if apply_reference_shift:
		reference_invariants = compute_invariants(
			np.array([1.0], dtype=np.float64),
			np.array([1.0], dtype=np.float64),
			np.array([1.0], dtype=np.float64),
		)
		with torch.no_grad():
			reference_energy = float(
				model(torch.tensor(reference_invariants, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)[0]
			)
		energy = energy - reference_energy

	return {
		'lambda1': lambda1,
		'lambda2': lambda2,
		'lambda3': lambda3,
		'i1': invariants[:, 0].astype(np.float64),
		'i2': invariants[:, 1].astype(np.float64),
		'energy': energy,
		'stress_lambda1': stress_lambda1,
		'stress_lambda2': stress_lambda2,
		'reference_energy': reference_energy,
	}


def evaluate_energy_loadcases(
	model_path: Path,
	mode_name: np.ndarray,
	stretch: np.ndarray,
	batch_size: int,
	apply_reference_shift: bool,
) -> dict[str, np.ndarray | float]:
	torch, model = load_nn_model(model_path)
	get_mode_stretches = _load_mode_stretches_helper()
	lambda1_chunks: list[np.ndarray] = []
	lambda2_chunks: list[np.ndarray] = []
	lambda3_chunks: list[np.ndarray] = []
	invariant_chunks: list[np.ndarray] = []
	stress_factor_chunks: list[np.ndarray] = []
	dwd_i2_factor_chunks: list[np.ndarray] = []
	nominal_stress_chunks: list[np.ndarray] = []
	energy_chunks: list[np.ndarray] = []
	reference_energy = 0.0

	for current_mode in ('UT', 'UC', 'PS', 'ET', 'SS', 'SN'):
		mode_mask = mode_name == current_mode
		if not np.any(mode_mask):
			continue
		mode_stretch = stretch[mode_mask].astype(np.float64)
		lambda1, lambda2, lambda3 = get_mode_stretches(mode_stretch.astype(np.float32), current_mode)
		lambda1 = lambda1.astype(np.float64)
		lambda2 = lambda2.astype(np.float64)
		lambda3 = lambda3.astype(np.float64)
		invariants = compute_invariants(lambda1, lambda2, lambda3)
		response = _evaluate_model_invariant_response(torch, model, invariants, batch_size)
		energy = response['energy']
		if current_mode in ('UT', 'UC'):
			stress_factor = 2.0 * (mode_stretch - mode_stretch ** -2.0)
			dwd_i2_factor = 1.0 / mode_stretch
		elif current_mode in ('SS', 'SN'):
			stress_factor = 2.0 * mode_stretch
			dwd_i2_factor = np.ones_like(mode_stretch)
		elif current_mode == 'PS':
			stress_factor = 2.0 * (mode_stretch - mode_stretch ** -3.0)
			dwd_i2_factor = np.ones_like(mode_stretch)
		else:
			stress_factor = 2.0 * (mode_stretch - mode_stretch ** -5.0)
			dwd_i2_factor = mode_stretch ** 2.0
		nominal_stress = stress_factor * (response['dwd_i1'] + dwd_i2_factor * response['dwd_i2'])
		lambda1_chunks.append(lambda1)
		lambda2_chunks.append(lambda2)
		lambda3_chunks.append(lambda3)
		invariant_chunks.append(invariants.astype(np.float64))
		stress_factor_chunks.append(stress_factor.astype(np.float64))
		dwd_i2_factor_chunks.append(dwd_i2_factor.astype(np.float64))
		nominal_stress_chunks.append(nominal_stress.astype(np.float64))
		energy_chunks.append(energy.astype(np.float64))

	combined_lambda1 = np.concatenate(lambda1_chunks).astype(np.float64)
	combined_lambda2 = np.concatenate(lambda2_chunks).astype(np.float64)
	combined_lambda3 = np.concatenate(lambda3_chunks).astype(np.float64)
	combined_invariants = np.concatenate(invariant_chunks, axis=0).astype(np.float64)
	combined_energy = np.concatenate(energy_chunks).astype(np.float64)
	combined_stress_factor = np.concatenate(stress_factor_chunks).astype(np.float64)
	combined_dwd_i2_factor = np.concatenate(dwd_i2_factor_chunks).astype(np.float64)
	combined_nominal_stress = np.concatenate(nominal_stress_chunks).astype(np.float64)

	if apply_reference_shift:
		reference_invariants = compute_invariants(
			np.array([1.0], dtype=np.float64),
			np.array([1.0], dtype=np.float64),
			np.array([1.0], dtype=np.float64),
		)
		with torch.no_grad():
			reference_energy = float(
				model(torch.tensor(reference_invariants, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)[0]
			)
		combined_energy = combined_energy - reference_energy

	return {
		'mode_name': mode_name,
		'stretch': stretch,
		'lambda1': combined_lambda1,
		'lambda2': combined_lambda2,
		'lambda3': combined_lambda3,
		'i1': combined_invariants[:, 0],
		'i2': combined_invariants[:, 1],
		'energy': combined_energy,
		'nominal_stress': combined_nominal_stress,
		'stress_factor': combined_stress_factor,
		'dwd_i2_factor': combined_dwd_i2_factor,
		'reference_energy': reference_energy,
	}


def write_dataset(output_csv: Path, metadata_json: Path, dataset: dict[str, np.ndarray | float], args) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.writer(handle)
		if args.sampling_mode == 'grid':
			writer.writerow(['lambda1', 'lambda2', 'lambda3', 'I1', 'I2', 'energy', 'stress_lambda1', 'stress_lambda2'])
			for row in zip(
				dataset['lambda1'],
				dataset['lambda2'],
				dataset['lambda3'],
				dataset['i1'],
				dataset['i2'],
				dataset['energy'],
				dataset['stress_lambda1'],
				dataset['stress_lambda2'],
				strict=True,
			):
				writer.writerow([f'{float(value):.17g}' for value in row])
		else:
			writer.writerow(['mode_id', 'stretch', 'lambda1', 'lambda2', 'lambda3', 'I1', 'I2', 'energy', 'nominal_stress', 'stress_factor', 'dwd_i2_factor'])
			mode_id_map = {'UT': 1.0, 'PS': 2.0, 'ET': 3.0, 'UC': 4.0, 'SS': 5.0, 'SN': 6.0}
			for mode_name, row in zip(
				dataset['mode_name'],
				zip(
					dataset['stretch'],
					dataset['lambda1'],
					dataset['lambda2'],
					dataset['lambda3'],
					dataset['i1'],
					dataset['i2'],
					dataset['energy'],
					dataset['nominal_stress'],
					dataset['stress_factor'],
					dataset['dwd_i2_factor'],
					strict=True,
				),
				strict=True,
			):
				writer.writerow([f'{mode_id_map[str(mode_name)]:.17g}', *(f'{float(value):.17g}' for value in row)])

	metadata = {
		'model_path': str(args.model_path.resolve()),
		'output_csv': str(output_csv.resolve()),
		'sampling_mode': args.sampling_mode,
		'grid_point_count': int(len(dataset['energy'])),
		'reference_shift_applied': not args.no_reference_shift,
		'reference_energy': float(dataset['reference_energy']),
	}
	if args.sampling_mode == 'grid':
		metadata.update(
			lambda1_range=[args.lambda1_min, args.lambda1_max],
			lambda2_range=[args.lambda2_min, args.lambda2_max],
			lambda1_points=args.lambda1_points,
			lambda2_points=args.lambda2_points,
		)
	else:
		metadata.update(
			ut_stretch_range=[float(args.ut_stretch_range[0]), float(args.ut_stretch_range[1])],
			ps_stretch_range=[float(args.ps_stretch_range[0]), float(args.ps_stretch_range[1])],
			et_stretch_range=[float(args.et_stretch_range[0]), float(args.et_stretch_range[1])],
			ut_points=args.ut_points,
			ps_points=args.ps_points,
			et_points=args.et_points,
		)
		if getattr(args, 'uc_points', 0):
			metadata.update(
				uc_stretch_range=[float(args.uc_stretch_range[0]), float(args.uc_stretch_range[1])],
				uc_points=args.uc_points,
			)
		if getattr(args, 'ss_points', 0):
			metadata.update(
				ss_shear_range=[float(args.ss_shear_range[0]), float(args.ss_shear_range[1])],
				ss_points=args.ss_points,
			)
	metadata_json.parent.mkdir(parents=True, exist_ok=True)
	metadata_json.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def run_export(args) -> None:
	if args.sampling_mode == 'grid':
		lambda1, lambda2, lambda3 = build_lambda_grid(args)
		dataset = evaluate_energy_grid(
			model_path=args.model_path,
			lambda1=lambda1,
			lambda2=lambda2,
			lambda3=lambda3,
			batch_size=args.batch_size,
			apply_reference_shift=not args.no_reference_shift,
		)
	else:
		loadcase_samples = build_loadcase_samples(args)
		dataset = evaluate_energy_loadcases(
			model_path=args.model_path,
			mode_name=loadcase_samples['mode_name'],
			stretch=loadcase_samples['stretch'],
			batch_size=args.batch_size,
			apply_reference_shift=not args.no_reference_shift,
		)
	metadata_json = args.metadata_json or args.output_csv.with_name(f'{args.output_csv.stem}_metadata.json')
	write_dataset(args.output_csv, metadata_json, dataset, args)
	print(f'Exported {len(dataset["energy"])} samples to {args.output_csv}.')
