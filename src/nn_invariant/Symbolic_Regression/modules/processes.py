from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

from .config import ENTRYPOINT_PATH


def module_is_available(module_name: str) -> bool:
	try:
		importlib.import_module(module_name)
		return True
	except ImportError:
		return False


def interpreter_has_module(interpreter: Path, module_name: str) -> bool:
	command = [str(interpreter), '-c', f'import {module_name}']
	result = subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	return result.returncode == 0


def normalize_interpreter_path(path: Path) -> Path:
	return path.expanduser() if path.is_absolute() else (Path.cwd() / path).resolve()


def resolve_child_python(candidate: Path | None, default_path: Path, module_name: str) -> Path:
	resolved_candidate = normalize_interpreter_path(candidate) if candidate is not None else None
	if resolved_candidate is not None:
		if not resolved_candidate.is_file():
			raise FileNotFoundError(f'Specified interpreter does not exist: {resolved_candidate}')
		return resolved_candidate

	normalized_default = normalize_interpreter_path(default_path)
	if normalized_default.is_file():
		return normalized_default

	raise FileNotFoundError(
		f'Could not find a Python interpreter for module {module_name!r}. '
		f'Looked for {normalized_default}. Pass an explicit interpreter path instead.'
	)


def build_subcommand_args(args, command_name: str) -> list[str]:
	command_args = [command_name]

	if command_name in ('export', 'pipeline'):
		command_args.extend(
			[
				'--model-path', str(args.model_path),
				'--sampling-mode', str(args.sampling_mode),
				'--lambda1-min', str(args.lambda1_min),
				'--lambda1-max', str(args.lambda1_max),
				'--lambda1-points', str(args.lambda1_points),
				'--lambda2-min', str(args.lambda2_min),
				'--lambda2-max', str(args.lambda2_max),
				'--lambda2-points', str(args.lambda2_points),
				'--ut-stretch-range', str(args.ut_stretch_range[0]), str(args.ut_stretch_range[1]),
				'--ut-points', str(args.ut_points),
				'--ps-stretch-range', str(args.ps_stretch_range[0]), str(args.ps_stretch_range[1]),
				'--ps-points', str(args.ps_points),
				'--et-stretch-range', str(args.et_stretch_range[0]), str(args.et_stretch_range[1]),
				'--et-points', str(args.et_points),
				'--uc-stretch-range', str(args.uc_stretch_range[0]), str(args.uc_stretch_range[1]),
				'--uc-points', str(args.uc_points),
				'--ss-shear-range', str(args.ss_shear_range[0]), str(args.ss_shear_range[1]),
				'--ss-points', str(args.ss_points),
				'--batch-size', str(args.batch_size),
				'--output-csv', str(args.output_csv),
			]
		)
		if args.metadata_json is not None:
			command_args.extend(['--metadata-json', str(args.metadata_json)])
		if args.no_reference_shift:
			command_args.append('--no-reference-shift')

	if command_name in ('fit', 'pipeline'):
		input_csv = args.input_csv if hasattr(args, 'input_csv') else args.output_csv
		command_args.extend(
			[
				'--input-csv', str(input_csv),
				'--feature-space', str(args.feature_space),
				'--output-dir', str(args.output_dir),
				'--niterations', str(args.niterations),
				'--population-size', str(args.population_size),
				'--populations', str(args.populations),
				'--maxsize', str(args.maxsize),
				'--energy-loss-weight', str(args.energy_loss_weight),
				'--stress-loss-weight', str(args.stress_loss_weight),
				'--simplify-rmse-tolerance', str(args.simplify_rmse_tolerance),
				'--simplify-energy-weight', str(args.simplify_energy_weight),
				'--simplify-stress-weight', str(args.simplify_stress_weight),
				'--seed', str(args.seed),
			]
		)
		if args.unary_operators:
			command_args.append('--unary-operators')
			command_args.extend(str(operator) for operator in args.unary_operators)
		if args.binary_operators:
			command_args.append('--binary-operators')
			command_args.extend(str(operator) for operator in args.binary_operators)
		if args.deterministic:
			command_args.append('--deterministic')
		if args.valanis_landel:
			command_args.append('--valanis-landel')
		if args.simplify_expression:
			command_args.append('--simplify-expression')
		command_args.extend(
			[
				'--ut-dataset', str(args.ut_dataset),
				'--ps-dataset', str(args.ps_dataset),
				'--et-dataset', str(args.et_dataset),
				'--biaxial-prefix', str(args.biaxial_prefix),
				'--biaxial-base-dir', str(args.biaxial_base_dir),
			]
		)
		for option_name, option_value in (
			('--uc-dataset', getattr(args, 'uc_dataset', None)),
			('--ss-dataset', getattr(args, 'ss_dataset', None)),
			('--sn-dataset', getattr(args, 'sn_dataset', None)),
		):
			if option_value is not None:
				command_args.extend([option_name, str(option_value)])
		if args.enable_biaxial:
			command_args.append('--enable-biaxial')
		if hasattr(args, 'skip_experiment_prediction') and args.skip_experiment_prediction:
			command_args.append('--skip-experiment-prediction')

	if command_name == 'predict-experiment':
		command_args.extend(
			[
				'--equation-file', str(args.equation_file),
				'--feature-space', str(args.feature_space),
				'--output-dir', str(args.output_dir),
				'--ut-dataset', str(args.ut_dataset),
				'--ps-dataset', str(args.ps_dataset),
				'--et-dataset', str(args.et_dataset),
				'--biaxial-prefix', str(args.biaxial_prefix),
				'--biaxial-base-dir', str(args.biaxial_base_dir),
			]
		)
		for option_name, option_value in (
			('--uc-dataset', getattr(args, 'uc_dataset', None)),
			('--ss-dataset', getattr(args, 'ss_dataset', None)),
			('--sn-dataset', getattr(args, 'sn_dataset', None)),
		):
			if option_value is not None:
				command_args.extend([option_name, str(option_value)])
		if args.enable_biaxial:
			command_args.append('--enable-biaxial')

	return command_args


def run_subcommand_with_interpreter(interpreter: Path, command_args: list[str]) -> None:
	command = [str(interpreter), str(ENTRYPOINT_PATH), *command_args]
	subprocess.run(command, check=True)
