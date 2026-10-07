from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cli import parse_args
from .config import DEFAULT_FFNN_PYTHON, DEFAULT_PYSR_PYTHON
from .experimental_prediction import run_predict_experiment
from .exporting import run_export
from .fitting import fit_symbolic_expression
from .processes import (
	build_subcommand_args,
	interpreter_has_module,
	module_is_available,
	resolve_child_python,
	run_subcommand_with_interpreter,
)


def run_pipeline(args) -> None:
	if module_is_available('torch'):
		run_export(args)
	else:
		export_python = resolve_child_python(args.export_python, DEFAULT_FFNN_PYTHON, 'torch')
		print(f'Current environment has no torch. Delegating export to {export_python}.')
		run_subcommand_with_interpreter(export_python, build_subcommand_args(args, 'export'))

	fit_namespace = argparse.Namespace(**vars(args), input_csv=args.output_csv)
	current_python = Path(sys.executable)
	if interpreter_has_module(current_python, 'pysr'):
		# Keep PySR in a fresh process so juliacall is not imported after torch.
		run_subcommand_with_interpreter(current_python, build_subcommand_args(fit_namespace, 'fit'))
		return

	fit_python = resolve_child_python(args.fit_python, DEFAULT_PYSR_PYTHON, 'pysr')
	print(f'Current environment has no pysr. Delegating fit to {fit_python}.')
	run_subcommand_with_interpreter(fit_python, build_subcommand_args(fit_namespace, 'fit'))


def main() -> None:
	args = parse_args()
	if args.command == 'export':
		run_export(args)
		return

	if args.command == 'fit':
		fit_symbolic_expression(args)
		return

	if args.command == 'predict-experiment':
		run_predict_experiment(args)
		return

	if args.command == 'pipeline':
		run_pipeline(args)
		return

	raise ValueError(f'Unsupported command: {args.command}')