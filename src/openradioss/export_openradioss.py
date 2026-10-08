#!/usr/bin/env python3
"""Export an NN_invariant material package for OpenRadioss USER01.

This is the Starter-facing deployment entry point.  It intentionally produces
a small flat text file so the Fortran user material does not need JSON, SymPy,
PyTorch, or PySR at Engine time.
"""

from __future__ import annotations

import argparse
import ast
import os
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from importlib.util import find_spec


NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"

FLAT_HEADER = "NN_INVARIANT_MATERIAL_V1"
# Terms with non-negative integer powers of I1/I2 only.
FLAT_EXPRESSION_TYPE_POLY = "INVARIANT_POLYNOMIAL_TERMS_V1"
# Terms with integer powers of either sign (e.g. 1/I2 produced by PySR '/' or
# by post-fit pruning).  Same line layout; the Engine evaluator must accept
# negative exponents.
FLAT_EXPRESSION_TYPE_LAURENT = "INVARIANT_LAURENT_TERMS_V1"
FLAT_EXPRESSION_TYPES = (FLAT_EXPRESSION_TYPE_POLY, FLAT_EXPRESSION_TYPE_LAURENT)

# Operators that keep PySR expressions inside the family the Engine evaluator
# can consume: sums of coef * I1**p * I2**q with integer p, q.  '/' is allowed
# because division by a monomial yields negative integer powers; division by a
# sum still fails at export time with a clear error.
SUPPORTED_BINARY_OPERATORS = ("+", "-", "*", "/")
SUPPORTED_UNARY_OPERATORS = ("square", "cube")
DEFAULT_BINARY_OPERATORS = ["+", "-", "*"]
DEFAULT_UNARY_OPERATORS = ["square"]


def _read_package(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _openradioss_root() -> Path:
    return Path(os.environ.get("OPENRADIOSS_ROOT", str(Path(__file__).resolve().parents[2])))


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "nn_invariant"


def _is_material_package(payload: dict) -> bool:
    return payload.get("material_model") == "openradioss_invariant_hyperelastic"


def _flat_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="ascii") as handle:
        header = handle.readline().strip()
        expression_type = handle.readline().strip()
    return header == FLAT_HEADER and expression_type in FLAT_EXPRESSION_TYPES


def _as_path(value: str | None, *, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def _config_environment(config: dict, config_dir: Path, cache_dir: Path, nn_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["NN_INVARIANT_OUTPUT_DIR"] = str(cache_dir / "output")
    env["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    env["PYTHONPATH"] = str(nn_root) + os.pathsep + env.get("PYTHONPATH", "")
    for key, value in config.get("environment", {}).items():
        if str(key).endswith(("_PATH", "_PROJECT", "_EXE", "_DIR")):
            env[str(key)] = str(_as_path(value, base=config_dir))
        else:
            env[str(key)] = str(value)
    return env


def _config_paths(config_path: Path, config: dict, output: Path | None = None) -> tuple[Path, Path, Path]:
    config_dir = config_path.resolve().parent
    nn_root = _as_path(config.get("nn_invariant_root"), base=config_dir) or _vendor_root()
    nn_root = nn_root.resolve()
    cache_dir = _as_path(config.get("cache_dir"), base=config_dir)
    if cache_dir is None:
        cache_dir = output.resolve().parent if output is not None else config_dir / "nn_cache"
    return config_dir, nn_root, cache_dir.resolve()


def _run(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path | None = None) -> None:
    print("+ " + " ".join(command))
    if log_path is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("+ " + " ".join(command) + "\n")
        log_file.flush()
        subprocess.run(command, cwd=cwd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def _install_julia_packages(config: dict, env: dict[str, str], log_path: Path) -> None:
    packages = config.get("julia_packages", [])
    if not packages:
        return
    julia_exe = env.get("PYTHON_JULIAPKG_EXE")
    julia_project = env.get("PYTHON_JULIAPKG_PROJECT")
    if not julia_exe or not julia_project:
        raise ValueError("julia_packages requires PYTHON_JULIAPKG_EXE and PYTHON_JULIAPKG_PROJECT in environment.")
    package_list = ", ".join(json.dumps(str(package)) for package in packages)
    script = f"import Pkg; Pkg.add([{package_list}]); Pkg.resolve(); Pkg.precompile()"
    _run(
        [julia_exe, f"--project={julia_project}", "--startup-file=no", "-e", script],
        cwd=Path.cwd(),
        env=env,
        log_path=log_path,
    )


def _append_dataset_args(args: list[str], config: dict, config_dir: Path) -> None:
    datasets = config.get("datasets", {})
    mapping = (("UT", "--ut-dataset"), ("PS", "--ps-dataset"), ("ET", "--et-dataset"))
    for key, option in mapping:
        dataset = _as_path(datasets.get(key), base=config_dir)
        if dataset is not None:
            args.extend([option, str(dataset)])


def _dataset_stretch_range(path: Path) -> tuple[float, float]:
    stretch_min: float | None = None
    stretch_max: float | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            columns = stripped.replace(",", " ").split()
            if len(columns) < 2:
                continue
            try:
                stretch = float(columns[1])
            except ValueError:
                continue
            stretch_min = stretch if stretch_min is None else min(stretch_min, stretch)
            stretch_max = stretch if stretch_max is None else max(stretch_max, stretch)
    if stretch_min is None or stretch_max is None:
        raise ValueError(f"Could not infer stretch range from dataset: {path}")
    return stretch_min, stretch_max


def _append_experimental_ranges(args: list[str], config: dict, symbolic_config: dict, config_dir: Path) -> None:
    datasets = config.get("datasets", {})
    mapping = (
        ("UT", "ut_stretch_range", "--ut-stretch-range"),
        ("PS", "ps_stretch_range", "--ps-stretch-range"),
        ("ET", "et_stretch_range", "--et-stretch-range"),
    )
    for dataset_key, range_key, option in mapping:
        if range_key in symbolic_config:
            continue
        dataset = _as_path(datasets.get(dataset_key), base=config_dir)
        if dataset is None:
            continue
        lower, upper = _dataset_stretch_range(dataset)
        args.extend([option, f"{lower:.12g}", f"{upper:.12g}"])


def _append_scalar_args(args: list[str], payload: dict, mapping: dict[str, str]) -> None:
    for key, option in mapping.items():
        if key in payload and payload[key] is not None:
            value = payload[key]
            if isinstance(value, bool):
                if value:
                    args.append(option)
            elif isinstance(value, (list, tuple)):
                args.append(option)
                args.extend(str(item) for item in value)
            else:
                args.extend([option, str(value)])


def _hidden_layers(train_config: dict) -> list[int] | None:
    """Return the hidden-layer widths requested by the config, or None for single-layer."""
    layers = train_config.get("hidden_layers")
    if layers is None:
        return None
    if isinstance(layers, int):
        layers = [layers]
    widths = [int(width) for width in layers]
    if not widths or any(width < 1 for width in widths):
        raise ValueError("train.hidden_layers must be a non-empty list of positive integers.")
    return widths


def _symbolic_operators(symbolic_config: dict) -> tuple[list[str], list[str]]:
    """Return (binary_operators, unary_operators) for the PySR search.

    The Engine only evaluates sums of coef * I1**p * I2**q with integer p, q,
    so operators such as exp/log/sqrt are rejected unless
    ``allow_non_polynomial_operators`` is set explicitly.
    """
    binary = symbolic_config.get("binary_operators", DEFAULT_BINARY_OPERATORS)
    unary = symbolic_config.get("unary_operators", DEFAULT_UNARY_OPERATORS)
    if isinstance(binary, str):
        binary = binary.split()
    if isinstance(unary, str):
        unary = unary.split()
    binary = [str(item) for item in binary]
    unary = [str(item) for item in unary]
    if not symbolic_config.get("allow_non_polynomial_operators", False):
        bad_binary = [op for op in binary if op not in SUPPORTED_BINARY_OPERATORS]
        bad_unary = [op for op in unary if op not in SUPPORTED_UNARY_OPERATORS]
        if bad_binary or bad_unary:
            raise ValueError(
                "symbolic.binary_operators/unary_operators contain operators that do not produce "
                f"polynomial/Laurent expressions ({bad_binary + bad_unary}); the OpenRadioss Engine "
                "only evaluates INVARIANT_POLYNOMIAL_TERMS_V1 / INVARIANT_LAURENT_TERMS_V1. Remove "
                "them or set symbolic.allow_non_polynomial_operators=true (export will then fail if "
                "the best expression is not a sum of integer-power monomials)."
            )
    return binary, unary


def _run_training_pipeline(config_path: Path, output: Path, nu: float) -> Path:
    config = _read_package(config_path)
    if _is_material_package(config):
        return config_path

    config_dir, nn_root, cache_dir = _config_paths(config_path, config, output)
    if not (nn_root / "main.py").is_file():
        raise FileNotFoundError(f"NN_invariant deployment root is missing main.py: {nn_root}")

    output_root = cache_dir / "output"
    sr_output_dir = output_root / "SR_output"
    nn_output_dir = output_root / "NN_output"
    material_package = _as_path(config.get("material_package_json"), base=config_dir) or sr_output_dir / "material_package.json"
    log_path = cache_dir / "nn_invariant.log"

    if material_package.is_file() and not config.get("force_retrain", False):
        print(f"Reusing NN_invariant material package: {material_package}")
        return material_package

    python = str(_as_path(config.get("python"), base=config_dir) or Path(sys.executable))
    env = _config_environment(config, config_dir, cache_dir, nn_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_config = config.get("train", {})
    symbolic_config = config.get("symbolic", {})
    workflow = config.get("workflow", "experimental")

    if not config.get("skip_train", False):
        train_args = [
            python,
            str(nn_root / "main.py"),
            "--train-mode",
            str(train_config.get("train_mode", "stress")),
            "--hidden-neurons",
            str(train_config.get("hidden_neurons", 3)),
            "--activation",
            str(train_config.get("activation", "softplus")),
            "--optimizer",
            str(train_config.get("optimizer", "lbfgs")),
            "--learning-rate",
            str(train_config.get("learning_rate", 0.2)),
            "--epochs",
            str(train_config.get("epochs", 200)),
            "--print-every",
            str(train_config.get("print_every", 20)),
            "--lbfgs-max-iter",
            str(train_config.get("lbfgs_max_iter", 30)),
            "--lbfgs-history-size",
            str(train_config.get("lbfgs_history_size", 50)),
            "--random-seed",
            str(train_config.get("random_seed", 42)),
        ]
        hidden_layers = _hidden_layers(train_config)
        if hidden_layers is not None:
            train_args.append("--hidden-layers")
            train_args.extend(str(width) for width in hidden_layers)
        if workflow == "synthetic":
            train_args.append("--use-synthetic-data")
        _append_dataset_args(train_args, config, config_dir)
        _append_scalar_args(
            train_args,
            train_config,
            {
                "synthetic_model": "--synthetic-model",
                "synthetic_point_count": "--synthetic-point-count",
                "synthetic_noise_std": "--synthetic-noise-std",
                "synthetic_ut_range": "--synthetic-ut-range",
                "synthetic_ps_range": "--synthetic-ps-range",
                "synthetic_et_range": "--synthetic-et-range",
                "mooney_rivlin_c10": "--mooney-rivlin-c10",
                "mooney_rivlin_c01": "--mooney-rivlin-c01",
                "arruda_boyce_mu": "--arruda-boyce-mu",
                "arruda_boyce_lambda_m": "--arruda-boyce-lambda-m",
                "ogden_mu": "--ogden-mu",
                "ogden_alpha": "--ogden-alpha",
            },
        )
        _run(train_args, cwd=nn_root, env=env, log_path=log_path)

    if not config.get("skip_symbolic", False):
        _install_julia_packages(config, env, log_path)
        sr_csv = sr_output_dir / "nn_loadcase_samples.csv"
        symbolic_args = [
            python,
            str(nn_root / "Symbolic_Regression" / "nn_to_symbolic.py"),
            "pipeline",
            "--sampling-mode",
            str(symbolic_config.get("sampling_mode", "loadcases")),
            "--model-path",
            str(nn_output_dir / "ffbp_model.pt"),
            "--output-csv",
            str(sr_csv),
            "--output-dir",
            str(sr_output_dir),
            "--material-package-json",
            str(material_package),
            "--feature-space",
            str(symbolic_config.get("feature_space", "invariants")),
            "--energy-loss-weight",
            str(symbolic_config.get("energy_loss_weight", 1.0)),
            "--stress-loss-weight",
            str(symbolic_config.get("stress_loss_weight", 1.0)),
            "--niterations",
            str(symbolic_config.get("niterations", 80)),
            "--population-size",
            str(symbolic_config.get("population_size", 60)),
            "--populations",
            str(symbolic_config.get("populations", 12)),
            "--maxsize",
            str(symbolic_config.get("maxsize", 18)),
            "--bulk-kappa",
            str(symbolic_config.get("bulk_kappa", 0.0)),
            "--simplify-rmse-tolerance",
            str(symbolic_config.get("simplify_rmse_tolerance", 0.02)),
            "--simplify-energy-weight",
            str(symbolic_config.get("simplify_energy_weight", 0.0)),
            "--simplify-stress-weight",
            str(symbolic_config.get("simplify_stress_weight", 1.0)),
        ]
        binary_operators, unary_operators = _symbolic_operators(symbolic_config)
        symbolic_args.append("--binary-operators")
        symbolic_args.extend(binary_operators)
        symbolic_args.append("--unary-operators")
        symbolic_args.extend(unary_operators)
        _append_dataset_args(symbolic_args, config, config_dir)
        if workflow == "experimental":
            _append_experimental_ranges(symbolic_args, config, symbolic_config, config_dir)
        _append_scalar_args(
            symbolic_args,
            symbolic_config,
            {
                "ut_stretch_range": "--ut-stretch-range",
                "ps_stretch_range": "--ps-stretch-range",
                "et_stretch_range": "--et-stretch-range",
                "ut_points": "--ut-points",
                "ps_points": "--ps-points",
                "et_points": "--et-points",
                "singularity_threshold": "--singularity-threshold",
                "seed": "--seed",
                "export_python": "--export-python",
                "fit_python": "--fit-python",
            },
        )
        if symbolic_config.get("deterministic", True):
            symbolic_args.append("--deterministic")
        if symbolic_config.get("valanis_landel", False):
            symbolic_args.append("--valanis-landel")
        if symbolic_config.get("simplify_expression", False):
            symbolic_args.append("--simplify-expression")
        if symbolic_config.get("skip_experiment_prediction", False):
            symbolic_args.append("--skip-experiment-prediction")
        _run(symbolic_args, cwd=nn_root, env=env, log_path=log_path)

    if not material_package.is_file():
        raise FileNotFoundError(f"NN_invariant did not create material package: {material_package}")
    return material_package


def _coefficient(pattern: str, text: str, name: str) -> float:
    match = re.search(pattern, text.replace(" ", ""))
    if not match:
        raise ValueError(f"Could not extract coefficient {name!r} from expression: {text}")
    return float(match.group(1))


def _extract_law291_coefficients(package: dict) -> tuple[float, float, float, float, float, float]:
    constitutive = package.get("constitutive", {})
    energy = str(constitutive.get("isochoric_energy", ""))
    dwd_i1 = str(constitutive.get("dW_dI1", ""))
    dwd_i2 = str(constitutive.get("dW_dI2", ""))

    # Current NN_invariant LAW291 deployment family:
    # W = A*I1**5 + B*I1*I2**2 + C*I1 + D*I2 + E
    # dW/dI1 = 5*A*I1**4 + B*I2**2 + C
    # dW/dI2 = 2*B*I1*I2 + D
    five_a = _coefficient(NUMBER + r"\*I1\*\*4", dwd_i1, "5*A")
    bcoef = _coefficient(NUMBER + r"\*I2\*\*2", dwd_i1, "B")
    dcoef = _coefficient(NUMBER + r"$", dwd_i2, "D")
    ccoef = _coefficient(NUMBER + r"$", dwd_i1, "C")
    acoef = five_a / 5.0
    ecoef = _coefficient(NUMBER + r"$", energy, "E")
    return acoef, bcoef, ccoef, dcoef, ecoef, 0.0


def _extract_polynomial_terms(package: dict) -> list[tuple[float, int, int]]:
    expression_text = str(package.get("constitutive", {}).get("isochoric_energy", ""))
    try:
        return _extract_polynomial_terms_with_ast(expression_text)
    except ValueError:
        pass

    try:
        import sympy
    except ImportError as error:
        raise ImportError(
            "Could not parse the symbolic expression with the built-in polynomial parser, "
            "and sympy is not installed in this Python. Install sympy or run the exporter "
            "with the NN/PySR Python environment."
        ) from error

    i1_symbol, i2_symbol = sympy.symbols("I1 I2")
    locals_map = {
        "I1": i1_symbol,
        "I2": i2_symbol,
        "square": lambda value: value ** 2,
        "cube": lambda value: value ** 3,
    }
    expression = sympy.sympify(expression_text.replace("^", "**"), locals=locals_map)
    expanded = sympy.expand(expression)
    if expanded.free_symbols - {i1_symbol, i2_symbol}:
        raise ValueError(f"Unsupported symbols in expression: {expanded.free_symbols}")
    polynomial: dict[tuple[int, int], float] = {}
    for term in sympy.Add.make_args(expanded):
        coefficient, factors = term.as_coeff_mul()
        powers = {i1_symbol: 0, i2_symbol: 0}
        for factor in factors:
            base, exponent = factor.as_base_exp()
            if base not in powers or not exponent.is_Integer:
                raise ValueError(
                    f"Term {term} is not coef * I1**p * I2**q with integer p, q; "
                    "the OpenRadioss Engine cannot evaluate it."
                )
            powers[base] += int(exponent)
        if not coefficient.is_number:
            raise ValueError(f"Term {term} has a non-numeric coefficient.")
        key = (powers[i1_symbol], powers[i2_symbol])
        polynomial[key] = polynomial.get(key, 0.0) + float(coefficient)
    terms = [(coef, p1, p2) for (p1, p2), coef in polynomial.items() if abs(coef) > 0.0]
    terms.sort(key=lambda item: (item[1] + item[2], item[1], item[2]))
    if not terms:
        raise ValueError("Symbolic expression contains no polynomial terms.")
    return terms


def _poly_add(left: dict[tuple[int, int], float], right: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    result = dict(left)
    for powers, coefficient in right.items():
        result[powers] = result.get(powers, 0.0) + coefficient
    return result


def _poly_scale(poly: dict[tuple[int, int], float], factor: float) -> dict[tuple[int, int], float]:
    return {powers: coefficient * factor for powers, coefficient in poly.items()}


def _poly_mul(left: dict[tuple[int, int], float], right: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for (i1_left, i2_left), coef_left in left.items():
        for (i1_right, i2_right), coef_right in right.items():
            powers = (i1_left + i1_right, i2_left + i2_right)
            result[powers] = result.get(powers, 0.0) + coef_left * coef_right
    return result


def _poly_inverse(poly: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Invert a single monomial coef * I1**p * I2**q; sums cannot be inverted."""
    nonzero = {powers: coef for powers, coef in poly.items() if coef != 0.0}
    if len(nonzero) != 1:
        raise ValueError("Division is only supported by a single monomial term.")
    ((p1, p2), coef), = nonzero.items()
    return {(-p1, -p2): 1.0 / coef}


def _poly_pow(poly: dict[tuple[int, int], float], exponent: int) -> dict[tuple[int, int], float]:
    if exponent < 0:
        poly = _poly_inverse(poly)
        exponent = -exponent
    result = {(0, 0): 1.0}
    for _ in range(exponent):
        result = _poly_mul(result, poly)
    return result


def _ast_to_poly(node: ast.AST) -> dict[tuple[int, int], float]:
    if isinstance(node, ast.Expression):
        return _ast_to_poly(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return {(0, 0): float(node.value)}
    if isinstance(node, ast.Name):
        if node.id == "I1":
            return {(1, 0): 1.0}
        if node.id == "I2":
            return {(0, 1): 1.0}
        raise ValueError(f"Unsupported symbol {node.id!r}.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _poly_scale(_ast_to_poly(node.operand), -1.0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _ast_to_poly(node.operand)
    if isinstance(node, ast.BinOp):
        left = _ast_to_poly(node.left)
        right = _ast_to_poly(node.right)
        if isinstance(node.op, ast.Add):
            return _poly_add(left, right)
        if isinstance(node.op, ast.Sub):
            return _poly_add(left, _poly_scale(right, -1.0))
        if isinstance(node.op, ast.Mult):
            return _poly_mul(left, right)
        if isinstance(node.op, ast.Div):
            return _poly_mul(left, _poly_inverse(right))
        if isinstance(node.op, ast.Pow):
            exponent = _integer_exponent(node.right)
            if exponent is None:
                raise ValueError("Only integer powers are supported.")
            return _poly_pow(left, exponent)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
        if node.func.id == "square":
            return _poly_pow(_ast_to_poly(node.args[0]), 2)
        if node.func.id == "cube":
            return _poly_pow(_ast_to_poly(node.args[0]), 3)
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def _integer_exponent(node: ast.AST) -> int | None:
    """Return the integer value of a literal exponent such as 2, -1, or 2.0."""
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sign = -1
        node = node.operand
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        value = float(node.value)
        if value.is_integer():
            return sign * int(value)
    return None


def _extract_polynomial_terms_with_ast(expression_text: str) -> list[tuple[float, int, int]]:
    tree = ast.parse(expression_text.replace("^", "**"), mode="eval")
    polynomial = _ast_to_poly(tree)
    terms = [
        (coefficient, powers[0], powers[1])
        for powers, coefficient in polynomial.items()
        if abs(coefficient) > 0.0
    ]
    terms.sort(key=lambda item: (item[1] + item[2], item[1], item[2]))
    if not terms:
        raise ValueError("Symbolic expression contains no polynomial terms.")
    return terms


def _range(package: dict, key: str) -> tuple[float, float]:
    values = package.get("valid_domain", {}).get(key, [math.nan, math.nan])
    return float(values[0]), float(values[1])


def export_openradioss(config: Path, output: Path, nu: float) -> None:
    output = output.resolve()
    config_payload = _read_package(config)
    if (
        not _is_material_package(config_payload)
        and not config_payload.get("force_retrain", False)
        and config_payload.get("reuse_flat", True)
        and _flat_is_valid(output)
    ):
        print(f"Reusing OpenRadioss flat material package: {output}")
        return

    material_package_path = _run_training_pipeline(config, output, nu)
    package = _read_package(material_package_path)
    if int(package.get("format_version", 0)) != 1:
        raise ValueError("Only NN_invariant material package format_version=1 is supported.")
    if package.get("feature_space") != "invariants":
        raise ValueError("Only invariant-space material packages are supported.")

    terms = _extract_polynomial_terms(package)
    constitutive = package.get("constitutive", {})
    bulk_kappa = float(constitutive.get("bulk_kappa", 0.0))

    dwd_i1_ref = sum(coef * p1 * 3.0 ** (p1 - 1) * 3.0 ** p2 for coef, p1, p2 in terms if p1 != 0)
    dwd_i2_ref = sum(coef * p2 * 3.0 ** p1 * 3.0 ** (p2 - 1) for coef, p1, p2 in terms if p2 != 0)
    gref = 2.0 * (dwd_i1_ref + dwd_i2_ref)
    if gref <= 0.0:
        raise ValueError(f"Initial shear modulus must be positive, got {gref:g}.")
    if bulk_kappa <= 0.0:
        if not (0.0 <= nu < 0.5):
            raise ValueError("bulk_kappa <= 0 requires 0 <= --nu < 0.5.")
        bulk_kappa = 2.0 * gref * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))

    i1_min, i1_max = _range(package, "I1_range")
    i2_min, i2_max = _range(package, "I2_range")
    rmse = float(package.get("diagnostics", {}).get("rmse", math.nan))
    model_hash = str(package.get("model_hash", "unknown"))

    if any(p1 < 0 or p2 < 0 for _, p1, p2 in terms):
        expression_type = FLAT_EXPRESSION_TYPE_LAURENT
    else:
        expression_type = FLAT_EXPRESSION_TYPE_POLY

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii") as handle:
        handle.write(f"{FLAT_HEADER}\n")
        handle.write(f"{expression_type}\n")
        handle.write(f"{len(terms)}\n")
        for coef, p1, p2 in terms:
            handle.write(f"{coef:.17g} {p1:d} {p2:d}\n")
        handle.write(f"{bulk_kappa:.17g}\n")
        handle.write(f"{gref:.17g}\n")
        handle.write(f"{i1_min:.17g} {i1_max:.17g}\n")
        handle.write(f"{i2_min:.17g} {i2_max:.17g}\n")
        handle.write(f"{rmse:.17g}\n")
        handle.write(f"{model_hash[:16]}\n")

    _print_export_summary(material_package_path, package, terms, gref, bulk_kappa, output, expression_type)


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return _read_package(path)
    except (OSError, ValueError):
        return None


def _print_export_summary(
    material_package_path: Path,
    package: dict,
    terms: list[tuple[float, int, int]],
    gref: float,
    bulk_kappa: float,
    output: Path,
    expression_type: str,
) -> None:
    """Report what was deployed: network architecture, pruning result, polynomial terms."""
    sr_dir = material_package_path.resolve().parent
    nn_metadata = _load_json_if_exists(sr_dir.parent / "NN_output" / "ffbp_model.json")
    simplification = _load_json_if_exists(sr_dir / "simplification_summary.json")

    print(f"Wrote OpenRadioss flat material package: {output}")
    print(f"  format         {FLAT_HEADER} / {expression_type}")
    if nn_metadata is not None:
        layers = nn_metadata.get("hidden_layers") or [nn_metadata.get("hidden_neurons")]
        widths = " -> ".join(str(width) for width in layers)
        print(f"  network        2 -> {widths} -> 1 ({nn_metadata.get('activation', 'softplus')})")
    if simplification is not None:
        if simplification.get("changed"):
            removed = simplification.get("removed_terms", [])
            print(f"  pruning        removed {len(removed)} term(s): {', '.join(str(t) for t in removed)}")
            original = simplification.get("original_metrics", {}).get("stress_rmse")
            final = simplification.get("final_metrics", {}).get("stress_rmse")
            if original is not None and final is not None:
                print(f"  pruning rmse   stress {float(original):.6g} -> {float(final):.6g}")
        elif simplification.get("skipped_reason"):
            print(f"  pruning        skipped: {simplification['skipped_reason']}")
        else:
            print("  pruning        no term removed within tolerance")
    print(f"  expression     {package.get('constitutive', {}).get('isochoric_energy', '')}")
    print(f"  terms          {len(terms)}")
    for coef, p1, p2 in terms:
        print(f"    {coef:+.10g} * I1^{p1} * I2^{p2}")
    print(f"  shear modulus  {gref:.10g}")
    print(f"  bulk modulus   {bulk_kappa:.10g}")


def _module_status(module_name: str) -> str:
    if find_spec(module_name) is None:
        return "MISSING"
    return "OK"


def _python_module_status(python: Path, modules: tuple[str, ...], env: dict[str, str]) -> tuple[bool, str]:
    code = (
        "import importlib.util, sys\n"
        "missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]\n"
        "print('OK' if not missing else 'MISSING ' + ','.join(missing))\n"
        "raise SystemExit(0 if not missing else 1)\n"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code, *modules],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    message = (result.stdout + result.stderr).strip()
    return result.returncode == 0, message


def _pysr_import_status(python: Path, env: dict[str, str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [str(python), "-c", "import pysr; print(pysr.__version__)"],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    message = (result.stdout + result.stderr).strip()
    return result.returncode == 0, message


def doctor(config_path: Path | None = None) -> None:
    print("NN_invariant OpenRadioss exporter: OK")
    print(f"Python executable: {sys.executable}")
    vendor = _vendor_root()
    print(f"OpenRadioss root: {_openradioss_root()}")
    print(f"Bundled NN_invariant root: {vendor}")
    print(f"Bundled source present: {(vendor / 'main.py').is_file()}")
    print("Python modules:")
    for module in ("numpy", "torch", "sklearn", "sympy", "matplotlib", "pysr"):
        print(f"  {module:<10} {_module_status(module)}")

    if config_path is None:
        return

    config = _read_package(config_path)
    if _is_material_package(config):
        print(f"Config: {config_path} is an NN_invariant material package.")
        return

    config_dir, nn_root, cache_dir = _config_paths(config_path, config)
    env = _config_environment(config, config_dir, cache_dir, nn_root)
    train_python = _as_path(config.get("python"), base=config_dir) or Path(sys.executable)
    fit_python = _as_path(config.get("symbolic", {}).get("fit_python"), base=config_dir) or train_python

    print("Config check:")
    print(f"  config        {config_path.resolve()}")
    print(f"  nn_root       {nn_root} {'OK' if (nn_root / 'main.py').is_file() else 'MISSING'}")
    print(f"  cache_dir     {cache_dir}")
    print(f"  train_python  {train_python}")
    ok, message = _python_module_status(Path(train_python), ("numpy", "torch", "sklearn", "sympy", "matplotlib"), env)
    print(f"  train deps    {'OK' if ok else 'MISSING'} {message}")
    print(f"  fit_python    {fit_python}")
    ok, message = _python_module_status(Path(fit_python), ("numpy", "sympy", "pysr"), env)
    print(f"  fit deps      {'OK' if ok else 'MISSING'} {message}")
    ok, message = _pysr_import_status(Path(fit_python), env)
    print(f"  pysr import   {'OK' if ok else 'FAILED'}")
    if message:
        for line in message.splitlines()[-20:]:
            print(f"    {line}")
    if config.get("julia_packages"):
        print(f"  julia pkgs    declared {', '.join(str(pkg) for pkg in config['julia_packages'])}")

    train_config = config.get("train", {})
    symbolic_config = config.get("symbolic", {})
    print("Pipeline options:")
    try:
        hidden_layers = _hidden_layers(train_config) or [int(train_config.get("hidden_neurons", 3))]
        print(f"  network       2 -> {' -> '.join(str(w) for w in hidden_layers)} -> 1 "
              f"({train_config.get('activation', 'softplus')})")
    except ValueError as error:
        print(f"  network       INVALID {error}")
    try:
        binary_operators, unary_operators = _symbolic_operators(symbolic_config)
        print(f"  operators     binary {' '.join(binary_operators)} | unary {' '.join(unary_operators) or '-'}")
    except ValueError as error:
        print(f"  operators     INVALID {error}")
    if symbolic_config.get("simplify_expression", False):
        print(
            "  pruning       enabled "
            f"(rmse tolerance {symbolic_config.get('simplify_rmse_tolerance', 0.02)}, "
            f"energy weight {symbolic_config.get('simplify_energy_weight', 0.0)}, "
            f"stress weight {symbolic_config.get('simplify_stress_weight', 1.0)})"
        )
    else:
        print("  pruning       disabled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-openradioss")
    export_parser.add_argument("--config", required=True, type=Path)
    export_parser.add_argument("--output", required=True, type=Path)
    export_parser.add_argument("--nu", type=float, default=0.495)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "doctor":
        doctor(args.config)
    elif args.command == "export-openradioss":
        export_openradioss(args.config, args.output, args.nu)


if __name__ == "__main__":
    main()
