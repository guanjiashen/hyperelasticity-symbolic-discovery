# NN_invariant OpenRadioss integration

This directory contains the Starter-facing deployment wrapper for the
NN_invariant invariant hyperelastic workflow.

The OpenRadioss side calls:

```bash
python3 tools/nn_invariant/export_openradioss.py export-openradioss \
  --config nn_openradioss_config.json \
  --output nn_cache/material.flat \
  --nu 0.495
```

`--config` may be either:

- an existing NN_invariant `material_package.json`; or
- an OpenRadioss workflow config like `examples/nn_openradioss_config.json`.

The wrapper writes a flat `NN_INVARIANT_MATERIAL_V1` file.  Each term stores a
coefficient and the integer powers of `I1` and `I2`:

- `INVARIANT_POLYNOMIAL_TERMS_V1`: all powers are non-negative.
- `INVARIANT_LAURENT_TERMS_V1`: same layout, but powers may be negative
  (for example `-1.688/I2` produced when PySR uses `/` or when post-fit
  pruning keeps such a term).  Since the isochoric invariants satisfy
  `I1, I2 >= 3`, negative powers are always well defined.

The exporter picks the tag automatically.  Starter reads those terms into
`UPARAM`; Engine evaluates the derivatives numerically and does not import
Python, PyTorch, PySR, Julia, or JSON.  Expressions that are not sums of
`coef * I1**p * I2**q` (for example `1/(I1+I2)`, `exp(I1)`) are rejected at
export time with a clear error.

## Workflow config options

The `train` block maps to `main.py` arguments:

| key | CLI | note |
| --- | --- | --- |
| `hidden_neurons` | `--hidden-neurons` | single hidden layer width (legacy) |
| `hidden_layers` | `--hidden-layers N1 N2 ...` | widths of all hidden layers, e.g. `[16, 16, 8]`; overrides `hidden_neurons` |
| `activation`, `optimizer`, `learning_rate`, `epochs`, `print_every`, `lbfgs_max_iter`, `lbfgs_history_size`, `random_seed` | same name | |
| `synthetic_model`, `synthetic_point_count`, `synthetic_noise_std`, `synthetic_*_range` | same name | only with `"workflow": "synthetic"` |
| `mooney_rivlin_c10/c01`, `ogden_mu/alpha`, `arruda_boyce_mu`, `arruda_boyce_lambda_m` | same name | synthetic model parameters |

The `symbolic` block maps to `Symbolic_Regression/nn_to_symbolic.py pipeline`:

| key | CLI | default |
| --- | --- | --- |
| `binary_operators` | `--binary-operators` | `["+", "-", "*"]` |
| `unary_operators` | `--unary-operators` | `["square"]` |
| `simplify_expression` | `--simplify-expression` | `false` |
| `simplify_rmse_tolerance` | `--simplify-rmse-tolerance` | `0.02` |
| `simplify_energy_weight` | `--simplify-energy-weight` | `0.0` |
| `simplify_stress_weight` | `--simplify-stress-weight` | `1.0` |
| `niterations`, `population_size`, `populations`, `maxsize`, `seed`, `deterministic`, `energy_loss_weight`, `stress_loss_weight`, `bulk_kappa`, `*_points`, `*_stretch_range`, `singularity_threshold`, `fit_python`, `export_python`, `valanis_landel`, `skip_experiment_prediction` | same name | |

`simplify_expression` enables the post-fit pruning step: additive terms of the
best PySR expression are removed greedily, the remaining coefficients are
refitted, and the simpler model is accepted while the selection score does not
rise by more than `simplify_rmse_tolerance`.  The result is written to
`SR_output/simplification_summary.json` (raw expression in
`raw_best_equation.txt`), and the exporter prints the removed terms.

Only `+ - * /` and `square`/`cube` are accepted by default because the Engine
evaluator handles integer-power monomials only.  Set
`"allow_non_polynomial_operators": true` to pass other operators through to
PySR; export then fails if the selected expression cannot be decomposed.

The exporter prints a deployment summary (network layout, pruning result,
exported terms, shear/bulk moduli) after writing the flat file, and
`doctor --config ...` reports the resolved pipeline options.

Runtime environment:

- `PACKAGE` mode only needs OpenRadioss and the user material shared library.
- `TRAIN` mode needs Python with the dependencies in `requirements.txt`, and
  PySR needs a working Julia installation.
- `RAD_NN_PYTHON` can point Starter to the desired Python interpreter.
- `RAD_NN_EXPORTER` can override the exporter script path.
- The workflow config may include an `environment` object.  This is the
  preferred place to set PySR/Julia variables such as `PYTHON_JULIAPKG_EXE`,
  `PYTHON_JULIAPKG_PROJECT`, and `JULIA_DEPOT_PATH` so Julia writes into the
  case cache instead of a read-only virtual environment.
- Existing `material.flat` and `material_package.json` files are reused by
  default.  Set `force_retrain: true` to force regeneration.

The bundled source snapshot is under `vendor/NN_invariant`.  It intentionally
excludes virtual environments, caches, generated plots, checkpoints, and
training data.  Two files carry an OpenRadioss-specific patch that must be
kept when re-syncing from the upstream NN_invariant tree: `main.py` and
`Symbolic_Regression/modules/config.py` read `NN_INVARIANT_OUTPUT_DIR` so the
pipeline writes into the case cache instead of `vendor/NN_invariant/output`.

Re-sync recipe:

```bash
rsync -a --delete \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.venv/' --exclude '.trash/' \
  --exclude 'output/' --exclude 'FEM/' --exclude 'synthetic_data/' \
  --exclude 'notes.md' --exclude 'SR_notes.md' --exclude 'run_fem.sh' \
  --exclude 'copy_output_to_windows.sh' \
  /path/to/NN_invariant/ tools/nn_invariant/vendor/NN_invariant/
# then re-apply the NN_INVARIANT_OUTPUT_DIR patch in main.py and modules/config.py
```
