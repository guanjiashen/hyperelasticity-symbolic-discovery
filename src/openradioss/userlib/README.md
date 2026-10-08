# LAW291 dynamic user library

This directory implements the NN_invariant invariant hyperelastic constitutive
equation through `/MAT/USER01` without modifying Starter or Engine sources. The
Starter routine can still read legacy hand-entered LAW291 coefficients, and it
also supports NN_invariant `PACKAGE` and `TRAIN` cards.

The implementation targets solid elements with `Ismstr=0`, as used by the
O-Ring example. Thermal expansion is not included by the USER01 interface.

Build with:

```bash
./build.sh
```

The resulting library is `libraduser_law291.so`. The associated verification
case is `exec/cases/O-Ring_USER291_2026`.

Create a flat OpenRadioss package from an NN_invariant material package:

```bash
python3 tools/nn_invariant/export_openradioss.py export-openradioss \
  --config /path/to/material_package.json \
  --output exec/cases/O-Ring_USER291_2026/nn_material.flat \
  --nu 0.499828708839903
```

Use an existing flat package in `/MAT/USER01`:

```text
/MAT/USER01/2
Rubber_USER291_dynamic
2.0e-9
PACKAGE
nn_material.flat
0.499828708839903 1e30
2
```

`TRAIN` has the same card layout but takes the JSON package/config path and a
cache directory:

```text
/MAT/USER01/2
Rubber_USER291_dynamic
2.0e-9
TRAIN
nn_train_config.json
nn_cache
0.499828708839903 1e30
2
```

Starter calls `RAD_NN_PYTHON` and `RAD_NN_EXPORTER` when set.  The example
`run.sh` sets `RAD_NN_EXPORTER` to the OpenRadioss deployment wrapper.  The
Engine routine only consumes numeric `UPARAM` values and does not depend on
Python, PyTorch, PySR, Julia, or JSON.  NN_invariant packages are exported as
`INVARIANT_POLYNOMIAL_TERMS_V1` (non-negative powers) or
`INVARIANT_LAURENT_TERMS_V1` (integer powers of either sign, e.g. `1/I2`), so
the Engine can evaluate PySR expressions beyond the old fixed LAW291
coefficient template.  Both tags share the same line layout:

```text
NN_INVARIANT_MATERIAL_V1
INVARIANT_LAURENT_TERMS_V1
<nterms>
<coef> <p1> <p2>          (nterms lines: W += coef * I1**p1 * I2**p2)
<bulk_kappa>
<gref>
<I1_min> <I1_max>
<I2_min> <I2_max>
<rmse>
<model_hash>
```

`UPARAM` layout written by `lecmuser01`: `uparam(1)=2` (format),
`uparam(2)=nterms`, then `(coef, p1, p2)` triplets, followed by
`bulk_kappa, sigcut, iform, gref, nu, I1_min, I1_max, I2_min, I2_max, rmse`.
At most 20 terms fit in the 80-entry `UPARAM` budget.

## Training options relevant to the material card

The JSON given to a `TRAIN` card is the workflow config documented in
`tools/nn_invariant/README.md`.  The options added with the latest
NN_invariant snapshot are:

- `train.hidden_layers`: list of hidden-layer widths, e.g. `[16, 16, 8]`
  (multi-layer network); `hidden_neurons` remains for the single-layer case.
- `symbolic.simplify_expression` plus `simplify_rmse_tolerance`,
  `simplify_energy_weight`, `simplify_stress_weight`: post-fit pruning of
  additive terms with refit of the remaining coefficients.  Pruned models
  usually have fewer `UPARAM` terms and may include negative powers, which is
  why the Laurent tag exists.
- `symbolic.binary_operators` / `symbolic.unary_operators`: PySR operator
  sets.  Keep them within `+ - * /` and `square`/`cube`; the exporter rejects
  others unless `allow_non_polynomial_operators` is set.

The Starter listing prints every exported term (`coef`, `I1^p1`, `I2^p2`) so
the deployed expression can be checked in the `_0000.out` file.
