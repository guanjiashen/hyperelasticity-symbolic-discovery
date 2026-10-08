# From neural networks to symbolic expressions

This directory converts the invariant-based neural network stored in `ffbp_model.pt` into an interpretable symbolic strain-energy expression.
The original checkpoint path was `/home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output/NN_output/ffbp_model.pt`.
Commands below retain historical development paths; adapt them to your installation and consult the repository-level README for the release layout.

The grid workflow has two stages:

1. Generate strain-energy data with the neural network on a grid with $\lambda_1,\lambda_2\in[1,3]$.
2. Fit these data with PySR, using $W(I_1,I_2)$ by default.

An alternative workflow samples loading paths:

1. Generate neural-network data for uniaxial tension (UT), pure shear (PS), and equibiaxial tension (ET).
2. Export both strain energy and nominal stress along each path.
3. Search with both energy and nominal-stress errors while still returning a strain-energy expression $W$.

## Script commands

- `nn_to_symbolic.py export`: load the trained network. With `--sampling-mode grid`, export `lambda1, lambda2, lambda3, I1, I2, energy, stress_lambda1, stress_lambda2`. With `--sampling-mode loadcases`, export `stretch, lambda1, lambda2, lambda3, I1, I2, energy, nominal_stress` along UT/PS/ET paths.
- `nn_to_symbolic.py fit`: read the exported CSV, run PySR, and output the selected expression, energy predictions, and stress predictions for the experimental data used to train the network.
- `nn_to_symbolic.py pipeline`: run `export + fit` consecutively in one Python environment.
- `nn_to_symbolic.py predict-experiment`: read an existing expression, reconstruct stresses at the network-training experimental points, and output comparisons.

## Exporting energy data

With `torch` installed, run:

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python \
  /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/Symbolic_Regression/nn_to_symbolic.py \
  export \
  --lambda1-min 1.0 --lambda1-max 3.0 --lambda1-points 81 \
  --lambda2-min 1.0 --lambda2-max 3.0 --lambda2-points 81
```

Default outputs are `../output/SR_output/nn_energy_grid.csv` and `../output/SR_output/nn_energy_grid_metadata.json`.
By default, the undeformed energy $W(1,1)$ is subtracted so that the reference configuration has zero exported energy.

To sample UT/PS/ET paths directly:

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python \
  /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/Symbolic_Regression/nn_to_symbolic.py \
  export \
  --sampling-mode loadcases \
  --output-csv /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output/SR_output/nn_loadcase_samples.csv \
  --ut-stretch-range 1.0 8.0 --ut-points 80 \
  --ps-stretch-range 1.0 8.0 --ps-points 80 \
  --et-stretch-range 1.0 8.0 --et-points 80
```

All three stretch intervals can be supplied through the command line.

## Fitting with PySR

With `pysr` installed, run:

```bash
python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/Symbolic_Regression/nn_to_symbolic.py \
  fit \
  --input-csv /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output/SR_output/nn_energy_grid.csv \
  --feature-space invariants \
  --binary-operators + - '*' / \
  --stress-loss-weight 1.0 \
  --niterations 60 \
  --population-size 50 \
  --populations 10 \
  --maxsize 18 \
  --deterministic
```

The recommended starting point is `--feature-space invariants`, producing $W(I_1,I_2)$ in the same input space as the network and a natural form for a hyperelastic constitutive law.

Configure binary operators through `--binary-operators`. Experimental regression uses `+ - * /`, allowing rational terms. Division is numerically evaluated using PySR's protections at sampled search points, but the final expression still requires checks for denominator zeros and asymptotic behavior along loading paths.

When `--stress-loss-weight > 0`, the search uses a joint objective:

$$
\text{loss}=\text{MSE}_{energy}+w_{stress}\cdot\text{MSE}_{stress}.
$$

The energy term uses the exported neural-network energy grid. The stress term compares against stresses obtained by differentiating the network energy on the same `lambda1-lambda2` grid. The searched expression remains an energy function $W$.
With `--sampling-mode loadcases`, the stress term instead uses nominal-stress errors along UT/PS/ET paths. These target stresses are likewise network-derived rather than experimental stresses supplied directly to the search.

With `--feature-space invariants`, symbolic stresses are obtained from $\partial W/\partial I_1$ and $\partial W/\partial I_2$ using the chain rule.
With `--feature-space stretches`, PySR fits $W(\lambda_1,\lambda_2,\lambda_3)$, and stresses are reconstructed from the three principal-stretch derivatives along incompressible UT/PS/ET paths.
Adding `--valanis-landel` constrains the latter to $W=w(\lambda_1)+w(\lambda_2)+w(\lambda_3)$; this option currently requires both `--sampling-mode loadcases` and `--feature-space stretches`.

After fitting, the raw PySR expression is normalized at the reference state:

$$
W_{final}=W_{raw}-W_{raw}(I_1=3,I_2=3).
$$

Thus `best_equation.txt`, `predictions.csv`, `fit_summary.json`, and experimental-prediction files use the final zero-reference-energy expression rather than the uncorrected raw expression.

### Automatic post-fit simplification

`--simplify-expression` simplifies an invariant expression after PySR finishes. The program expands additive terms, attempts to remove one term at a time, and refits remaining coefficients by joint least squares using normalized network energy and experimental stress. A deletion is accepted only if the refitted model's selection-score increase relative to the original expression is no greater than `--simplify-rmse-tolerance`.

- `--simplify-rmse-tolerance`: allowed relative accuracy loss; default `0.02`.
- `--simplify-energy-weight`: network-energy weight during coefficient refitting; default `0.0`.
- `--simplify-stress-weight`: experimental-stress weight during coefficient refitting; default `1.0`.

Automatic pruning currently supports additive $W(I_1,I_2)$ expressions linear in their coefficients.
`raw_best_equation.txt` preserves the expression before simplification; `simplification_summary.json` records term contributions, deletions, and accuracy–complexity candidates. Once accepted, the simplified and refitted expression is used in `best_equation.txt`, predictions, and the material package.

### Output files

Default outputs:

- `../output/SR_output/best_equation.txt`
- `../output/SR_output/predictions.csv`
- `../output/SR_output/fit_summary.json`
- `../output/SR_output/energy_comparison_contours.png`
- `../output/SR_output/experiment_predictions.csv`
- `../output/SR_output/experiment_predictions.png`
- `../output/SR_output/experiment_prediction_summary.json`

`best_equation.txt` and `fit_summary.json` also record:

- `raw_initial_state_energy`: reference-state energy of the raw PySR expression.
- `initial_state_energy`: reference-state energy of the normalized expression, theoretically zero.
- `initial_state_energy_is_zero`: whether the final expression passes the zero-reference-energy check.
- `stress_loss_weight`: stress-loss weight used during search.
- `experiment_stress_loss`: overall experimental-stress MSE of the final symbolic expression.
- `experiment_stress_rmse`: overall experimental-stress RMSE of that expression.

`energy_comparison_contours.png` contains network-energy contours, symbolic-energy contours, and absolute-error contours.

Experimental-prediction files:

- `experiment_predictions.csv`: experimental stretches, measured stresses, symbolic predictions, and corresponding $I_1,I_2,\partial W/\partial I_1,\partial W/\partial I_2$, grouped by loading mode.
- `experiment_predictions.png`: measured stress points compared with symbolic stress curves.
- `experiment_prediction_summary.json`: RMSE and $R^2$ for each loading mode.

Final `rmse` and $R^2$ are recomputed using normalized `W_final`, so they can differ slightly from metrics for the uncorrected raw expression.

## Running the complete pipeline

If `torch` and `pysr` are installed in the same environment:

```bash
python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/Symbolic_Regression/nn_to_symbolic.py pipeline
```

Both `pipeline` and `fit` generate experimental-data predictions after symbolic fitting by default.
To regenerate those predictions from an existing expression:

```bash
python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/Symbolic_Regression/nn_to_symbolic.py \
  predict-experiment \
  --equation-file /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output/SR_output/best_equation.txt \
  --output-dir /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output/SR_output
```

## Environment notes

Historically, network training used `/home/guanjs/NN-constitutive/FFNN/.venv`, while the PySR environment could be found under `/home/guanjs/NN-constitutive/SymbolicRegression/PySR-master`.
If `torch` and `pysr` are in separate environments, run `export` and `fit` as two stages.
When `--stress-loss-weight > 0`, install `Zygote.jl` in the Julia environment: the search differentiates candidate energy expressions to construct the stress loss.
