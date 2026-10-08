# NN_invariant user guide

This directory trains invariant-based hyperelastic neural networks using:

- Experimental UT/PS/ET data.
- Optional biaxial data (BS or BL).
- Optional synthetic data from Ogden, Arruda–Boyce, or Mooney–Rivlin models.

The commands and defaults below retain paths from the original development environment. Adapt these paths to your installation; see the repository-level README for the release layout and portability status.

## 1. Running the program

The program can be run from the repository root or this directory. The original project virtual environment was used as follows:

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py [CLI_ARGUMENTS]
```

The simplest invocation uses Treloar UT/PS/ET data and stress-mode training:

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py
```

Default output directories:

- `output/NN_output`: trained network outputs, including `ffbp_model.pt`, `ffbp_weights.txt`, stress predictions, energy contours, and loss curves.
- `output/SR_output`: symbolic-regression outputs, including exported energy CSV files, PySR fits, and energy-comparison contours.

## 2. Command-line arguments

Arguments are defined in `cli/cli.py`.

### Training objective

`--train-mode {energy,stress}` selects the training target. Default: `stress`.

### Network architecture

- `--hidden-neurons N`: single-hidden-layer width; default `3`. Retained for compatibility with earlier training commands and models.
- `--hidden-layers N1 N2 ...`: one or more hidden-layer widths; overrides `--hidden-neurons`. For example, `--hidden-layers 16 16 8` gives `2 → 16 → 16 → 8 → 1`.

Example with multiple hidden layers:

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python \
  /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py \
  --hidden-layers 16 16 8 \
  --activation softplus
```

### Synthetic data

| Argument | Meaning | Default |
| --- | --- | --- |
| `--use-synthetic-data` | Generate data from an analytical model instead of reading experimental paths | Disabled unless supplied |
| `--synthetic-model {ogden,arruda_boyce,mooney_rivlin}` | Analytical model | `ogden` |
| `--mooney-rivlin-c10 C10` | Mooney–Rivlin C10 when that model is selected | `0.18` |
| `--mooney-rivlin-c01 C01` | Mooney–Rivlin C01 when that model is selected | `0.02` |
| `--synthetic-point-count N` | Number of samples per loading mode | `30` |
| `--synthetic-noise-std STD` | Standard deviation of Gaussian noise added to synthetic stresses | `0.0` |
| `--synthetic-ut-range MIN MAX` | UT stretch interval | `1.0 3.0` |
| `--synthetic-ps-range MIN MAX` | PS stretch interval | `1.0 3.0` |
| `--synthetic-et-range MIN MAX` | ET stretch interval | `1.0 3.0` |

### Experimental data paths (UT/PS/ET)

Use `--ut-dataset PATH`, `--ps-dataset PATH`, and `--et-dataset PATH`. Their historical defaults are:

- `fitting-data-PK/Treloar_1944/UT/stress_stretch.txt`
- `fitting-data-PK/Treloar_1944/PS/stress_stretch.txt`
- `fitting-data-PK/Treloar_1944/ET/stress_stretch.txt`

### Biaxial data

- `--enable-biaxial`: enable biaxial data loading; disabled unless supplied.
- `--biaxial-prefix {BS,BL}`: loading-mode prefix; default `BS`. BS usually corresponds to `BT_small`, and BL to `BT_large`.
- `--biaxial-base-dir PATH`: root directory containing `B1`, `B2`, etc., each with `stress_stretch.txt`. Default: `fitting-data-PK/Kawabata_1981/BT_small`.

The number of biaxial datasets is not fixed: the loader scans all matching `B*` subdirectories.

## 3. Example commands

### Example 0: training followed by symbolic regression

Training and symbolic regression are handled by separate scripts. The default script workflow uses synthetic data:

```bash
bash /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/run_train.sh
bash /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/run_symbolic.sh
```

The defaults correspond to `main.py --train-mode stress --use-synthetic-data --synthetic-model mooney_rivlin` for training, and `nn_to_symbolic.py pipeline` for symbolic regression.
`run_train.sh` generates Mooney–Rivlin UT/PS/ET data and trains NN_invariant. `run_symbolic.sh` performs symbolic regression on the saved network and runs the FEM element test by default. Both scripts must select the same workflow.

Switch to experimental data explicitly:

```bash
bash /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/run_train.sh --experimental
bash /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/run_symbolic.sh --experimental
```

Available script switches:

- `--synthetic`: select the default synthetic workflow.
- `--experimental`: select the Treloar UT/PS/ET experimental workflow.
- `--run-fem`: run the final FEM element test (`run_symbolic.sh` only; default).
- `--skip-fem`: skip that test (`run_symbolic.sh` only).

Edit `train_args=(...)` in `run_train.sh` and `symbolic_args=(...)` in `run_symbolic.sh` to change the respective settings. For example:

```bash
train_args=(
  --train-mode stress
)

symbolic_args=(
  pipeline
  --lambda1-points 41
  --lambda2-points 41
  --niterations 20
  --deterministic
)
```

### Example A: default experimental stress training

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress
```

### Example B: small-deformation biaxial data (BT_small → BS)

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress --enable-biaxial --biaxial-prefix BS --biaxial-base-dir /home/guanjs/NN-constitutive/FFNN/fitting-data-PK/Kawabata_1981/BT_small
```

### Example C: large-deformation biaxial data (BT_large → BL)

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress --enable-biaxial --biaxial-prefix BL --biaxial-base-dir /home/guanjs/NN-constitutive/FFNN/fitting-data-PK/Kawabata_1981/BT_large
```

### Example D: synthetic Ogden data

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress --use-synthetic-data --synthetic-model ogden --synthetic-point-count 40 --synthetic-ut-range 1.0 2.5 --synthetic-ps-range 1.0 2.5 --synthetic-et-range 1.0 2.5
```

### Example E: synthetic Mooney–Rivlin data

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress --use-synthetic-data --synthetic-model mooney_rivlin --mooney-rivlin-c10 0.18 --mooney-rivlin-c01 0.02 --synthetic-point-count 40 --synthetic-ut-range 1.0 3.0 --synthetic-ps-range 1.0 3.0 --synthetic-et-range 1.0 3.0
```

To add controlled Gaussian noise with standard deviation `0.02`:

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress --use-synthetic-data --synthetic-model mooney_rivlin --mooney-rivlin-c10 0.18 --mooney-rivlin-c01 0.02 --synthetic-point-count 40 --synthetic-noise-std 0.02 --synthetic-ut-range 1.0 3.0 --synthetic-ps-range 1.0 3.0 --synthetic-et-range 1.0 3.0
```

## 4. Troubleshooting

If a data file cannot be found, check `--ut-dataset`, `--ps-dataset`, and `--et-dataset`. For biaxial data, verify that `--biaxial-base-dir` contains `B*` subdirectories with `stress_stretch.txt` in each.

To inspect available options, run:

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --help
```
