"""
Collect CANN training-convergence histories for the Fig. 2 benchmarks.

Run from the NN_invariant directory of the pipeline repository, e.g.

    cd .../FFNN/hyperelasticity/NN_invariant
    ../../.venv/bin/python <paper>/scripts/si_collect_convergence.py \
        <paper>/data/figure_data/si_cann_convergence.json

For every benchmark the script re-runs the CANN training of the pipeline
(main.py) with the published settings and captures the train/validation loss
per L-BFGS outer iteration by intercepting the loss_plot call. The Yeoh and
Arruda-Boyce cases repeat the twelve-restart protocol (init seeds 0-11).
Training data are read from the archived synthetic_data of the published runs
(Yeoh, Arruda-Boyce) or regenerated noise-free (Mooney-Rivlin).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import main as pipeline  # noqa: E402  (the pipeline entry point)

COMMON = [
    "--train-mode", "stress", "--activation", "softplus", "--optimizer", "lbfgs",
    "--learning-rate", "0.2", "--epochs", "200", "--print-every", "200",
    "--lbfgs-max-iter", "30", "--lbfgs-history-size", "50", "--random-seed", "42",
]

def data_args(run):
    base = Path("output") / run / "synthetic_data"
    return ["--ut-dataset", str(base / "UT" / "stress_stretch.txt"),
            "--ps-dataset", str(base / "PS" / "stress_stretch.txt"),
            "--et-dataset", str(base / "ET" / "stress_stretch.txt")]

CASES = [
    dict(key="mooney_rivlin", hidden=3, seeds=[None],
         args=["--use-synthetic-data", "--synthetic-model", "mooney_rivlin",
               "--mooney-rivlin-c10", "0.18", "--mooney-rivlin-c01", "0.02",
               "--synthetic-point-count", "40", "--synthetic-noise-std", "0.0",
               "--synthetic-ut-range", "1.0", "3.0",
               "--synthetic-ps-range", "1.0", "3.0",
               "--synthetic-et-range", "1.0", "3.0"]),
    dict(key="yeoh", hidden=6, seeds=list(range(12)),
         args=data_args("benchmarks/yeoh_h6_noise_0")),
    dict(key="arruda_boyce", hidden=3, seeds=list(range(12)),
         args=data_args("benchmarks/arruda_boyce_mu1_lm5_et5")),
]

captured = {}

def capture_loss_plot(base_dir, train_losses=None, val_losses=None, **kw):
    captured["train"] = [float(v) for v in train_losses]
    captured["val"] = [float(v) for v in val_losses]

pipeline.loss_plot = capture_loss_plot
pipeline.stress_strain_plot = lambda *a, **k: None
pipeline.energy_contour_plot = lambda *a, **k: None
pipeline.save_weights = lambda *a, **k: None


def main(dst):
    result = {}
    for c in CASES:
        runs = {}
        for seed in c["seeds"]:
            argv = ["main.py", "--hidden-neurons", str(c["hidden"])] + COMMON + c["args"]
            if seed is not None:
                argv += ["--init-seed", str(seed)]
            sys.argv = argv
            captured.clear()
            pipeline.main()
            label = "42" if seed is None else str(seed)
            runs[label] = dict(train=captured["train"], val=captured["val"])
            print(f'{c["key"]} seed {label}: final train {captured["train"][-1]:.6f} '
                  f'val {captured["val"][-1]:.6f}')
        final = {s: r["train"][-1] for s, r in runs.items()}
        selected = min(final, key=final.get)
        result[c["key"]] = dict(hidden_neurons=c["hidden"], selected_seed=selected, runs=runs)
        print(f'{c["key"]}: selected seed {selected} (lowest final training loss)')
    Path(dst).write_text(json.dumps(result))
    print("written", dst)


if __name__ == "__main__":
    main(sys.argv[1])