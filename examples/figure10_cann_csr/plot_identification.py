"""Plot the CANN-CSR identification for the heterogeneous specimen (Sec. 4.4).

Panels: UT / PS / ET nominal stress versus stretch.
  - open markers : digitized experimental data (figure15_data)
  - blue solid   : CANN (8-neuron Softplus) evaluated by the deployment wrapper
                   along the three loading paths (nn_loadcase_samples.csv)
  - red dashed   : pruned symbolic law from material_package.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "nn_cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["UT", "PS", "ET"]
TITLES = {"UT": "Uniaxial tension", "PS": "Pure shear", "ET": "Equibiaxial tension"}
MARKERS = {"UT": "o", "PS": "D", "ET": "^"}
MODE_ID = {1: "UT", 2: "PS", 3: "ET"}


def kinematics(mode: str, lam: np.ndarray):
    if mode == "UT":
        l1, l2, l3 = lam, lam ** -0.5, lam ** -0.5
    elif mode == "PS":
        l1, l2, l3 = lam, np.ones_like(lam), 1.0 / lam
    elif mode == "ET":
        l1, l2, l3 = lam, lam, lam ** -2.0
    else:
        raise ValueError(mode)
    return l1, l2, l3


def nominal_stress(mode, lam, dW1, dW2):
    l1, l2, l3 = kinematics(mode, lam)
    I1 = l1 ** 2 + l2 ** 2 + l3 ** 2
    I2 = (l1 * l2) ** 2 + (l2 * l3) ** 2 + (l3 * l1) ** 2
    w1 = dW1(I1, I2)
    w2 = dW2(I1, I2)
    return 2.0 * (l1 - l3 ** 2 / l1) * (w1 + l2 ** 2 * w2)


def make_eval(expr: str):
    code = compile(expr, "<expr>", "eval")

    def f(I1, I2):
        I1 = np.asarray(I1, dtype=float)
        I2 = np.asarray(I2, dtype=float)
        out = eval(code, {"__builtins__": {}}, {"I1": I1, "I2": I2, "np": np})
        return np.broadcast_to(np.asarray(out, dtype=float), I1.shape)

    return f


def load_data(path: Path):
    arr = np.loadtxt(path)
    stress, stretch = arr[:, 0], arr[:, 1]
    return np.concatenate([[1.0], stretch]), np.concatenate([[0.0], stress])


def load_nn_samples(path: Path):
    out = {m: ([], []) for m in MODES}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            m = MODE_ID[int(float(row["mode_id"]))]
            out[m][0].append(float(row["stretch"]))
            out[m][1].append(float(row["nominal_stress"]))
    return {m: (np.array([1.0] + xs), np.array([0.0] + ys)) for m, (xs, ys) in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--output", default="../../figs/heterogeneous_cann_csr_fit.png")
    args = ap.parse_args()
    case = Path(args.case)
    sr = case / "nn_cache" / "output" / "SR_output"

    package = json.loads((sr / "material_package.json").read_text())
    dW1 = make_eval(package["constitutive"]["dW_dI1"])
    dW2 = make_eval(package["constitutive"]["dW_dI2"])
    summary = json.loads((sr / "experiment_prediction_summary.json").read_text())
    nn = load_nn_samples(sr / "nn_loadcase_samples.csv")

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for ax, mode in zip(axes, MODES):
        lam_d, p_d = load_data(case.parents[1] / "data" / "experimental" / "Meunier_2008" / mode / "stress_stretch.txt")
        lam_nn, p_nn = nn[mode]
        lam_f = np.linspace(1.0, lam_d.max(), 200)
        p_sr = nominal_stress(mode, lam_f, dW1, dW2)
        ax.plot(lam_nn, p_nn, "-", color="#1f77e0", lw=2.2, label="CANN")
        ax.plot(lam_f, p_sr, "--", color="#d62728", lw=2.0, label="CSR")
        ax.plot(lam_d, p_d, MARKERS[mode], mfc="white", mec="black", ms=6.5, mew=1.1, ls="none", label="Experiment")
        ax.set_title(TITLES[mode])
        ax.set_xlabel("Stretch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$P_{11}$ (MPa)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [2, 0, 1]
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", frameon=False)
    fig.tight_layout()
    out = (case / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")

    # Console summary useful for the manuscript.
    print("expression:", package["constitutive"]["isochoric_energy"])
    for m in MODES + ["overall"]:
        s = summary[m]
        print(f"{m:8s} stress MSE = {s['stress_loss']:.4e}  RMSE = {s['rmse']:.4e}")


if __name__ == "__main__":
    main()
