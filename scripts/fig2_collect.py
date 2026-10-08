"""
Collect the pipeline outputs used by Fig. 2 (benchmarks) into one JSON file.

Run from the NN_invariant directory of the pipeline repository, e.g.

    cd .../FFNN/hyperelasticity/NN_invariant
    ../../.venv/bin/python <paper>/scripts/fig2_collect.py <paper>/data/figure_data/fig2_benchmarks.json

For every benchmark it stores the noisy synthetic data actually used for pruning
(taken from SR_output/experiment_predictions.csv), the raw and final symbolic
expressions, and the full pruning trace (simplification_summary.json).
"""
import csv
import json
import sys
from pathlib import Path

CASES = [
    dict(key="mooney_rivlin", run="output/noise_study_20260801/noise_0", sigma=0.0),
    dict(key="yeoh", run="output/benchmarks/yeoh_h6_noise_0", sigma=0.0),
    dict(key="arruda_boyce", run="output/benchmarks/arruda_boyce_mu1_lm5_et5", sigma=0.0),
    dict(key="ogden", run="output/benchmarks/ogden_noise_0", sigma=0.0),
]


def read_data(csv_path):
    rows = {"UT": [], "PS": [], "ET": []}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["mode"]].append((float(r["stretch"]), float(r["stress_target"])))
    out = {}
    for mode, pts in rows.items():
        # the first entry of every path is the stress-free reference anchor
        # (stretch 1, stress exactly 0) appended by the pipeline, not a data point
        if pts and pts[0] == (1.0, 0.0):
            pts = pts[1:]
        out[mode] = dict(stretch=[p[0] for p in pts], stress=[p[1] for p in pts])
    return out


def main(dst):
    result = {}
    for c in CASES:
        run = Path(c["run"])
        simp = json.loads((run / "SR_output" / "simplification_summary.json").read_text())
        result[c["key"]] = dict(
            run=str(run),
            sigma=c["sigma"],
            data=read_data(run / "SR_output" / "experiment_predictions.csv"),
            raw_expression=simp["original_expression"],
            final_expression=simp["simplified_expression"],
            criterion=simp["criterion"],
            tolerance=simp["rmse_tolerance"],
            pareto=[{k: p.get(k) for k in ("term_count", "expression", "removed_term",
                                           "stress_rmse", "selection_score", "accepted")}
                    for p in simp["pareto"]],
        )
        print(c["key"], len(simp["pareto"]), "pruning steps ->", simp["simplified_expression"])
    Path(dst).write_text(json.dumps(result, indent=1))
    print("written", dst)


if __name__ == "__main__":
    main(sys.argv[1])
