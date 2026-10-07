"""
Collect the pipeline outputs used by Fig. 3 (measured materials) into one JSON
file: raw and final symbolic expressions plus the full pruning trace of every
material (simplification_summary.json). The measured/predicted stresses are
kept separately in figs/data/fig3_*_predictions.csv.

    python fig3_collect.py <paper>/figs/data/fig3_experiments.json
"""
import json
import sys
from pathlib import Path

BASE = Path("/home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output")
CASES = [
    dict(key="treloar", run=BASE / "experimental_tuned/Treloar_1944/SR_output_simple_16"),
    dict(key="yohsuke", run=BASE / "experimental_tuned/Yohsuke_2011/SR_output_simple_12"),
    dict(key="brain", run=BASE / "SR_output-brainCX"),
]


def main(dst):
    result = {}
    for c in CASES:
        simp = json.loads((c["run"] / "simplification_summary.json").read_text())
        result[c["key"]] = dict(
            run=str(c["run"]),
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
