#!/usr/bin/env python3
"""Median of repeated runs of benchmark_evaluation_cost.py.

The Python timings fluctuate by some tens of percent between processes, so the
figures quoted in the paper are medians over three independent runs of the
benchmark; this script merges the per-run JSON files into evaluation_cost.json.
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

CASE = Path(__file__).resolve().parent
paths = [Path(p) for p in sys.argv[1:]] or sorted(CASE.glob("evaluation_cost_run*.json"))
runs = [json.loads(p.read_text()) for p in paths]
keys = list(runs[0]["seconds_per_evaluation"])

median = {k: statistics.median(r["seconds_per_evaluation"][k] for r in runs) for k in keys}
out = dict(runs[-1])
out["runs"] = len(runs)
out["aggregation"] = "median over independent runs of the benchmark"
out["seconds_per_evaluation"] = median
out["nanoseconds_per_evaluation"] = {k: 1e9 * v for k, v in median.items()}
out["per_run_nanoseconds_per_evaluation"] = {
    k: [1e9 * r["seconds_per_evaluation"][k] for r in runs] for k in keys
}
out["speedup"] = {
    "batch_numpy_nn_over_sr": median["nn_numpy_batch"] / median["sr_numpy_batch"],
    "batch_autograd_nn_over_sr": median["nn_autograd_batch"] / median["sr_numpy_batch"],
    "pointwise_scalar_nn_over_sr": median["nn_scalar_pointwise"] / median["sr_scalar_pointwise"],
    "pointwise_autograd_nn_over_sr": median["nn_autograd_pointwise"] / median["sr_scalar_pointwise"],
}
ev = out["element_evaluations"]
out["projected_seconds_pointwise"] = {
    "coarse_nn_autograd": ev["coarse"] * median["nn_autograd_pointwise"],
    "coarse_sr": ev["coarse"] * median["sr_scalar_pointwise"],
    "fine_nn_autograd": ev["fine"] * median["nn_autograd_pointwise"],
    "fine_sr": ev["fine"] * median["sr_scalar_pointwise"],
}
(CASE / "evaluation_cost.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in ("runs", "nanoseconds_per_evaluation",
                                      "per_run_nanoseconds_per_evaluation", "speedup",
                                      "projected_seconds_pointwise")}, indent=2))
