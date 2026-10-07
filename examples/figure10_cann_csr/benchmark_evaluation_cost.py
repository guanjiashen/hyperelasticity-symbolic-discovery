#!/usr/bin/env python3
"""Benchmark the constitutive-evaluation cost of the trained CANN versus the CSR law.

Both models return the energy derivatives (dW/dI1, dW/dI2) required by the
material routine.  Timings are wall-clock, single-threaded, float64.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

CASE = Path(__file__).resolve().parent
class FFBPNetworkInvariant(torch.nn.Module):
    def __init__(self, hidden_layers, activation):
        super().__init__()
        self.hidden = torch.nn.Linear(2, hidden_layers[0])
        self.output = torch.nn.Linear(hidden_layers[0], 1)
    def forward(self, x):
        return self.output(torch.nn.functional.softplus(self.hidden(x)))

torch.set_num_threads(1)
N_BATCH = 1_000_000
N_SCALAR = 1_000_000  # same sample count as the vectorized mode
COARSE_EVALS = 2962 * 1_717_697
FINE_EVALS = 7706 * 2_257_221

# ---------------------------------------------------------------- models
state = torch.load(CASE / "nn_cache/output/NN_output/ffbp_model.pt", map_location="cpu")
model = FFBPNetworkInvariant(hidden_layers=[int(state["hidden.weight"].shape[0])], activation="softplus")
model.load_state_dict(state); model.double(); model.eval()
Wh = state["hidden.weight"].double().numpy(); bh = state["hidden.bias"].double().numpy()
Wo = state["output.weight"].double().numpy()[0]
n_params = sum(p.numel() for p in model.parameters())

pkg = json.loads((CASE / "nn_cache/output/SR_output/material_package.json").read_text())
expr1, expr2 = pkg["constitutive"]["dW_dI1"], pkg["constitutive"]["dW_dI2"]
print("dW/dI1 =", expr1); print("dW/dI2 =", expr2)
sr_vec1 = eval("lambda I1, I2: " + expr1, {"np": np, "sqrt": np.sqrt, "exp": np.exp, "log": np.log})
sr_vec2 = eval("lambda I1, I2: " + expr2, {"np": np, "sqrt": np.sqrt, "exp": np.exp, "log": np.log})
sr_sca1 = eval("lambda I1, I2: " + expr1, {"sqrt": math.sqrt, "exp": math.exp, "log": math.log})
sr_sca2 = eval("lambda I1, I2: " + expr2, {"sqrt": math.sqrt, "exp": math.exp, "log": math.log})

# ---------------------------------------------------------------- samples
rng = np.random.default_rng(0)
# Sample incompressible biaxial states within both exported invariant ranges.
flat = (CASE / "nn_cache/material.flat").read_text().splitlines()
lo1, hi1 = map(float, flat[11].split())
lo2, hi2 = map(float, flat[12].split())
accepted = []
while sum(len(a) for a in accepted) < N_BATCH:
    a = rng.uniform(1.0, 2.17, N_BATCH)
    b = rng.uniform(1.0 / np.sqrt(a), a)
    c = 1.0/(a*b)
    q1 = a*a+b*b+c*c
    q2 = (a*b)**2+(b*c)**2+(c*a)**2
    mask = (q1 >= lo1)&(q1 <= hi1)&(q2 >= lo2)&(q2 <= hi2)
    accepted.append(np.column_stack([a[mask],b[mask]]))
l1, l2 = np.concatenate(accepted)[:N_BATCH].T
l3 = 1.0 / (l1 * l2)
I1 = l1**2 + l2**2 + l3**2; I2 = (l1*l2)**2 + (l2*l3)**2 + (l3*l1)**2
inv_np = np.column_stack((I1, I2)); inv_t = torch.from_numpy(inv_np)

# ---------------------------------------------------------------- NN evaluators
def nn_autograd(x):
    x = x.detach().requires_grad_(True)
    return torch.autograd.grad(model(x).sum(), x)[0]

def nn_numpy(I1, I2):
    z = np.column_stack((I1, I2)) @ Wh.T + bh
    sig = 1.0 / (1.0 + np.exp(-z))          # softplus'(z)
    g = sig * Wo                             # n x h
    return g @ Wh[:, 0], g @ Wh[:, 1]

Wh_l = Wh.tolist(); bh_l = bh.tolist(); Wo_l = Wo.tolist(); H = len(bh_l)
def nn_scalar(i1, i2):
    w1 = 0.0; w2 = 0.0
    for k in range(H):
        z = Wh_l[k][0]*i1 + Wh_l[k][1]*i2 + bh_l[k]
        s = Wo_l[k] / (1.0 + math.exp(-z))
        w1 += s * Wh_l[k][0]; w2 += s * Wh_l[k][1]
    return w1, w2

# ---------------------------------------------------------------- consistency check
ref = nn_autograd(inv_t[:1000]).numpy()
a, b = nn_numpy(I1[:1000], I2[:1000])
assert np.allclose(ref[:, 0], a) and np.allclose(ref[:, 1], b)
s = np.array([nn_scalar(float(x), float(y)) for x, y in inv_np[:1000]])
assert np.allclose(ref, s)
sa, sb = sr_vec1(I1[:1000], I2[:1000]), sr_vec2(I1[:1000], I2[:1000])
ss = np.array([(sr_sca1(float(x), float(y)), sr_sca2(float(x), float(y))) for x, y in inv_np[:1000]])
assert np.allclose(sa, ss[:, 0]) and np.allclose(np.broadcast_to(sb, sa.shape), ss[:, 1])

def timeit(fn, repeats=3, warm=1):
    for _ in range(warm): fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

res = {}
# vectorized (batch of N_BATCH)
res["nn_autograd_batch"] = timeit(lambda: nn_autograd(inv_t)) / N_BATCH
res["nn_numpy_batch"] = timeit(lambda: nn_numpy(I1, I2)) / N_BATCH
res["sr_numpy_batch"] = timeit(lambda: (sr_vec1(I1, I2), np.broadcast_to(sr_vec2(I1, I2), I1.shape))) / N_BATCH
# scalar, one material point per call
sub = inv_np[:N_SCALAR].tolist(); sub_t = inv_t[:N_SCALAR]

# One call per material point; the results are computed and dropped, so that the
# three evaluators share the same loop harness and no result storage is timed.
def loop_nn_autograd():
    for i in range(N_SCALAR):
        nn_autograd(sub_t[i:i+1])

def loop_nn_scalar():
    for x, y in sub:
        nn_scalar(x, y)

def loop_sr_scalar():
    for x, y in sub:
        sr_sca1(x, y)
        sr_sca2(x, y)

res["nn_autograd_pointwise"] = timeit(loop_nn_autograd, repeats=3, warm=1) / N_SCALAR
res["nn_scalar_pointwise"] = timeit(loop_nn_scalar, repeats=3, warm=1) / N_SCALAR
res["sr_scalar_pointwise"] = timeit(loop_sr_scalar, repeats=3, warm=1) / N_SCALAR

out = {
    "case": "Meunier, main-text Figure 4",
    "cpu": __import__("platform").processor(),
    "platform": __import__("platform").platform(),
    "dtype": "float64", "threads": 1,
    "invariant_bounds": {"I1": [lo1,hi1], "I2": [lo2,hi2]},
    "torch_version": torch.__version__, "numpy_version": np.__version__,
    "nn": {"hidden_neurons": H, "activation": "softplus", "parameters": n_params},
    "sr": {"dW_dI1": expr1, "dW_dI2": expr2},
    "batch_size": N_BATCH, "pointwise_calls": N_SCALAR,
    "seconds_per_evaluation": res,
    "nanoseconds_per_evaluation": {k: 1e9 * v for k, v in res.items()},
    "speedup": {
        "batch_numpy_nn_over_sr": res["nn_numpy_batch"] / res["sr_numpy_batch"],
        "batch_autograd_nn_over_sr": res["nn_autograd_batch"] / res["sr_numpy_batch"],
        "pointwise_scalar_nn_over_sr": res["nn_scalar_pointwise"] / res["sr_scalar_pointwise"],
        "pointwise_autograd_nn_over_sr": res["nn_autograd_pointwise"] / res["sr_scalar_pointwise"],
    },
    "element_evaluations": {"coarse": COARSE_EVALS, "fine": FINE_EVALS},
    "projected_seconds_pointwise_scalar": {
        "coarse_nn": COARSE_EVALS * res["nn_scalar_pointwise"], "coarse_sr": COARSE_EVALS * res["sr_scalar_pointwise"],
        "fine_nn": FINE_EVALS * res["nn_scalar_pointwise"], "fine_sr": FINE_EVALS * res["sr_scalar_pointwise"],
    },
}
(CASE / "evaluation_cost_meunier.json").write_text(json.dumps(out, indent=2) + "\n")
for k, v in res.items(): print(f"{k:26s} {1e9*v:10.1f} ns/eval")
for k, v in out["speedup"].items(): print(f"{k:32s} {v:8.2f}x")
print("element evaluations coarse/fine:", COARSE_EVALS, FINE_EVALS)