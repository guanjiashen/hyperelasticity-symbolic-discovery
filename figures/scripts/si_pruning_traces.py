"""
SI figure: contribution-aware pruning traces for the six reported cases.

Reads data/fig3_experiments.json (Treloar, Bitoh, brain) and
data/fig2_benchmarks.json (Mooney-Rivlin, Yeoh, Arruda-Boyce) and draws a
2 x 3 panel figure of the refitted stress RMSE, as a percentage of the peak
stress of the data, against the number of retained terms.

    filled dot        accepted deletion (and the raw expression itself)
    ring              the selected law (last accepted state)
    grey hollow dot   the first rejected deletion
    dashed line       the acceptance bound (1 + tau) E_P^(0), tau = 0.02

Peak stresses: max |stress_target| in data/fig3_<name>_predictions.csv for
the measured materials, max |stress| of the "data" block of
fig2_benchmarks.json for the benchmarks.

Run from the SI directory:  python3 scripts/si_pruning_traces.py
Writes figs/si_pruning_traces.pdf and .png.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402

DATA = ROOT / "figs" / "data"
OUT = ROOT / "figs" / "si_pruning_traces"

# (json file, key, csv name or None, panel title, unit)
CASES = [
    ("fig3_experiments.json", "treloar", "treloar", "Treloar (vulcanized rubber)", "MPa"),
    ("fig3_experiments.json", "yohsuke", "yohsuke", "Bitoh et al. (polymer gel)", "kPa"),
    ("fig3_experiments.json", "brain", "brain", "Budday et al. (brain cortex)", "kPa"),
    ("fig2_benchmarks.json", "mooney_rivlin", None, "Mooney–Rivlin", "MPa"),
    ("fig2_benchmarks.json", "yeoh", None, "Yeoh", "MPa"),
    ("fig2_benchmarks.json", "arruda_boyce", None, "Arruda–Boyce", "MPa"),
]


def peak_stress(entry, csv_name):
    if csv_name is not None:
        with open(DATA / f"fig3_{csv_name}_predictions.csv") as f:
            rows = list(csv.DictReader(f))
        return max(abs(float(r["stress_target"])) for r in rows)
    return max(np.max(np.abs(np.asarray(arr["stress"], float)))
               for arr in entry["data"].values())


def main():
    cache = {}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))
    letters = "ABCDEF"
    for i, (fname, key, csv_name, title, unit) in enumerate(CASES):
        if fname not in cache:
            cache[fname] = json.load(open(DATA / fname))
        entry = cache[fname][key]
        peak = peak_stress(entry, csv_name)
        pareto = entry["pareto"]
        tau = float(entry.get("tolerance", 0.02))
        n = np.array([p["term_count"] for p in pareto])
        e = 100.0 * np.array([p["stress_rmse"] for p in pareto]) / peak
        acc = np.array([bool(p["accepted"]) for p in pareto])
        bound = (1.0 + tau) * e[0]

        ax = axes.flat[i]
        fs.style_axes(ax)
        ax.set_yscale("log")

        # connecting trace (accepted part solid, rejected step dotted)
        k_last = int(np.max(np.nonzero(acc)[0]))
        ax.plot(n[: k_last + 1], e[: k_last + 1], "-", color=fs.INK2, lw=0.8, zorder=2)
        if k_last + 1 < len(n):
            ax.plot(n[k_last: k_last + 2], e[k_last: k_last + 2], ":", color=fs.C_REJ,
                    lw=0.8, zorder=2)

        # acceptance bound
        ax.axhline(bound, ls="--", lw=0.7, color=fs.INK3, zorder=1)
        ax.text(n.min() - 0.42, bound, r"$1.02\,E_P^{(0)}$", fontsize=5.5,
                color=fs.INK2, va="top", ha="right")

        # accepted deletions (filled), first rejected (grey hollow)
        ax.plot(n[acc], e[acc], "o", ms=4.2, color=fs.C_ACCENT, mec=fs.C_ACCENT,
                zorder=4, label="accepted")
        rej = np.nonzero(~acc)[0]
        if len(rej):
            r = rej[0]
            ax.plot(n[r], e[r], "o", ms=4.2, mfc="white", mec=fs.C_REJ, mew=1.0,
                    zorder=4, label="rejected")
        # ring on the selected law
        ax.plot(n[k_last], e[k_last], "o", ms=8.5, mfc="none", mec=fs.C_ACCENT, mew=0.9,
                zorder=5, label="selected")

        # annotate the removed term of each step (short)
        for p, x, y in zip(pareto[1:], n[1:], e[1:]):
            t = p["removed_term"]
            if t is None:
                continue
            t = t.split("*", 1)[-1] if "*" in t else t
            t = (t.replace("I1", r"I_1").replace("I2", r"I_2")
                  .replace("**", "^").replace("*", r"\,"))
            ax.annotate(rf"$-\,{t}$", (x, y), textcoords="offset points",
                        xytext=(0, -9 if p["accepted"] else 7), fontsize=5.2,
                        color=fs.INK2, ha="center",
                        va="top" if p["accepted"] else "bottom")

        ax.set_title(rf"$\bf{{{letters[i]}}}$   {title}", fontsize=7, color=fs.INK,
                     pad=4, loc="left")
        ax.set_xticks(np.arange(n.min(), n.max() + 1))
        ax.set_xlim(n.min() - 0.5, n.max() + 0.5)
        ax.invert_xaxis()
        lo, hi = e.min(), e.max()
        ax.set_ylim(lo / 3.0, hi * 3.0)
        subs = (1.0,) if np.log10(hi / lo) > 1.5 else (1.0, 2.0, 5.0)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=subs))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        if i % 3 == 0:
            ax.set_ylabel("stress RMSE (% of peak)")
        if i >= 3:
            ax.set_xlabel("retained terms")

    h, l = axes.flat[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=6)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.92,
                        wspace=0.32, hspace=0.55)
    OUT.parent.mkdir(exist_ok=True)
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
