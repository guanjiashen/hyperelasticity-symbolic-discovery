"""
Fig. 2 of the PNAS manuscript: known physics is recovered, or compressed, from noise-free
synthetic data. Three columns, one per classical model, ordered from exactly
representable in (I1, I2) polynomials (Mooney-Rivlin, Yeoh) to not representable
(Arruda-Boyce).
  top    : verdict chip, ground-truth energy, discovered energy
  middle : nominal stress on UT / PS / ET; synthetic data (hollow markers),
           discovered law (thick line), ground truth (thin dashed)
           below each stress panel a residual strip, (P_disc - P_truth)/P_peak in %
  bottom : deletion ledger, the pruning test of Eq. (2); stress RMSE after
           deleting each term of the raw expression and refitting the rest
           (% of peak stress), with the admissible budget (1+tau)E_P^(0) as a
           dashed line; retained terms filled in the chip colour, pruned grey

Input : data/figure_data/fig2_benchmarks.json   (written by scripts/fig2_collect.py)
        data/figure_data/fig2_prune_ledger.json (written by scripts/fig2_prune_collect.py
        against the pipeline runs; values verified against the recorded trace)
Output: figs/fig2_benchmarks.pdf (+ .png preview)

    python scripts/fig2_benchmarks.py
"""
from pathlib import Path
import json
import sys

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
from figstyle import INK, INK2, INK3, GRID, PATHS, C_REC, C_CMP, C_PRUNED, fnum

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figure_data" / "fig2_benchmarks.json"
PRUNE = ROOT / "data" / "figure_data" / "fig2_prune_ledger.json"
OUT = ROOT / "figs" / "fig2_benchmarks"

# ------------------------------------------------------------------ kinematics
LAM2 = {"UT": lambda l: l ** -0.5, "PS": lambda l: np.ones_like(l), "ET": lambda l: l}


def kin(mode, l1):
    l2 = LAM2[mode](l1)
    l3 = 1.0 / (l1 * l2)
    I1 = l1 ** 2 + l2 ** 2 + l3 ** 2
    I2 = (l1 * l2) ** 2 + (l2 * l3) ** 2 + (l3 * l1) ** 2
    return l2, l3, I1, I2


def P11(mode, l1, W1, W2):
    """Nominal stress of an incompressible solid from dW/dI1 and dW/dI2."""
    l2 = LAM2[mode](l1)
    return 2.0 * (l1 - l1 ** -3 * l2 ** -2) * (W1 + l2 ** 2 * W2)


# ------------------------------------------------------------------ ground truths
def truth_mr(mode, l, C10=0.18, C01=0.02):
    return P11(mode, l, C10 + 0 * l, C01 + 0 * l)


def truth_yeoh(mode, l, C=(0.17, -1.5e-3, 2.0e-5)):
    x = kin(mode, l)[2] - 3.0
    return P11(mode, l, C[0] + 2 * C[1] * x + 3 * C[2] * x ** 2, 0 * l)


AB_C = (1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750)


def truth_ab(mode, l, mu=1.0, lm=5.0):
    I1 = kin(mode, l)[2]
    W1 = mu * sum((i + 1) * c * I1 ** i / lm ** (2 * i) for i, c in enumerate(AB_C))
    return P11(mode, l, W1, 0 * l)


def truth_ogden(mode, l, mu=(0.63, 0.0012, -0.01), al=(1.3, 5.0, -2.0)):
    l2, l3 = kin(mode, l)[:2]
    return sum(m * (l ** a - l3 ** a) for m, a in zip(mu, al)) / l


# ------------------------------------------------------------------ symbolic laws
I1s, I2s, xs, ys = sp.symbols("I1 I2 x y")


def law_stress(expr_str):
    W = sp.sympify(expr_str, locals={"I1": I1s, "I2": I2s})
    f1 = sp.lambdify((I1s, I2s), sp.diff(W, I1s), "numpy")
    f2 = sp.lambdify((I1s, I2s), sp.diff(W, I2s), "numpy")

    def stress(mode, l):
        _, _, I1, I2 = kin(mode, l)
        return P11(mode, l, f1(I1, I2) + 0 * l, f2(I1, I2) + 0 * l)
    return stress


def n_terms(expr_str):
    """Number of non-constant monomials of an invariant polynomial."""
    p = sp.Poly(sp.expand(sp.sympify(expr_str, locals={"I1": I1s, "I2": I2s})), I1s, I2s)
    return sum(1 for m in p.monoms() if any(m))


def mono(sym, k):
    return sym if k == 1 else f"{sym}^{{{k}}}"


def fmt_poly(terms):
    """terms: list of (coef, latex_monomial) -> list of signed latex pieces."""
    out = []
    for i, (c, m) in enumerate(terms):
        sgn = ("-" if c < 0 else "") if i == 0 else ("-" if c < 0 else "+")
        out.append(f"{sgn}{fnum(c)}\\,{m}")
    return out


def shifted_basis(expr_str):
    """Rewrite in powers of (I1 - 3) and (I2 - 3) (exact for polynomials; constant drops)."""
    W = sp.sympify(expr_str, locals={"I1": I1s, "I2": I2s}).subs({I1s: xs + 3, I2s: ys + 3})
    p = sp.Poly(sp.expand(W), xs, ys)
    terms = []
    for (a, b), c in sorted(p.terms(), key=lambda t: (sum(t[0]), -t[0][0])):
        if (a == 0 and b == 0) or abs(float(c)) < 1e-14:
            continue
        m = " ".join(filter(None, [mono(r"(I_1{-}3)", a) if a else "",
                                   mono(r"(I_2{-}3)", b) if b else ""]))
        terms.append((float(c), m))
    return terms


def referenced_basis(expr_str):
    """Keep monomials of I1, I2 and write each as (m - m|_ref), which absorbs the constant."""
    p = sp.Poly(sp.expand(sp.sympify(expr_str, locals={"I1": I1s, "I2": I2s})), I1s, I2s)
    terms = []
    for (a, b), c in sorted(p.terms(), key=lambda t: (t[0][1], t[0][0])):
        if a == 0 and b == 0:
            continue
        m = " ".join(filter(None, [mono("I_1", a) if a else "", mono("I_2", b) if b else ""]))
        terms.append((float(c), rf"({m}-{3 ** (a + b)})"))
    return terms


# ------------------------------------------------------------------ cases
CASES = [
    dict(key="mooney_rivlin", name="Mooney–Rivlin", kind="recovered", n_truth=2,
         truth=truth_mr, basis=shifted_basis,
         truth_tex=[(r"$\Psi=0.18\,(I_1{-}3)+0.02\,(I_2{-}3)$", 0.14)],
         note=[]),
    dict(key="yeoh", name="Yeoh", kind="recovered", n_truth=3,
         truth=truth_yeoh, basis=shifted_basis,
         truth_tex=[(r"$\Psi=0.17\,(I_1{-}3)-1.5{\times}10^{-3}\,(I_1{-}3)^{2}$", 0.14),
                    (r"$\quad\ \ +2.0{\times}10^{-5}\,(I_1{-}3)^{3}$", 0.14)],
         note=[]),
    dict(key="arruda_boyce", name="Arruda–Boyce", kind="compressed", n_truth=5,
         truth=truth_ab, basis=referenced_basis,
         truth_tex=[(r"$\Psi=\mu\,\sum_{i=1}^{5}C_i\,\lambda_m^{2-2i}\,(I_1^{\,i}-3^{i})$", 0.22)],
         note=[r"$\mu=1$ MPa, $\lambda_m=5$",
               r"$C_i=5.0000{\times}10^{-1},\ 5.0000{\times}10^{-2},$",
               r"$\qquad\ 1.0476{\times}10^{-2},\ 2.7143{\times}10^{-3},$",
               r"$\qquad\ 7.7032{\times}10^{-4}$"]),
]


# ------------------------------------------------------------------ term ledger
def monomials(expr_str):
    """Additive monomials of an expanded invariant expression: [(coef, sympy monomial)]."""
    W = sp.expand(sp.sympify(expr_str, locals={"I1": I1s, "I2": I2s}))
    out = []
    for t in sp.Add.make_args(W):
        c, m = t.as_coeff_Mul()
        if m == 1:
            continue                      # constant: no stress
        out.append((float(c), m))
    return out


def mono_tex(m):
    """mathtext label of a monomial in I1, I2 (negative powers as a fraction)."""
    num, den = [], []
    for base, k in sorted(m.as_powers_dict().items(), key=lambda t: str(t[0])):
        sym = {I1s: "I_1", I2s: "I_2"}[base]
        k = int(k)
        (num if k > 0 else den).append(mono(sym, abs(k)))
    top = "".join(num) or "1"
    return "$" + (top if not den else f"{top}/{''.join(den)}") + "$"


def term_ledger(expr_raw, expr_final, dat, peak):
    """RMS nominal stress carried by each monomial of the raw expression, % of peak,
    sorted descending, with a retained flag (monomial present in the final law)."""
    kept = {m for _, m in monomials(expr_final)}
    rows = []
    for c, m in monomials(expr_raw):
        f1 = sp.lambdify((I1s, I2s), sp.diff(c * m, I1s), "numpy")
        f2 = sp.lambdify((I1s, I2s), sp.diff(c * m, I2s), "numpy")
        vals = []
        for mode, (l, _) in dat.items():
            _, _, I1, I2 = kin(mode, l)
            vals.append(P11(mode, l, f1(I1, I2) + 0 * l, f2(I1, I2) + 0 * l))
        rms = np.sqrt(np.mean(np.concatenate(vals) ** 2))
        rows.append((100 * rms / peak, mono_tex(m), m in kept))
    rows.sort(key=lambda r: -r[0])
    return rows


LEDGER_LO, LEDGER_HI = 0.003, 100.0   # log axis of the ledger (% of peak stress)
C_THR = "#a04040"                     # pruning-budget line, (1+tau)*E_P^(0)


def main():
    D = json.loads(DATA.read_text())
    L = json.loads(PRUNE.read_text())
    fig = plt.figure(figsize=(7.0, 6.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.20, 1.55, 0.78], hspace=0.40, wspace=0.34,
                          left=0.075, right=0.985, top=0.96, bottom=0.105)
    n_bars = max(len(monomials(D[cs["key"]]["pareto"][0]["expression"])) for cs in CASES)
    for j, cs in enumerate(CASES):
        d = D[cs["key"]]
        disc = law_stress(d["final_expression"])
        n_fin = n_terms(d["final_expression"])
        n_raw = n_terms(d["raw_expression"])
        col = C_REC
        # ---------------- data, truth and error measures
        dat = {m: (np.array(v["stretch"]), np.array(v["stress"])) for m, v in d["data"].items()}
        peak = max(np.max(cs["truth"](m, l)) for m, (l, _) in dat.items())
        err_truth = np.sqrt(np.mean(np.concatenate(
            [(disc(m, l) - cs["truth"](m, l)) ** 2 for m, (l, _) in dat.items()])))
        print(cs["key"], "peak", peak, "sigma/peak", d["sigma"] / peak, "rmse vs truth", err_truth,
              "terms", n_raw, "->", n_fin)

        # ---------------- top: chip, truth, discovered law
        ax = fig.add_subplot(gs[0, j]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        # text spacing fixed in inches: k rescales axes fractions to a 1.30-in reference height
        sc = 1.30 / (ax.get_position().height * fig.get_figheight())
        ax.text(0.0, 1 + 0.10 * sc, cs["name"], fontsize=9, fontweight="bold", color=INK, va="top")
        nt = str(cs["n_truth"]) if cs["n_truth"] else "∞"
        chip = f"{cs['kind']} {nt}→{n_fin} term" + ("s" if n_fin > 1 else "")
        ax.text(0.012, 1 - 0.10 * sc, chip, fontsize=7.4, color="white", va="top", ha="left", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.35", fc=col, ec="none"))
        y = 1 - 0.29 * sc
        ax.text(0.0, y, "ground truth", fontsize=7.0, color=INK3, va="top")
        y -= 0.105 * sc
        for line, dy in cs["truth_tex"]:
            ax.text(0.0, y, line, fontsize=7.8, color=INK, va="top"); y -= 0.98 * dy * sc
        notes = cs["note"] if isinstance(cs["note"], list) else [cs["note"]]
        for kn, nt_ in enumerate(notes):
            if kn:
                y -= 0.095 * sc
            ax.text(0.0, y, nt_, fontsize=6.8, color=INK3, va="top")
        y -= 0.15 * sc
        ax.text(0.0, y, "discovered", fontsize=7.0, color=col, va="top", fontweight="bold")
        y -= 0.105 * sc
        pieces = fmt_poly(cs["basis"](d["final_expression"]))
        rend = fig.canvas.get_renderer()
        width = ax.get_window_extent(rend).width * 1.12

        def w(s):
            tt = ax.text(0, 0, "$" + s + "$", fontsize=7.8)
            ww = tt.get_window_extent(rend).width
            tt.remove()
            return ww
        lines, cur = [], r"\Psi="
        for k, p in enumerate(pieces):
            if k and w(cur + p) > width:
                lines.append(cur); cur = r"\quad\ \ "
            cur += p
        lines.append(cur)
        for line in lines:
            ax.text(0.0, y, "$" + line + "$", fontsize=7.8, color=INK, va="top"); y -= 0.14 * sc

        # ---------------- middle: stress curves + residual strip
        gs_m = gs[1, j].subgridspec(2, 1, height_ratios=[1.0, 0.22], hspace=0.08)
        ax = fig.add_subplot(gs_m[0])
        axr = fig.add_subplot(gs_m[1], sharex=ax)
        lmax = max(l.max() for l, _ in dat.values())
        ends = []
        for m, (l, s) in dat.items():
            pc, mk = PATHS[m]
            ax.plot(l[::2], s[::2], mk, ms=2.6, mfc="none", mec=pc, mew=0.55, alpha=0.85, zorder=3)
            lf = np.linspace(1.0, l.max(), 200)
            yd = disc(m, lf)
            ax.plot(lf, yd, "-", lw=1.5, color=pc, zorder=4, solid_capstyle="round")
            ax.plot(lf, cs["truth"](m, lf), "--", lw=0.6, color=INK2, dashes=(3, 2), zorder=5)
            ends.append([yd[-1], lf[-1], m, pc])
            axr.plot(l, 100 * (disc(m, l) - cs["truth"](m, l)) / peak, "-", lw=0.8, color=pc, zorder=3)
        ytop = 1.22 * max(e[0] for e in ends)
        gap = 0.075 * ytop
        ends.sort()
        for k in range(1, len(ends)):
            if ends[k][0] - ends[k - 1][0] < gap and abs(ends[k][1] - ends[k - 1][1]) < 0.1 * (lmax - 1):
                ends[k][0] = ends[k - 1][0] + gap
        for yl, xl, m, pc in ends:
            ax.text(xl + 0.03 * (lmax - 1), yl, m, fontsize=6, color=pc, va="center", ha="left")
        ax.set_xlim(1, lmax + 0.16 * (lmax - 1))
        ax.set_ylim(min(0, ax.get_ylim()[0]), ytop)
        ax.tick_params(labelbottom=False)
        if j == 0:
            ax.set_ylabel(r"$P_{11}$ (MPa)")
        fs.style_axes(ax)
        if d["sigma"] > 0:
            ax.text(0.03, 0.88, rf"noise $\sigma={d['sigma']:g}$ MPa ({100 * d['sigma'] / peak:.1f}%)",
                    transform=ax.transAxes, fontsize=5.8, color=INK2, va="top")
        # RMSE label removed from the panel (value is reported in the main text)
        # ax.text(0.03, 0.97, f"RMSE vs. truth {100 * err_truth / peak:.2f}%",
        #         transform=ax.transAxes, fontsize=5.8, color=INK2, va="top")
        # residual strip
        rmax = max(abs(v) for line in axr.get_lines() for v in line.get_ydata())
        rmax = max(1.15 * rmax, 1e-3)
        axr.set_ylim(-rmax, rmax)
        axr.set_yticks([-rmax / 1.15, 0, rmax / 1.15])
        axr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.1g}"))
        axr.tick_params(labelsize=5.5)
        axr.set_xlabel(r"stretch $\lambda_1$", labelpad=1)
        if j == 0:
            axr.set_ylabel("resid. (%)", labelpad=2, fontsize=6)
        fs.style_axes(axr)

        # ---------------- bottom: deletion ledger (the pruning test of Eq. 2)
        ax = fig.add_subplot(gs[2, j])
        led = L[cs["key"]]
        rows = sorted(((100 * t["deletion_rmse"] / peak,
                        mono_tex(sp.sympify(t["basis"])),
                        not t["removed"]) for t in led["terms"]),
                      key=lambda r: -r[0])
        thr = 100 * led["limit_rmse"] / peak
        ax.axvline(thr, color=C_THR, lw=0.7, ls="--", dashes=(4, 2.5), zorder=2)
        for i, (val, lab, kept) in enumerate(rows):
            v = max(val, LEDGER_LO * 1.02)
            ax.barh(i, v - LEDGER_LO, left=LEDGER_LO, height=0.6, zorder=3,
                    color=C_REC if kept else C_PRUNED, ec="none")
            lab_v = f"{val:.2f}%" if val < 1 else (f"{val:.1f}%" if val < 10 else f"{val:.0f}%")
            ax.text(v * 1.12, i, lab_v + ("   pruned" if not kept else ""), fontsize=5.5,
                    color=INK2 if kept else INK3, va="center", ha="left", zorder=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[1] for r in rows], fontsize=6.5, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(n_bars - 0.5, -0.5)
        ax.set_xscale("log")
        ax.set_xlim(LEDGER_LO, LEDGER_HI * 4)
        ax.set_xticks([0.01, 0.1, 1, 10, 100])
        ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("stress RMSE if term deleted (% of peak)", labelpad=1)
        fs.style_axes(ax)
        ax.grid(False, axis="y")

    # ---------------- panel letters and shared legend
    for j, L in enumerate("ABC"):
        x0 = gs[1, j].get_position(fig).x0
        fs.panel_letter(fig, x0 - (0.062 if j == 0 else 0.04), 0.985, L)
    h = [Line2D([], [], color=PATHS[m][0], marker=PATHS[m][1], mfc="none", ls="-", lw=1.5, ms=3.2,
                label=m) for m in ("UT", "PS", "ET")]
    h += [Line2D([], [], color=C_THR, ls="--", dashes=(4, 2.5), lw=0.7, label="prune budget"),
          Patch(fc=C_REC, ec="none", label="retained term"),
          Patch(fc=C_PRUNED, ec="none", label="pruned term")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=6,
               bbox_to_anchor=(0.53, 0.0), handlelength=2.0, columnspacing=1.2, handletextpad=0.5)
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
