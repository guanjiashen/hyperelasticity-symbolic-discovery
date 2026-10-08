"""
Design mock-up for the PNAS benchmark figure
"Known physics is recovered, or compressed, from noisy data".

Layout: one column per classical model (Mooney-Rivlin, Yeoh, Arruda-Boyce, Ogden).
  header : ground-truth law  ->  discovered law, with term counts (recovered / compressed)
  row A  : nominal stress on UT / PS / ET; noisy synthetic data (markers),
           discovered law (solid), ground truth (thin dashed)
  row B  : pruning trace: stress error vs. number of retained terms, the 2 % budget,
           the accepted stopping point, and the true term count

Everything in row A is computed from the analytic laws.  The "discovered" laws and
the pruning traces are an ILLUSTRATIVE EMULATION (least-squares fit on a small
invariant-polynomial basis + greedy contribution-aware pruning with tau = 0.02);
replace them by the CANN-CSR pipeline output (raw SR expression, per-deletion refit
errors, final law) once the Yeoh and Ogden runs exist.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- style
SURF   = "#ffffff"
INK    = "#1c1c1c"; INK2 = "#5a5a5a"; INK3 = "#9a9a9a"
GRID   = "#e6e6e6"
C_UT, C_PS, C_ET = "#2f6db3", "#c8571b", "#6a4fb3"       # validated CVD-safe trio
MK_UT, MK_PS, MK_ET = "o", "s", "^"
C_TRUTH = "#1c1c1c"
C_KEEP  = "#2f6db3"; C_DROP = "#bdbdbd"; C_BUDGET = "#e9eef6"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": INK3, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "mathtext.fontset": "dejavusans",
})

# ---------------------------------------------------------------- kinematics
def paths(lmax_ut=7.0, lmax_ps=7.0, lmax_et=5.0, n=40):
    out = {}
    for name, lmax, f in (("UT", lmax_ut, lambda l: l**-0.5),
                          ("PS", lmax_ps, lambda l: np.ones_like(l)),
                          ("ET", lmax_et, lambda l: l)):
        l1 = np.linspace(1.0, lmax, n); l2 = f(l1); l3 = 1/(l1*l2)
        I1 = l1**2 + l2**2 + l3**2
        I2 = l1**2*l2**2 + l2**2*l3**2 + l3**2*l1**2
        out[name] = dict(l1=l1, l2=l2, l3=l3, I1=I1, I2=I2)
    return out

def P11_from_invariants(p, dW1, dW2):
    return 2*(p["l1"] - p["l1"]**-3 * p["l2"]**-2) * (dW1 + p["l2"]**2 * dW2)

# ---------------------------------------------------------------- ground truths
def mooney_rivlin(p, C10=0.18, C01=0.02):
    return P11_from_invariants(p, C10*np.ones_like(p["I1"]), C01*np.ones_like(p["I1"]))

def yeoh(p, C10=0.17, C20=-1.5e-3, C30=2.0e-5):
    x = p["I1"] - 3
    return P11_from_invariants(p, C10 + 2*C20*x + 3*C30*x**2, np.zeros_like(x))

def arruda_boyce(p, mu=1.0, lm=5.0):
    C = [0.5, 1/20, 11/1050, 19/7000, 519/673750]
    dW1 = sum(mu*C[k]*(k+1)*p["I1"]**k / lm**(2*k) for k in range(5))
    return P11_from_invariants(p, dW1, np.zeros_like(dW1))

def ogden(p, mu=(0.63, 0.0012, -0.01), al=(1.3, 5.0, -2.0)):
    l1, l3 = p["l1"], p["l3"]
    return sum(m*(l1**a - l3**a) for m, a in zip(mu, al)) / l1

# ---------------------------------------------------------------- basis for the emulated search
BASIS = [
    (r"I_1",        lambda p: (np.ones_like(p["I1"]),            np.zeros_like(p["I1"]))),
    (r"I_1^{2}",    lambda p: (2*p["I1"],                        np.zeros_like(p["I1"]))),
    (r"I_1^{3}",    lambda p: (3*p["I1"]**2,                     np.zeros_like(p["I1"]))),
    (r"I_2",        lambda p: (np.zeros_like(p["I1"]),           np.ones_like(p["I1"]))),
    (r"I_1 I_2",    lambda p: (p["I2"],                          p["I1"])),
    (r"I_2/I_1",    lambda p: (-p["I2"]/p["I1"]**2,              1/p["I1"])),
]

def design_matrix(P, cols):
    return np.column_stack([np.concatenate([P11_from_invariants(P[k], *BASIS[i][1](P[k]))
                                            for k in ("UT", "PS", "ET")]) for i in cols])

def refit(P, y, cols):
    A = design_matrix(P, cols)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c, np.sqrt(np.mean((A @ c - y)**2))

def greedy_prune(P, y, cols, tau=0.02):
    """Contribution-aware pruning against a fixed reference error."""
    c, e0 = refit(P, y, cols)
    trace = [(len(cols), e0)]
    active = list(cols)
    while len(active) > 1:
        trials = []
        for k in active:
            rest = [i for i in active if i != k]
            _, e = refit(P, y, rest)
            trials.append((e, k))
        e_best, k_best = min(trials)
        if e_best <= (1+tau)*e0:
            active.remove(k_best); trace.append((len(active), e_best))
        else:
            trace.append((len(active)-1, e_best)); break    # first rejected deletion (shown hollow)
    c, e = refit(P, y, active)
    return active, c, e, trace


# ---------------------------------------------------------------- expression trees
OPC = "#6a4fb3"      # operator node stroke (as in Fig. 1C)
def phi_tree(name):
    I1, I2 = ("leaf", "$I_1$"), ("leaf", "$I_2$")
    return {r"I_1": I1, r"I_2": I2,
            r"I_1^{2}": ("sq", I1),
            r"I_1^{3}": ("×", I1, ("sq", I1)),
            r"I_1 I_2": ("×", I1, I2),
            r"I_2/I_1": ("÷", I2, I1)}[name]

def fmt_coef(ci):
    a = abs(ci); sg = "\u2212" if ci < 0 else ""
    if a >= 0.01: return f"{sg}{a:.3g}"
    m, e = f"{a:.2e}".split("e"); return rf"{sg}{m}$\times$10$^{{{int(e)}}}$"

def law_tree(c, active):
    terms = [("×", ("cleaf", rf"$c_{{{k+1}}}$", False), phi_tree(BASIS[i][0]))
             for k, (ci, i) in enumerate(zip(c, active))]
    return terms[0] if len(terms) == 1 else ("+",) + tuple(terms)

def layout(node, depth=0, st=None):
    """Return (nodes, edges, x) with unique ids; nodes = [(id, x, depth, node)]."""
    if st is None: st = {"x": 0, "n": 0, "nodes": [], "edges": []}
    my = st["n"]; st["n"] += 1
    if node[0] in ("leaf", "cleaf"):
        x = st["x"]; st["x"] += 1
        st["nodes"].append((my, x, depth, node)); return st, x
    xs = []
    for ch in node[1:]:
        cid = st["n"]
        _, xc = layout(ch, depth+1, st); xs.append(xc); st["edges"].append((my, cid))
    x = float(np.mean(xs)); st["nodes"].append((my, x, depth, node))
    return st, x

def draw_tree(ax, tree, fs=5.6):
    st, _ = layout(tree)
    pos = {i: (x, d, n) for i, x, d, n in st["nodes"]}
    nleaf = st["x"]; dmax = max(v[1] for v in pos.values())
    if nleaf > 8: fs -= 0.9
    def xy(k):
        x, d, _ = pos[k]
        return (x + 0.5) / nleaf, 0.97 - d / max(dmax, 1) * 0.72
    for a_, b_ in st["edges"]:
        (xa, ya), (xb, yb) = xy(a_), xy(b_)
        ax.plot([xa, xb], [ya, yb], color=INK3, lw=0.6, zorder=1, transform=ax.transAxes)
    for k, (x, d, node) in pos.items():
        X, Y = xy(k); kind = node[0]
        if kind == "leaf":
            ax.text(X, Y, node[1], fontsize=fs, ha="center", va="center", color=INK, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.22", fc=SURF, ec=INK3, lw=0.6), zorder=3)
        elif kind == "cleaf":
            ax.text(X, Y, node[1], fontsize=fs, ha="center", va="center", color=INK, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.22", fc="#f2f2f2", ec=INK3, lw=0.6), zorder=3)
        else:
            lab = {"sq": "□²"}.get(kind, kind)
            ax.text(X, Y, lab, fontsize=fs, ha="center", va="center", color=OPC, transform=ax.transAxes,
                    bbox=dict(boxstyle="circle,pad=0.18", fc=SURF, ec=OPC, lw=0.8), zorder=3)

# ---------------------------------------------------------------- cases
CASES = [
    dict(name="Mooney–Rivlin", fn=mooney_rivlin, n_truth=2, sigma=0.10, unit="MPa",
         truth=r"$C_{10}(I_1\!-\!3)+C_{01}(I_2\!-\!3)$",
         raw=[0, 1, 3, 4], kind="recovered"),
    dict(name="Yeoh", fn=yeoh, n_truth=3, sigma=0.10, unit="MPa",
         truth=r"$\sum_{k=1}^{3}C_{k0}(I_1\!-\!3)^{k}$",
         raw=[0, 1, 2, 3, 4], kind="recovered"),
    dict(name="Arruda–Boyce", fn=arruda_boyce, n_truth=5, sigma=0.50, unit="MPa",
         truth=r"$\mu\sum_{k=1}^{5}\frac{C_k}{\lambda_m^{2k-2}}(I_1^{k}\!-\!3^{k})$",
         raw=[0, 2, 3, 4], kind="compressed"),
    dict(name="Ogden", fn=ogden, n_truth=None, sigma=0.20, unit="MPa",
         truth=r"$\sum_{p=1}^{3}\frac{\mu_p}{\alpha_p}(\lambda_1^{\alpha_p}\!+\!\lambda_2^{\alpha_p}\!+\!\lambda_3^{\alpha_p}\!-\!3)$",
         raw=[0, 1, 2, 3, 5], kind="compressed"),
]

P = paths()
fig = plt.figure(figsize=(7.1, 6.0), facecolor=SURF)
gs = fig.add_gridspec(4, 4, height_ratios=[0.55, 0.95, 1.55, 1.05], hspace=0.55, wspace=0.42,
                      left=0.075, right=0.975, top=0.96, bottom=0.09)

for j, cs in enumerate(CASES):
    # ---- data
    y_true = {k: cs["fn"](P[k]) for k in ("UT", "PS", "ET")}
    y_noisy = {k: v + rng.normal(0, cs["sigma"], v.shape) for k, v in y_true.items()}
    y_stack = np.concatenate([y_noisy[k] for k in ("UT", "PS", "ET")])
    active, c, e_fin, trace = greedy_prune(P, y_stack, cs["raw"])
    peak = max(v.max() for v in y_true.values())

    # ---- header: truth -> discovered
    ax = fig.add_subplot(gs[0, j]); ax.axis("off")
    n_raw, n_fin = len(cs["raw"]), len(active)
    ax.text(0.0, 1.05, cs["name"], fontsize=8, fontweight="bold", color=INK, va="top")
    ARROW, INF = "\u2192", "\u221e"
    nt = cs["n_truth"] if cs["n_truth"] else INF
    tag = (f"recovered  {nt}{ARROW}{n_fin} terms" if cs["kind"] == "recovered"
           else f"compressed  {nt}{ARROW}{n_fin} terms")
    ax.text(0.0, 0.70, tag, fontsize=5.8, color=SURF, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc=C_KEEP if cs["kind"] == "recovered" else "#6a4fb3", ec="none"))
    ax.text(0.0, 0.30, "ground truth", fontsize=5.5, color=INK3, va="top")
    ax.text(0.0, 0.10, cs["truth"], fontsize=6.2, color=INK, va="top")

    # ---- tree row: discovered law as an expression tree
    tax = fig.add_subplot(gs[1, j]); tax.axis("off")
    tax.text(0.0, 1.10, "discovered", fontsize=5.5, color=INK3, va="bottom", transform=tax.transAxes)
    draw_tree(tax, law_tree(c, active))
    coefs = ",  ".join(rf"$c_{{{k+1}}}$={fmt_coef(ci)}" for k, ci in enumerate(c))
    tax.text(0.0, -0.12, coefs, fontsize=5.2, color=INK2, va="top", transform=tax.transAxes, wrap=True)

    # ---- row A: stress curves
    ax = fig.add_subplot(gs[2, j])
    ax.set_facecolor(SURF)
    for k, col, mk in (("UT", C_UT, MK_UT), ("PS", C_PS, MK_PS), ("ET", C_ET, MK_ET)):
        p = P[k]
        ax.plot(p["l1"][::2], y_noisy[k][::2], mk, ms=3.0, mfc="none", mec=col, mew=0.7, zorder=3)
        ax.plot(p["l1"], y_true[k], "--", lw=0.7, color=C_TRUTH, alpha=0.75, zorder=4)
        y_disc = np.zeros_like(p["l1"])
        for ci, i in zip(c, active):
            y_disc += ci * P11_from_invariants(p, *BASIS[i][1](p))
        ax.plot(p["l1"], y_disc, "-", lw=1.4, color=col, zorder=5)
        ax.text(p["l1"][-1], y_disc[-1], f" {k}", fontsize=6, color=col, va="center", ha="left")
    ax.set_xlim(1, 7.9); ax.set_xlabel(r"$\lambda_1$", labelpad=1)
    if j == 0: ax.set_ylabel(r"$P_{11}$ (MPa)")
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.text(0.03, 0.96, rf"$\sigma={cs['sigma']}$ MPa", transform=ax.transAxes,
            fontsize=6, color=INK2, va="top")
    ax.text(0.03, 0.86, f"RMSE {e_fin/peak*100:.2f}% of peak", transform=ax.transAxes,
            fontsize=6, color=INK2, va="top")

    # ---- row B: pruning trace
    ax = fig.add_subplot(gs[3, j]); ax.set_facecolor(SURF)
    e0 = trace[0][1]
    ns = [t[0] for t in trace]; es = [t[1]/peak*100 for t in trace]
    nacc = n_raw - n_fin + 1
    ax.axhline(1.02*e0/peak*100, color=C_KEEP, lw=0.8, ls=(0, (3, 2)), alpha=0.6, zorder=1)
    ax.text(0.45, 1.02*e0/peak*100, "budget ", fontsize=5.5, color=C_KEEP, ha="right", va="bottom", alpha=0.9)
    ax.plot(ns[:nacc], es[:nacc], "-", color=C_KEEP, lw=1.2, zorder=3)
    ax.plot(ns[:nacc], es[:nacc], "o", ms=4, color=C_KEEP, mec=SURF, mew=0.8, zorder=4)
    ymax = max(es) * 1.25
    if len(trace) > nacc:               # the rejected deletion
        ax.plot(ns[-1], es[-1], "o", ms=4, mfc=SURF, mec=C_DROP, mew=1.0, zorder=4)
        ax.plot(ns[-2:], es[-2:], ":", color=C_DROP, lw=1.0, zorder=2)
        ax.annotate("rejected", (ns[-1], es[-1]), xytext=(0, -7), textcoords="offset points",
                    fontsize=5.5, color=INK3, ha="center", va="top")
    ax.plot(n_fin, e_fin/peak*100, "o", ms=6.5, mfc=SURF, mec=C_KEEP, mew=1.4, zorder=5)
    if cs["n_truth"] is not None:
        ax.axvline(cs["n_truth"], color=INK3, lw=0.7, ls=(0, (2, 2)), zorder=2)
        ax.text(cs["n_truth"] - 0.08, 0.97, "truth", fontsize=5.5, color=INK3, ha="right", va="top",
                transform=ax.get_xaxis_transform())
    else:
        ax.text(0.03, 0.96, "truth not in\n$(I_1,I_2)$ form", fontsize=5.5, color=INK3,
                ha="left", va="top", transform=ax.transAxes)
    ax.set_xlim(n_raw + 0.6, 0.4); ax.set_xticks(range(1, n_raw + 1))
    ax.set_ylim(0, ymax)
    ax.set_xlabel("terms retained", labelpad=1)
    if j == 0: ax.set_ylabel("stress error\n(% of peak)")
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

# ---- panel letters and one shared legend
for j, L in enumerate("ABCD"):
    x0 = gs[0, j].get_position(fig).x0
    fig.text(x0 - 0.045 if j == 0 else x0 - 0.05, 0.975, L, fontsize=9, fontweight="bold", va="top")
handles = [Line2D([], [], color=C_TRUTH, ls="--", lw=0.8, label="ground truth"),
           Line2D([], [], color=INK2, lw=1.4, label="discovered law"),
           Line2D([], [], color=INK2, marker="o", mfc="none", ls="", ms=3, label="noisy data"),
           Line2D([], [], color=C_KEEP, marker="o", ms=4, ls="-", label="accepted deletion"),
           Line2D([], [], color=C_KEEP, marker="o", mfc=SURF, mew=1.4, ms=6, ls="", label="selected law"),
           Line2D([], [], color=C_KEEP, lw=0.8, ls=(0, (3, 2)), alpha=0.6, label="2 % error budget")]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6,
           bbox_to_anchor=(0.53, 0.0), handlelength=1.8, columnspacing=1.4)
fig.text(0.99, 0.985, "DESIGN MOCK-UP: discovered laws and pruning traces are an emulation, not pipeline output",
         fontsize=5, color="#b04040", ha="right", va="top", style="italic")

out = "/mnt/user-data/outputs/fig2_benchmarks_mockup"
fig.savefig(out + ".png", dpi=300, facecolor=SURF)
fig.savefig(out + ".pdf", facecolor=SURF)
print("saved", out)
