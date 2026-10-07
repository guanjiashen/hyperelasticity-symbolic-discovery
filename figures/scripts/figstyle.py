"""
Shared plotting style for the PNAS manuscript figures (Figs 2-4).

Importing this module registers Arial when it is available (Windows / WSL
font folders), installs the rcParams used by every figure, and exposes the
ink greys, the loading-path palette and a few small helpers.

    import figstyle as fs
    fs.style_axes(ax); fs.panel_letter(fig, x, y, "A"); fs.save(fig, path)

The UT / PS / ET colour trio is CVD-validated; do not change the hexes.
"""
from pathlib import Path

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ fonts
for _f in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):
    for _d in (Path(__file__).resolve().parents[1] / "fonts",
               Path("/mnt/c/Windows/Fonts"), Path("C:/Windows/Fonts")):
        if (_d / _f).exists():
            font_manager.fontManager.addfont(str(_d / _f))
            break

# ------------------------------------------------------------------ colours
INK, INK2, INK3, GRID = "#1c1c1c", "#5a5a5a", "#9a9a9a", "#e6e6e6"

# loading paths: (line/marker colour, marker).  UC shares the UT colour.
PATHS = {
    "UT": ("#2f6db3", "o"),
    "PS": ("#c8571b", "s"),
    "ET": ("#6a4fb3", "^"),
    "SS": ("#3a8f7a", "D"),
    "UC": ("#2f6db3", "v"),
}
UT_LIGHT = "#9dbde0"            # lighter tint of the UT blue (secondary lines)

C_REC, C_CMP = "#2f6db3", "#6a4fb3"   # chips: recovered / compressed
C_ACCENT = "#2f6db3"                   # chips in Fig. 3
C_REJ = "#a8a8a8"                      # rejected / pruned
C_PRUNED = "#d9d9d9"                   # pruned bars in the term ledger
C_BAND = "#f0f0f0"                     # calibrated-range shading

FIELD_CMAP = "viridis"                 # sequential colormap for fields

# ------------------------------------------------------------------ rcParams
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.edgecolor": INK3, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.labelcolor": INK, "pdf.fonttype": 42,
}
plt.rcParams.update(RC)


# ------------------------------------------------------------------ helpers
def style_axes(ax, grid=True):
    """Hide top/right spines, light grid below the data."""
    if grid:
        ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_letter(fig, x, y, letter, **kw):
    """Bold 9 pt panel letter in figure coordinates (top-left anchored)."""
    opts = dict(fontsize=9, fontweight="bold", color=INK, va="top", ha="left")
    opts.update(kw)
    return fig.text(x, y, letter, **opts)


def fnum(c, sig=4):
    """|c| formatted for mathtext: plain for >= 1e-2, else m x 10^e."""
    a = abs(c)
    if a >= 1e-2:
        return f"{a:.{sig}g}"
    m, e = f"{a:.{sig - 2}e}".split("e")
    return rf"{m}{{\times}}10^{{{int(e)}}}"


def save(fig, path, dpi=300, **kw):
    """Write <path>.pdf and a <path>.png preview at `dpi`."""
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"), **kw)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, **kw)
    print("saved", path)
