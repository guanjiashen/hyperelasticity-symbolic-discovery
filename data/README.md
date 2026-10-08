# Data provenance

`experimental/` preserves numerical nominal-stress/deformation inputs from
the discovery workspace. Column order in `stress_stretch.txt` is nominal
stress followed by stretch (or shear amount), not the reverse. Confirm the
mode-specific conventions before use. Stress-free reference anchors can be
inserted by the loader and should not be counted as measured points.

- `Treloar_1944`: digitized vulcanized-rubber curves, Treloar (1944), MPa.
- `Meunier_2008`: the silicone-rubber tensile data used for the main-text
  identification panel (currently Figure 4A), from Meunier et al. (2008)
  as reproduced by Zhan et al. (2023), MPa. The files contain 16 UT, 16 PS,
  and 12 ET measurements; three zero-stress reference anchors are added by
  the loader, giving 47 samples. See `meunier-source.md`.
- `Yohsuke_2011`: digitized polymer-gel curves, Bitoh et al. (2011), kPa.
- `Budday_2017_brain_CX`: cortex tension/compression and shear, Budday et al.
  (2017), as distributed with Linka et al. (2023) in LivingMatterLab/CANN, kPa.
  See `brain-source.md` for source workbook and extraction conventions.

Original publication figures, scans and workbooks are not redistributed here.
The original experiments remain attributable to their authors; inclusion of
processed numeric tables does not assert ownership of the underlying data
or assign a new data license. Preserve these citations and source terms.
The local CANN source license is retained in `../LICENSES/CANN-MIT.txt`;
it is not asserted as a blanket license for every experimental dataset.

`../data/figure_data/` contains the frozen tables used for main-text
figures. Their target/prediction columns and synthetic-data records preserve
the exact values plotted; they should be used when reproducing published
plots rather than assuming every historical raw input uses the final protocol.
Synthetic generators are in `../src/nn_invariant/preprocess/synthetic_data.py`.
