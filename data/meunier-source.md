# Meunier data used in the main-text identification panel

The files in `experimental/Meunier_2008/` contain the experimental markers
used in the main-text homogeneous-test identification panel, currently
Figure 4A in `pnas-main.tex` (the panel referred to as Figure 14A in the
update request). They replace the broader historical dataset that also
contained compression measurements.

These are the tensile subsets of the silicone-rubber data of Meunier et al.
(2008), as reproduced in Figure 9 of Zhan et al. (2023). The manuscript
workspace stores them under
`simulation/figure10_cann_csr/figure15_data/{UT,PS,ET}/stress_stretch.txt`.
Both `plot_identification.py` and `plot_fig4_panels.py` use these inputs.
The published files retain the exact numerical values of those inputs.

| Mode | Measured points |
| --- | --- |
| Uniaxial tension (UT) | 16 |
| Pure shear (PS) | 16 |
| Equibiaxial tension (ET) | 12 |
| Total | 44 |

Each row contains nominal stress in MPa followed by dimensionless stretch.
The files contain measured points only. The NN_invariant loader prepends
`(stress, stretch) = (0, 1)` to each mode, producing 47 samples in total.
The reference anchors are not additional experimental measurements.

`examples/figure10_cann_csr/cann_csr_config.json` references these files
directly. Replacing the inputs does not retrain the archived network or
regenerate frozen model and solver results; these inputs are the same ones
used in the corresponding manuscript identification workflow.

Original experimental attribution and source-data terms remain applicable;
this update does not assign a new dataset license.
