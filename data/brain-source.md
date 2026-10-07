# Human brain-tissue data (Budday et al. 2017)

Source: LivingMatterLab/CANN repository (github.com/LivingMatterLab/CANN),
file BRAIN/Invariant-based/input/CANNsBRAINdata.xlsx, downloaded 2026-09-26.

Underlying experiments: S. Budday et al., "Mechanical characterization of
human brain tissue," Acta Biomaterialia 48:319-340 (2017) — bib key
Budday2017c in Ref.bib. Data as distributed with K. Linka, S.R. St. Pierre,
E. Kuhl, "Automated model discovery for human brain using constitutive
artificial neural networks," Acta Biomaterialia 160:134-151 (2023) — bib key
Linka2023.

Contents (Sheet1): averaged uniaxial tension/compression (lambda, P11 in kPa)
and simple-shear (gamma, P12 in kPa) curves for four regions:
CX = cortex, CR = corona radiata, BG = basal ganglia, CC = corpus callosum.

## Region and conventions used in the paper

Region: CX (cortex) only, matching Materials and Methods ("gray-matter
(cortex) specimens"). Averaged, preconditioned response curves as
distributed with the CANN code of Linka et al. (2023).

Extracted branch files (this folder, CX/<MODE>/stress_stretch.txt;
column 1 = nominal stress in kPa, column 2 = stretch lambda or amount of
shear gamma; ordered monotonically away from the reference state, which is
NOT included in the files — the pipeline loader inserts it):

- UT: tension,      lambda in (1, 1.1],   16 points
- UC: compression,  lambda in [0.9, 1),   16 points
- SS: shear,        gamma  in (0, 0.2],   16 points
- SN: shear,        gamma  in [-0.2, 0),  16 points

With the inserted reference point per branch this gives 17 points per
branch and 68 points overall — the counts reported in the manuscript
(consistent with the Treloar/Yohsuke convention, whose reported counts also
include the inserted reference points).

The UT/UC branches enter the loss through P11 = 2(lambda - lambda^-2)
(dPsi/dI1 + dPsi/dI2 / lambda); the SS/SN branches through
P12 = 2*gamma*(dPsi/dI1 + dPsi/dI2) with I1 = I2 = 3 + gamma^2.

## Pipeline run (2026-09-27)

Code: WSL, /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant
(UC/SS/SN modes added to preprocess, solver, CLI, and the
Symbolic_Regression modules). Data copied to
/home/guanjs/NN-constitutive/FFNN/fitting-data-PK/Budday_2017_brain_CX/.

CANN: 8 Softplus neurons, L-BFGS, lr 0.1, 400 outer iterations, seed 42
(output/NN_output-brainCX); validation stress RMSE 2.15e-2 kPa.

CSR: loadcase sampling 32 points per path (UT 1-1.1, UC 0.9-1, SS 0-0.2;
N_s = 96), energy/stress weights 0.1/1.0, operators + - * square,
12 populations x 60, 180 iterations, maxsize 12, deterministic, seed 42,
pruning tolerance 2% (output/SR_output-brainCX; copied to
results/SR_output in this folder).

Discovered law (kPa, after the Psi(3,3)=0 shift):
  Psi = (I1 - 1.636408)(1.1462871 I1 I2 - 19.059301) + 11.9214990958232
      = 1.1463 I1^2 I2 - 1.8758 I1 I2 - 19.059 I1 + 43.110

Overall experimental stress RMSE 1.8854e-2 kPa over 68 points; branch-wise
1.5943e-2 (UT), 1.9436e-2 (UC), 1.9874e-2 kPa (SS/SN). Initial shear
modulus mu = 2(dPsi/dI1 + dPsi/dI2)|(3,3) = 1.27 kPa. dPsi/dI1 < 0 and
dPsi/dI2 > 0 over the calibrated domain; 2(dPsi/dI1 + dPsi/dI2) in
[1.27, 6.08] kPa over 0.9 <= lambda1, lambda2 <= 1.1.

Manuscript: pnas-main.tex Eq. [11] (eq:brain) and Methods; si-body.tex
section S3.3 (sec:si-brain), Table tab:brain-settings, Figure
fig:brain-csr (figs/brain_csr_predictions_p11.png,
brain_csr_predictions_p12.png, brain_csr_energy_contour_simple.png).
