# Feasibility Review

## 1. User, scope and acceptance criteria

**User.** A radiotherapy planner contouring the prostate and organs at risk on pelvic MRI.
The prototype drafts 3D contours for review, not sign-off, since boundary errors near the
rectum or bladder change the dose those organs receive.

**Scope.** HipMRI 3D T2 MRI, 6 labels, downsampled to 128×128×64. Question: does 3D context
(and the Improved U-Net's design) reduce prostate errors at the base and apex, not just in
central slices?

**Acceptance criteria.**

1. Test prostate Dice ≥ 0.70 (Improved 3D U-Net).
2. Base/apex per-slice prostate Dice no worse than the baseline.
3. Prostate HD95 (mm) reported and compared with the baseline.
4. No patient in more than one split.
5. Peak VRAM ≤ 16 GB (fits a clinic workstation GPU); training ≤ 1 h; inference < 1 s/volume.



## 2. Model choice and course concepts

**Data.** 211 volumes / 38 patients, z-scored per volume, 2× downsampled, split by patient
141/35/35 volumes (24/7/7 patients) so weekly re-scans cannot leak into test.

**Baseline:** 3D U-Net (Çiçek et al., 2016): max-pool down, transposed conv up, concatenated
skips, 5.6M params. **Proposed (Hard):** Improved 3D U-Net (Isensee et al., 2017): residual
context modules with dropout, stride-2 conv downsampling, LeakyReLU, deep supervision, 9.5M params.

**Course concepts.** Skip connections restore "where" lost by pooling; 3×3×3 kernels see
neighbouring slices (the base/apex question); residual connections ease gradient flow (ResNet);
deep supervision gives deep decoder levels a direct loss signal; soft Dice + CE handles the
0.1% prostate class.

## 3. Preliminary feasibility evidence

Data audit: 256×256×128 volumes (one 144-slice scan cropped), 1.41–1.88 mm in-plane / 1.56 mm
slices, LPS orientation for all 211, labels 0–5, prostate 0.10% of voxels. No patient overlap
(asserted in `dataset.py`).


| Model (seed 42)     | Best val prostate Dice | Train time | Peak VRAM |
| ------------------- | ---------------------- | ---------- | --------- |
| 3D U-Net (baseline) | 0.866 (epoch 60)       | 13.3 min   | 1.38 GB   |
| Improved 3D U-Net   | 0.872 (epoch 97)       | 16.6 min   | 2.36 GB   |


![Figure 1: Baseline training curves](figures/unet3d_baseline_v2_curves.png)
*Figure 1: Baseline. Prostate Dice is 0 until ~epoch 10 (CE dominates); mild overfitting after
~epoch 60. The Improved model finds the prostate by ~epoch 3 and its val loss does not rise.*

## 4. Risks, budget and fallback

**Risks.** (1) Global Dice score hides boundary failures: both models already exceed 0.70, so the
Improved model may not beat the baseline's 0.866 overall yet still differ at the base and apex;
judge on per-slice Dice (base/mid/apex) and HD95. (2) One seed per model: +0.006 may be noise;
3 seeds each are running. (3) 2× downsampling blurs the prostate boundary; VRAM headroom
(2.4 of 40 GB) makes full-resolution crops feasible.

**Budget.** ~15 min per 150-epoch run on one A100; ~15 runs (3 seeds × 2 models plus ablations)
is about 4 GPU-hours. Queue time, not GPU time, is the constraint.

**Next experiment.** Evaluate both models once on the test set: per-class Dice, per-slice
prostate Dice and HD95, plus 3–5 failure cases.

**Fallback.** The baseline 3D U-Net (Normal tier) is trained and reproducible, so a complete
submission already exists.