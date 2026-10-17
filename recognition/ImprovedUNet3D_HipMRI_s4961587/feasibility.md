# Feasibility Review

## 1. User, scope and acceptance criteria

**User.** A radiotherapy planner contouring the prostate and organs at risk on pelvic MRI. The prototype drafts 3D contours for review, not sign-off, since boundary errors near the rectum or bladder change the dose they receive. 

**Scope.** HipMRI 3D T2 MRI, 6 labels, downsampled to 128x128x64. Question: does 3D context reduce prostate errors at the base and apex, not just in central slices? 

**Acceptance criteria.** 

1. Test prostate Dice >= 0.70 (Improved 3D U-Net). 
2. Base/apex per-slice prostate Dice no worse than the baseline.
3. Prostate HD95 (mm) reported and compared with the baseline. 
4. No patient in more than one split. 
5. Peak VRAM ≤ 16 GB, so inference fits a standard clinic workstation GPU.
6. Training <= 1 h, inference < 1 s/volume.

## 2. Model choice and course concepts

Baseline: 3D U-Net (Çiçek et al., 2016): 4 levels, max-pool down, transposed conv up, concatenated skips. 5.6M params. 

Proposed: Improved 3D U-Net (Isensee et al., 2017): residual "context modules," stirde-2 conv downsampling, LeakyReLU, InstanceNorm, dorpout, and deep supervision. HARD DIFFICULTY. 

Course concepts: 

- U-Net and skip connections: encoder captures "what" and skips restore "where"
- 3D convolutions: 3x3x3 kernels see neighbouring slices 
- Residula connections: easier graident flow in deeper networks (ResNet)
- Soft Dice + cross-entropy: handles Prostate class imbalance (~0.1%). Baseline training shows it takes ~10 epochs for the basic U-Net to begin to select Prostate areas

![Figure 1: Baseline U-Net training/validation Dice over epochs](figures/unet3d_baseline_v2_curves.png)

## 3. Preliminary feasibility evidence

**Data audit (HipMRI 3D,** `semantic_MRs` **+** `semantic_labels_only`**):**


| Property             | Value                                                         |
| -------------------- | ------------------------------------------------------------- |
| Volumes / patients   | 211 / 38 (1–8 weekly scans per patient)                       |
| Shape                | 256×256×128 (one scan 256×256×144, cropped to 128 slices)     |
| Voxel size           | 1.41–1.88 mm in-plane, 1.56 mm between slices                 |
| Orientation          | LPS for all 211 (checked from NIfTI affines)                  |
| Labels               | 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate |
| Class balance (B040) | 57.1 / 37.0 / 3.9 / 1.65 / 0.36 / **0.10** %                  |


**Patient-level split (seed 42):**


| Split | Volumes | Patients |
| ----- | ------- | -------- |
| Train | 141     | 24       |
| Val   | 35      | 7        |
| Test  | 35      | 7        |


No patient appears in more than one split (asserted in `dataset.py`).

**Preprocessing check:** after per-volume z-score and 2× downsampling,
images are (1, 128, 128, 64) with mean 0.00 and std 0.98; masks keep exactly
labels 0–5; the prostate shrinks to ~700–1,000 voxels.

**Downsampled MRI slices:**

![Figure 2: Baseline U-Net training/validation Dice over epochs](figures/downsample_check.png)

## 4. Risks, budget and fallback

The key risk is that Global Dice Loss hides boundary failures. The Imporved 3D U-Net may not beat baselines 0.866 overall, yet still differ at the base and apex, possibly indicating superior use in real world analysis. A basic 3D U-Net has already been trained, eliciting an overall 0.866 Prostate Dice after ~50 epochs of training (ran for 150 with little incremental improvement). Performance should be judged on per-slice dice base / mid / apex thirds plus a surface distance (HD95) in mm, not just global Dice. 

There is also risk the initial downsampling stipulated in the task sheet blurs the task sheet. Initial training indicates VRAM headroom (1.38GB of 40) makes full-resolution crops a feasible option. 

### Computing budget

- **Measured:** about 13 minutes and 1.38 GB peak VRAM per 150-epoch run on one A100 (5.6M parameters).
- **Planned:** roughly 10–20 runs before 30 Oct: Improved U-Net × 3 seeds, baseline × 3 seeds, plus 2–3 ablations (flip augmentation, spacing resampling, full-resolution crops). That’s about **4–5 GPU-hours** in total, well within budget. Queue time, not GPU time, is the real constraint.

### Next planned experiment

Train the **Improved 3D U-Net** (residual context modules, stride-2 downsampling, LeakyReLU, deep supervision) under the **identical** data split, loss, optimiser and 150-epoch schedule, then compare its val Dice per class and per slice with the baseline.

### Fallback

If the Improved U-Net fails to train or doesn’t run in time, submit the **working baseline 3D U-Net** (Normal tier, implementation capped at 15/20), with the same evaluation and boundary analysis. It’s already trained and reproducible, so the fallback is fully in place.