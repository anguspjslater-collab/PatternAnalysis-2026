# Feasibility Review

## 1. User, scope and acceptance criteria

*TODO*

## 2. Model choice and course concepts

*TODO*

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

## 4. Risks, budget and fallback

*TODO*