# Prostate Segmentation of 3D Pelvic MRI: 2D vs 3D vs Improved 3D U-Net

Angus Slater (49615878)

COMP3710, The Univserity of Queensland

October 30th, 2026

## 1. Problem and User 

Prostate radiotherapy planning requires precise segmentation of the clinical target (prostate) and the surrounding organs at risk (OAR: bladder, rectum, and bone). Once segmented, the selected area then receives a planned radiation dose to kill the unhealthy cells. If the segmentation boundaries leak into surrounding tissue the patient can suffer severe proctitis or cystitis. If the segmentation omits cancerous cells, the patient can experience  a relapse or metastasis.  

The contouring of the prostate boundaries is conventionally performed by a radiation oncologist, who oversees treatment and ensures safety throughout the therapy (David Geffen School of Medicine, 2016). Building a machine learning (ML) model to segment prostates could help reduce contouring time and variability between clinicians. A model would be used in conjunction with an oncologist to ensure appropriate decisions are made for each case. 

The engineering problem stems from the composition of HipMRI scans in which the prostate gland can account for ~0.1% of total volume. Further, the base and apex blend into the bladder neck and pelvic floor and the voxels are partial-volume. This makes cross-entropy loss alone insufficient as it is dominated by background, and global Dice limited as it fails to precisely capture boundaries and is instead inflated by the easily predicted central part of the prostate. 

The report details the implementation of a control 2D U-Net segmentation model with no inter-slice context (Ronneberger et al., 2015), a baseline 3D U-Net model based on the paper from Çiçek et al. (2016), and an improved 3D U-Net based on the development by Isensee et al. (2017). To address the problems associated with the task, these models were evaluated on per-class Dice, per-third Dice, 95th-percentile Hausdorff distance (HD95), resource cost (parameters, training time, and VRAM) and a failure autopsy.   

The goal of the report is to determine if 3D context and the improved 3D U-Net design reduce base / apex errors, or only improve the prediction of the centre of the prostate, and if these improvements are sufficient to warrant the increased computational cost. 

Table 1 summarises the key finding from each model. Overall, the recommendation of the report is to .... (smth about using 3d vs not using it etc.)

Table 1. Key results 

## 2. Feasibility Review

## 3. Data and Preprocessing

Data were provided by the HipMRI Study (Dowling & Greer, 2021). It comes from the HipMRI study, a prostate radiotherapy study in which patients were scanned with MRI each week during treatment (Dowling et al., 2015). Labels are segmented in 3D by an MR physicist with over 10 years' experience and include background, body, bone, bladder, rectum, and prostate (~0.1% volume) (task sheet Appendix A). It was accessed through the course copy on Rangpur at `/home/groups/comp3710/HipMRI_Study_open/` (`semantic_MRs`, `semantic_labels_only`).

The task required downsampling of the original image but did not specify a method. As the voxels are nearly cubic (about 1.4-1.9 mm in-plane, 1.56 mm between slices), the images were first downsampled by 2x on every axis (256 x 256 x 128 to 128 x 128 x 64, one image was also cropped at the head (far from prostate) as its initial size was 256 x 256 x 144). This gave 8x fewer voxels and a ~8x decrease in computation per epoch. Downsampling was done in `load_volume()` in the `dataset.py` file by first normalising the image (z-score) then resizing with `torch.nn.functional.interpolate`: trilinear for the image (each voxel is a smooth weighted blend of nearby originals), and nearest for labels (so labels stay as whole class IDs, no decimals). Figure 1 shows an example of a downsampled image against its corresponding original. Downsampling was required by the task, however, it means voxels are of size ~3.1mm (length) and prostate boundaries appear more block-like. 

![Figure 1: Downsampled example](figures/downsample_check.png)

In all there were 211 scans across 38 patients, with a median of 8 scans per patient (min 1, max 8). 22 patients have the full 8 weekly scans. The data was split into train, validation, and test sets by patient (shuffled, fixed seed ~70 / 15 / 15 split). As week-to-week scans for a given patient are near identical, splitting by volume would risk the model being able to identify and learn specific patients, possibly leading to overstated success in testing (if a patients scans happened to be present in both the train and test sets). No patient appears in more than one split (asserted in code). Validation selected the best checkpoint; the test set is evaluated once. Table 2 quantifies the split. 

Table 2. Train, validation, and test split composition
| Split       | Patients | Volumes (scans) | % of volumes |
| ----------- | -------- | --------------- | ------------ |
| Train       | 24       | 141             | 66.8%        |
| Validation  | 7        | 35              | 16.6%        |
| Test        | 7        | 35              | 16.6%        |
| **Total**   | **38**   | **211**         | **100%**     |

## 4. Models

## 5. Training Setup 

## 6. Results
### 6.1 Validation 
### 6.2 Test Set
### 6.3 Resources

## 7. Failure Autopsy

## 8. Recommendation and Trade-Offs

## 9. Limitations and Future Work

## 10. Reproducing this Work

## 11. AI Usage Disclosure

## 12. Reference List

https://medschool.ucla.edu/blog-post/what-does-a-radiation-oncologist-do 

https://doi.org/10.25919/45t8-p065 

https://doi.org/10.1016/j.ijrobp.2015.08.045 