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