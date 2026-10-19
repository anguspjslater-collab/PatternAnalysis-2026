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

A 2D control U-Net (Ronneberger et al., 2015), base 3D U-Net (Çiçek et al., 2016) and an improved 3D U-Net (Isensee et al., 2017) were all trained on the HipMRI data. They sahre the encoder-decoder (U) shape: 4 down-sampling levels, channels 16 -> 32 -> 64 -> 128 -> 256, skip connections concatenated on channels, and a 1x1(x1) output convolution. They also share InstanceNorm (batch size 2 makes BatchNorm's batch statistics unreliable), and the identical loss and training setup, so architecture is the only variable. 

See Appendix A for an AI-generated model architecture diagram for all three models. 

### 4.1 Control 2D U-Net

The 2D U-Net serves as a control. It has no inter-slice context, so comparing it with the 3D models isolates the value of the 3D context. The volume is split into 64 axial slices, each segmented independently (meaning no continuity across axial slices), prediction restacked into (B, 6, 128, 128, 64). 

The 4 downsampling layers and bottleneck of the 2D U-Net have a double convolution structure, where this process is repeated twice per layer:

1. 3x3 convolution (stride 1)
2. InstanceNorm 
3. ReLU 

Between encoder layers, a 2x2 max-pool halves the height and width (slices never pooled together).

This takes the number of channels from 1 to 256 at the bottleneck, where the data has the shape (B x 64, 256, 8, 8). From here it is passed to the decoder, which involves four decoder levels, each: 

1. 2x2 Transposed convolution (halves the channels, doubles height and width) 
2. Concatenate with the matching encoder level's output (skip connections)
3. Double convolution (3 steps from above, twice)

After this the data undergoes a final 1x1 convolution and restack into a volume to get a 6 channel output (one for every label). The model has 1,942,374 parameters, fewer than the 3D models due to the 9 vs 27 kernel weights. 

The 2D model uses the same widths, loss, data, and steps as the 3D models. The loss is computed on the restacked volume. 

### 4.2 Baseline 3D U-Net

Same as 4.1 except: 
- The standard 3D baseline from Çiçek et al. (2016), the established reference for volumetric segmentation that the Improved model is compared against.
- Processes the whole volume at once: (B, 1, 128, 128, 64) giving inter-slice context.
- Encoder has 3x3x3 convolutions with the same double convolution structure as the 2D model.
- a 2x2x2 max-pool between encoder levels halves all three axes, so the bottleneck is (B, 256, 8, 8, 4). 
- Decoder has the same 4 levels with 2x2x2 transposed convolutions, concatenated skips, then the double convolution.
- Output is a 1x1x1 convolution to 6 channels: (B, 6, 128, 128, 64), there is no restacking since it is already a volume.
- Parameters: 5,646,470 (more than 2D because each 3×3×3 kernel has 27 weights vs 9).
- Unlike the model from Çiçek et al. the model uses InstanceNorm instead of BatchNorm, widths starting at 16 instead of 32, and 4 pooling levels instead of 3. 

Because the 3x3x3 kernels and the 2x2x2 pooling span the slice axis, the model can use information from neighbouring slices, which is the property the 2D control lacks. 

### 4.3 Improved 3D U-Net






## 5. Training Setup 

The models use the Adam optimser (`lr=1e-3`) and loss function  Dice + cross-entropy (CE) (`DiceCELoss` in `modules.py`): CE + (1 - soft Dice averaged over the 6 classes). CE gives smooth, stable gradients from epoch 1, and its bias towards the background is mediated by the Soft Dice, which weights each class equally. Together this gives stable initial training and class balance. The Adam's momentum plus per-weight adaptive step size is standard for U-Nets and robust without tuning. Further, the learning rate scheudles is cosine annealing -> 0 over 150 epochs, reducing step size to allow model to settle to a minima. The best validation prostate Dice score over the 150 epoch training was saved as the final checkpoint (~70 epochs). Precision was bf16 autocast, loss computed in fp32 to be time and memory efficient. Training seeds 1 / 2 / 42 across all models show run-to-run variation on the same patients. 

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