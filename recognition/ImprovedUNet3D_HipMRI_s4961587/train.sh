#!/bin/bash
# train.sh - Slurm job: train a model on one A100.
# Usage:  sbatch train.sh [train.py args]
#   smoke:     sbatch --time=00:20:00 train.sh --epochs 3 --limit 4 --tag smoke
#   baseline:  sbatch train.sh --model unet3d --epochs 150
#SBATCH --job-name=unet3d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=unet3d_%j.out
#SBATCH --error=unet3d_%j.err
#SBATCH --mail-user=s4961587@student.uq.edu.au
#SBATCH --mail-type=END,FAIL

ARGS=("$@")  # save train.py args, then clear them: conda's activate script would otherwise read them
set --

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
source $HOME/miniconda3/bin/activate
conda activate torch
cd $HOME/PatternAnalysis-2026/recognition/ImprovedUNet3D_HipMRI_s4961587
python -u train.py "${ARGS[@]}"
echo "finished $(date)"