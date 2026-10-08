#!/bin/bash
#SBATCH --job-name=lob_train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=lob_train_%A_%a.out
#SBATCH --error=lob_train_%A_%a.err
#SBATCH --mail-user=s4961587@student.uq.edu.au
#SBATCH --mail-type=END,FAIL

# Usage: sbatch lob_train.sh timegan [extra train.py flags]
#   e.g. sbatch lob_train.sh timegan --constrained
#        sbatch lob_train.sh rnngan
# Job array: three tasks run in parallel, and each task's index (0, 1, 2) is used as the random seed.
MODEL=${1:-timegan}
shift                                           # anything after the model name is passed to train.py
echo "Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID: $MODEL $@ seed $SLURM_ARRAY_TASK_ID on $(hostname), started $(date)"
nvidia-smi
source $HOME/miniconda3/bin/activate
conda activate torch
cd $HOME/PatternAnalysis-2026/recognition/TimeGAN_LOB_s4961587
export PYTHONUNBUFFERED=1                       # live output in the .out file
torchrun --standalone --nproc_per_node=1 train.py --data_dir $HOME/data/LOBSTER \
    --model $MODEL --steps 5000 --eval_every 250 --seed $SLURM_ARRAY_TASK_ID "$@"
echo "Finished $(date)"