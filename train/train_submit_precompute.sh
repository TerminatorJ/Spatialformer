#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --partition=small
#SBATCH --job-name=train_pair_precompute
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=300G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=04:00:00
#SBATCH --output=train_pair_precompute_%j.out
#SBATCH --error=train_pair_precompute_%j.err
#SBATCH --array=0-10


source activate spatialformer_flash_attn
# debugging flags (optional)
# PARTITION=($(seq 1 37))
PARTITION=()

cd /scratch/project_465001820/Spatialformer
python  /scratch/project_465001820/Spatialformer/spatialformer/dataloader/dataloader_paired.py --partitions 37 --partition ${PARTITION[$SLURM_ARRAY_TASK_ID]}