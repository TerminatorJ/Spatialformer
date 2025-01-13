#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --partition=standard-g
#SBATCH --job-name=train_pair_model4
#SBATCH --account=project_465001027
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=train_pair_model4_%j.out
#SBATCH --error=train_pair_model4_%j.err



source /scratch/project_465001027/deeploc_torch/bin/activate
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_TIMEOUT=300
srun /scratch/project_465001027/deeploc_torch/bin/python /scratch/project_465001027/Spatialformer/scripts/train.py