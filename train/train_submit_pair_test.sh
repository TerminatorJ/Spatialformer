#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --partition=standard-g
#SBATCH --job-name=train_pair_model1
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

#SBATCH --nodes=4
#SBATCH --mem=300G  # Request all memory on the node
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=train_pair_model1_%j.out
#SBATCH --error=train_pair_model1_%j.err


source activate spatialformer_flash_attn
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_TIMEOUT=600  
export NCCL_IB_TIMEOUT=120 
export TORCH_DISTRIBUTED_DEBUG=DETAIL  
export NCCL_ASYNC_ERROR_HANDLING=1
export HOME=$HOME
export USER=$(whoami)
export WANDB_API_KEY="57f4851d7943ea1dec3b10273876045d051b40f1"  # Get from https://wandb.ai/authorize


export MPICH_GPU_SUPPORT_ENABLED=1

cd /scratch/project_465001820/Spatialformer
srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python -m  train.train2