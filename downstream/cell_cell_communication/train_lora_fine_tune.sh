#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --partition=standard-g
#SBATCH --job-name=fine_tune_pair_model1
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=0  # Request all memory on the node
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=5:00:00
#SBATCH --output=fine_tune_pair_model1_%j.out
#SBATCH --error=fine_tune_pair_model1_%j.err


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

cd /scratch/project_465001820/Spatialformer/downstream/cell_cell_communication
srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode lora --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_metadata.csv --sample_name MERFISH_Lung --zero_shot_cell_size 500 --tissue Lung --condition Disease --checkpoint /scratch/project_465001820/Spatialformer/output/checkpoints/stepstep=0176000-traintrain_total_loss=-2.8414-valval_total_loss=0.0000.ckpt --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 10000