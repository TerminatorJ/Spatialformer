#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --job-name=rerank_dataset
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --mem=100G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=rerank_dataset_%A_%a.out
#SBATCH --error=rerank_dataset_%A_%a.err


set -euo pipefail

source activate spatialformer

python /scratch/project_465001820/Spatialformer/data_preprocessing/rerank_dataset.py