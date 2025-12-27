#!/bin/bash
#SBATCH --partition=largemem
#SBATCH --job-name=concat_xenium_5k
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=128
#SBATCH --nodes=1
#SBATCH --mem=500G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=concat_xenium_5k_%A_%a.out
#SBATCH --error=concat_xenium_5k_%A_%a.err

set -euo pipefail

source activate spatialformer

python /scratch/project_465001820/Spatialformer/data_preprocessing/concat_xenium_5k.py