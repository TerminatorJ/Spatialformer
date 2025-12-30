#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --partition=standard-g
#SBATCH --job-name=Spatialformer_embeddings
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G  # Request all memory on the node
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=Spatialformer_embeddings_%j.out
#SBATCH --error=Spatialformer_embeddings_%j.err

# get tunneling info


port=12345
node=$(hostname -s)
user=$(whoami)

# run jupyter notebook
# If you have a custom environment, active it first here
jupyter-notebook --no-browser --port=${port} --ip=${node}