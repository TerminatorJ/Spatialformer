#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --partition=small-g
#SBATCH --job-name=jupyter_notebook
#SBATCH --account=project_465001027
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=jupyter_notebook_%j.out
#SBATCH --error=jupyter_notebook_%j.err


port=8848
node=$(hostname -s)
user=$(whoami)

# run jupyter notebook
# If you have a custom environment, active it first here
/scratch/project_465001027/deeploc_torch/bin/jupyter-notebook --no-browser --port=${port} --ip=${node}