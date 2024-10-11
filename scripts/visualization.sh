#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --job-name=Jupyter
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=8 --mem=10000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=jupyter_%j.out
#Skipping many options! see man sbatch
# From here on, we can start our program


~/miniconda3/envs/deeploc_torch/bin/jupyter-notebook --host=127.0.0.1 --port=8801 --no-browser