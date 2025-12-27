#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --job-name=get_gene_median
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --mem=200G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=get_gene_median_%A_%a.out
#SBATCH --error=get_gene_median_%A_%a.err
#SBATCH --array=0-7
set -euo pipefail

source activate spatialformer

AVAILABLE_DATANAMES=(
    # "Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs"
    # "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    "Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    "Xenium_Prime_Human_Skin_FFPE_xe_outs"
    "Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    "Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs"
)

python /scratch/project_465001820/Spatialformer/data_preprocessing/get_gene_median.py \
        --dataname ${AVAILABLE_DATANAMES[$SLURM_ARRAY_TASK_ID]}