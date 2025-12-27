#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --job-name=data_download
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=8G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=data_download_%j.out
#SBATCH --error=data_download_%j.err
#SBATCH --array=0-1
set -euo pipefail

source activate spatialformer

URLS=(
# "https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_Prime_Human_Ovary_Cancer_FF/Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs.zip"   
# "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Lung_Cancer_FFPE/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs.zip"
# ”https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_V1_Human_Lung_Cancer_FFPE/Xenium_V1_Human_Lung_Cancer_FFPE_xe_outs.zip“
# "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Breast_Cancer_FFPE/Xenium_Prime_Breast_Cancer_FFPE_xe_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Human_Ovary_FF/Xenium_Prime_Human_Ovary_FF_xe_outs.zip"
# "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Cervical_Cancer_FFPE/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs.zip"
# "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Skin_FFPE/Xenium_Prime_Human_Skin_FFPE_xe_outs.zip"
# "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Prostate_FFPE/Xenium_Prime_Human_Prostate_FFPE_xe_outs.zip"
# "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs.zip"
)

python download.py --url "${URLS[$SLURM_ARRAY_TASK_ID]}" 