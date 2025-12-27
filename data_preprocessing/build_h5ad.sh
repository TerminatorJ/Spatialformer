#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --job-name=build_h5ad
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=400G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=build_h5ad_%A_%a.out
#SBATCH --error=build_h5ad_%A_%a.err
#SBATCH --array=0-0
set -euo pipefail

source activate spatialformer



# DATANAME=(
#     "Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs"
#     "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
#     "Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
#     # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs/transcript_processed"
#     "Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
#     "Xenium_Prime_Human_Skin_FFPE_xe_outs"
#     "Xenium_Prime_Human_Prostate_FFPE_xe_outs"
#     "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
#     "Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs"
# )

DATANAME=(
    "Xenium_Prime_Human_Ovary_FF_xe_outs"
)
CONDITION=(
    "Disease"
)
TISSUES=(
    "Ovary"
)
SPECIES=(
    "Human"
)
SPECIES=(
    "Human"
)
ASSAY=(
    "Xenium"
)

# CONDITION=(
#     "Disease"
#     "Disease"
#     "Disease"
#     "Disease"
#     "Disease"
#     "Disease"
#     "Disease"
#     "Disease"
# )

# TISSUES=(
#     "Ovary"
#     "Ovary"
#     "Breast"
#     "Cervix"
#     "Skin"
#     "Prostate"
#     "LymphNode"
#     "Lung"
# )

# SPECIES=(
#     "Human"
#     "Human"
#     "Human"
#     "Human"
#     "Human"
#     "Human"
#     "Human"
#     "Human"
# )

# ASSAY=(
#     "Xenium"
#     "Xenium"
#     "Xenium"
#     "Xenium"
#     "Xenium"
#     "Xenium"
#     "Xenium"
#     "Xenium"
# )



echo "============================================"
echo "Array task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_MAX}"
echo "DATANAME ${DATANAME[$SLURM_ARRAY_TASK_ID]}"
echo "CONDITION ${CONDITION[$SLURM_ARRAY_TASK_ID]}"
echo "TISSUES ${TISSUES[$SLURM_ARRAY_TASK_ID]}"
echo "============================================"


python /scratch/project_465001820/Spatialformer/data_preprocessing/build_h5ad.py \
     --data_name ${DATANAME[$SLURM_ARRAY_TASK_ID]} \
     --condition ${CONDITION[$SLURM_ARRAY_TASK_ID]} \
     --tissues ${TISSUES[$SLURM_ARRAY_TASK_ID]} \
     --species ${SPECIES[$SLURM_ARRAY_TASK_ID]} \
     --assay ${ASSAY[$SLURM_ARRAY_TASK_ID]} \
     --datapath_name "xenium_5k" \
    --datapath "/scratch/project_465001820/Spatialformer/data/"

echo "Job finished successfully!"