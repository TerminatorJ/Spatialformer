#!/bin/bash
#SBATCH --partition=largemem
#SBATCH --job-name=h5toloader
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --mem=600G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=h5toloader_%A_%a.out
#SBATCH --error=h5toloader_%A_%a.err
#SBATCH --array=0-25
set -euo pipefail

source activate spatialformer

# AVAILABLE_DATANAMES=(
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs.h5ad"
#     # "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs.h5ad"
#     # "/scratch/project_465001820/Spatialfoqrmer/data/processed/Xenium_Prime_Breast_Cancer_FFPE_xe_outs.h5ad"
#     # "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Skin_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Prostate_FFPE_xe_outs.h5ad"
#     # "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs.h5ad"
#     # "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs.h5ad"
# )

# cell_num=(
#     295805
#     463596
#     642955
#     695740
#     268378
# )


# AVAILABLE_DATANAMES=(
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Breast_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Breast_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Breast_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Breast_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Breast_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs.h5ad"
# )


# CHUNK=(
#     100000
# )

# PARTITION=(
#     1
#     2
#     3
#     1
#     2
#     3
#     4
#     5
#     1
#     2
#     3
#     4
#     5
#     6
#     7
#     1
#     2
#     3
#     4
#     5
#     6
#     7
#     1
#     2
#     3
# )

# python /scratch/project_465001820/Spatialformer/data_preprocessing/h5toloader.py \
#         --data_path ${AVAILABLE_DATANAMES[$SLURM_ARRAY_TASK_ID]} \
#         --chunk ${CHUNK} \
#         --partition ${PARTITION[$SLURM_ARRAY_TASK_ID]}



# AVAILABLE_DATANAMES=(
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs.h5ad"
#     # "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs.h5ad"
#     # "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Breast_Cancer_FFPE_xe_outs.h5ad"
#     # "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Skin_FFPE_xe_outs.h5ad"
#     "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Prostate_FFPE_xe_outs.h5ad"
#     # "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs.h5ad"
#     # "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs.h5ad"
# )

# python /scratch/project_465001820/Spatialformer/data_preprocessing/h5toloader.py \
#         --data_path ${AVAILABLE_DATANAMES[$SLURM_ARRAY_TASK_ID]} 


CHUNK=(
    50000
)
PARTITION=($(seq 1 26))


   
python /scratch/project_465001820/Spatialformer/data_preprocessing/h5toloader.py \
        --data_path "/scratch/project_465001820/Spatialformer/data/processed/Xenium_Prime_Human_Ovary_FF_xe_outs.h5ad" \
        --chunk ${CHUNK} \
        --partition ${PARTITION[$SLURM_ARRAY_TASK_ID]}


