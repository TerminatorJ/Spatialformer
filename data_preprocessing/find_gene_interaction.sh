#!/bin/bash
#SBATCH --partition=small
#SBATCH --job-name=find_gene_interaction
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --mem=200G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00
#SBATCH --output=find_gene_interaction_%A_%a.out
#SBATCH --error=find_gene_interaction_%A_%a.err

set -euo pipefail

source activate spatialformer


TRANSCRIPT_FILE=(
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs/transcript_processed"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs/transcript_processed"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs/transcript_processed"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs/transcript_processed"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs/transcript_processed"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Skin_FFPE_xe_outs/transcript_processed"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs/transcript_processed"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs/transcript_processed"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs/transcript_processed"
)


CHUNKS=20000

CELL_NUM=(
    200900
    407124
    699110
    # 1157659
    840387
    112551
    193000
    708983
    278328
)
# Calculate which dataset and partition this array task should process
# Build a mapping: array_index -> (dataset_idx, partition_num)
TASK_MAP=()
for dataset_idx in "${!CELL_NUM[@]}"; do
    num_cells=${CELL_NUM[$dataset_idx]}
    num_partitions=$(( (num_cells + CHUNKS - 1) / CHUNKS ))  # Ceiling division
    
    for partition in $(seq 1 $num_partitions); do
        TASK_MAP+=("${dataset_idx}:${partition}")
    done
done


TOTAL_TASKS=${#TASK_MAP[@]}
MAX_ARRAY_INDEX=$((TOTAL_TASKS - 1))

# If this is the initial submission (no SLURM_ARRAY_TASK_ID set),
# resubmit with the correct --array range
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "============================================"
    echo "Initial run detected - calculating array size"
    echo "Total tasks needed: ${TOTAL_TASKS}"
    echo "Array range: 0-${MAX_ARRAY_INDEX}"
    echo "============================================"
    echo ""
    echo "Submitting job array with --array=0-${MAX_ARRAY_INDEX}..."
    
    # Resubmit this script with the correct array parameter
    sbatch --array=0-${MAX_ARRAY_INDEX} "$0"
    
    echo "Job submitted. Exiting initial shell."
    exit 0
fi

# ==========================================
# NORMAL EXECUTION (when running as array job)
# ==========================================


# Validate array task ID
if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL_TASKS} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} exceeds task map size ${TOTAL_TASKS}"
    exit 1
fi

TASK_INFO="${TASK_MAP[$SLURM_ARRAY_TASK_ID]}"
DATASET_IDX="${TASK_INFO%%:*}"
PARTITION="${TASK_INFO##*:}"

echo "============================================"
echo "Array task ${SLURM_ARRAY_TASK_ID} of ${MAX_ARRAY_INDEX}"
echo "Dataset index: ${DATASET_IDX}"
echo "Partition: ${PARTITION}"
echo "Dataset: ${TRANSCRIPT_FILE[$DATASET_IDX]}"
echo "Total cells: ${CELL_NUM[$DATASET_IDX]}"
echo "Chunk size: ${CHUNKS}"
echo "============================================"

python /scratch/project_465001820/Spatialformer/data_preprocessing/find_gene_interaction.py \
     --transcript_file "${TRANSCRIPT_FILE[${DATASET_IDX}]}" \
     --partition ${PARTITION} \
     --chunks ${CHUNKS} \
     --dataname "$(basename "$(dirname "${TRANSCRIPT_FILE[${DATASET_IDX}]}")")"