#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --job-name=find_gene_interaction_oom
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --mem=400G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00
#SBATCH --output=find_gene_interaction_oom_%A_%a.out
#SBATCH --error=find_gene_interaction_oom_%A_%a.err
#SBATCH --array=0-23
set -euo pipefail

source activate spatialformer



# CHUNKS=20000

# PARQUET_PARTITION=($(for i in {1..60}; do echo $(((i + 1) / 2)); done))

# PARTITION=($(for i in {0..59}; do echo $((($i % 2) + 1)); done))

CHUNKS=10000
PARQUET_PARTITION=(
     1
     1
     1
     1
     2
     2
     2
     2
     3
     3
     3
     3
     8
     8
     8
     8
     11
     11
     11
     11
     12
     12
     12
     12
)

PARTITION=(
     1
     2
     3
     4
     1
     2
     3
     4
     1
     2
     3
     4
     1
     2
     2
     3
     1
     2
     3
     4
     1
     2
     3
     4
)

echo "============================================"
echo "Array task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_MAX}"
echo "PARTITION ${PARTITION[$SLURM_ARRAY_TASK_ID]}"
echo "PARQUET PARTITION ${PARQUET_PARTITION[$SLURM_ARRAY_TASK_ID]}"
echo "CHUNK ${CHUNKS}"
echo "============================================"
#get the cell parquet for sorting the cells
# python /scratch/project_465001820/Spatialformer/data_preprocessing/find_gene_interaction_oom.py \
#      --transcript_file "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs/transcript_processed" \
#      --partition ${PARTITION[$SLURM_ARRAY_TASK_ID]} \
#      --chunks ${CHUNKS} \
#      --dataname Xenium_Prime_Human_Ovary_FF_xe_outs

#get the gene-gene co-occurrence
python /scratch/project_465001820/Spatialformer/data_preprocessing/find_gene_interaction.py \
     --transcript_file "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs/cell_parquet_files" \
     --parquet_partition ${PARQUET_PARTITION[$SLURM_ARRAY_TASK_ID]} \
     --partition ${PARTITION[$SLURM_ARRAY_TASK_ID]} \
     --chunks ${CHUNKS} \
     --dataname Xenium_Prime_Human_Ovary_FF_xe_outs



