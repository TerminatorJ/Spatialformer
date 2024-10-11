#!/bin/bash

batch_directory='/home/sxr280/Spatialformer/scripts/find_gene_interaction'

for batch_file in "$batch_directory"/*.sh; do
  echo $batch_file
  sbatch find_gene_interaction.sh "$batch_file"
done
