#!/bin/bash

batch_directory='/home/sxr280/Spatialformer/scripts/build_h5ad'

for batch_file in "$batch_directory"/*.sh; do
  echo $batch_file
  sbatch build_h5ad.sh "$batch_file"
done
