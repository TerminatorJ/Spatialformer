#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu
#SBATCH --job-name=build_h5ad
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=8 --mem=150000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=build_h5ad_%j.out

batch_file=$1


DIRECTORY="/tmp/erda"
sh /home/sxr280/Spatialformer/scripts/unmount_erda.sh
# Find all regular files in the directory. This includes hidden files by default.
FILE_COUNT=$(find "$DIRECTORY" -type d | wc -l)
ls -la "$DIRECTORY"
if [ "$FILE_COUNT" -eq 4 ]; then
    rm -rf /tmp/erda/Spatialformer
else
    echo "The directory '$DIRECTORY' contains $FILE_COUNT files."
fi

sh /home/sxr280/Spatialformer/scripts/mount_erda.sh
# Run each command in the specified batch file
while IFS= read -r cmd
do
  echo "Running command: $cmd"
  eval $cmd
done < "$batch_file"

sh /home/sxr280/Spatialformer/scripts/unmount_erda.sh