#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --partition=largemem
#SBATCH --job-name=run_perturbation
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=200G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1-00:00:00
#SBATCH --output=run_perturbation_%j.out
#SBATCH --error=run_perturbation_%j.err
#SBATCH --array=0-23

source activate spatialformer

DATANAME=(
    "lung_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "lung_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "brain_transcripts.csv.gz"
    "brain_transcripts.csv.gz"

)
OUTPUTNAME=(
    "lung_results_bd_noise_c_3.pkl"
    "lung_results_bd_noise_nc_3.pkl"
    "lung_results_bd_noise_c_4.pkl"
    "lung_results_bd_noise_nc_4.pkl"
    "lung_results_bd_noise_c_5.pkl"
    "lung_results_bd_noise_nc_5.pkl"
    "brain_results_bd_noise_c_3.pkl"
    "brain_results_bd_noise_nc_3.pkl"
    "brain_results_bd_noise_c_4.pkl"
    "brain_results_bd_noise_nc_4.pkl"
    "brain_results_bd_noise_c_5.pkl"
    "brain_results_bd_noise_nc_5.pkl"
    "lung_results_nbd_noise_c_3.pkl"
    "lung_results_nbd_noise_nc_3.pkl"
    "lung_results_nbd_noise_c_4.pkl"
    "lung_results_nbd_noise_nc_4.pkl"
    "lung_results_nbd_noise_c_5.pkl"
    "lung_results_nbd_noise_nc_5.pkl"
    "brain_results_nbd_noise_c_3.pkl"
    "brain_results_nbd_noise_nc_3.pkl"
    "brain_results_nbd_noise_c_4.pkl"
    "brain_results_nbd_noise_nc_4.pkl"
    "brain_results_nbd_noise_c_5.pkl"
    "brain_results_nbd_noise_nc_5.pkl"
    

)

CLUSTERING=(
    "--clustering"
    ""
    "--clustering"
    ""
    "--clustering"
    ""
    "--clustering"
    ""
    "--clustering"
    ""
    "--clustering"
    ""
    "--clustering"
    ""
    "--clustering"
    ""
    "--clustering"
    ""
    "--clustering"
    ""
    "--clustering"
    ""
    "--clustering"
    ""
)

BOUNDARY=(
    "--boundary_noise"
    "--boundary_noise"
    "--boundary_noise"
    "--boundary_noise"
    "--boundary_noise"
    "--boundary_noise"
    "--boundary_noise"
    "--boundary_noise"
    "--boundary_noise"
    "--boundary_noise"
    "--boundary_noise"
    "--boundary_noise"
    ""
    ""
    ""
    ""
    ""
    ""
    ""
    ""
    ""
    ""
    ""
    "" 
)

RADIUS=(
    3
    3
    4
    4
    5
    5
    3
    3
    4
    4
    5
    5
    3
    3
    4
    4
    5
    5
    3
    3
    4
    4
    5
    5
)

IDX=${SLURM_ARRAY_TASK_ID}
python /scratch/project_465001820/Spatialformer/downstream/Coordinate_perturbation/coordinate_perturbation.py \
    --data_name "${DATANAME[$IDX]}" \
    --output_name "${OUTPUTNAME[$IDX]}" \
    ${CLUSTERING[$IDX]} \
    ${BOUNDARY[$IDX]} \
    --radius ${RADIUS[$IDX]}