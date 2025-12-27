#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --job-name=xenium_process
#SBATCH --account=project_465001820
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=80G
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00
#SBATCH --output=xenium_5k_process_%A_%a.out
#SBATCH --error=xenium_5k_process_%A_%a.err
#SBATCH --array=0-167


set -euo pipefail

source activate spatialformer


OUTPUT=(
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Skin_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Skin_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Skin_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Skin_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Skin_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Skin_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs"
    # "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
    "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Ovary_FF_xe_outs"
)


PARTITIONS=()
PARTITION=()
for ((i=0; i<168; i++)); do
    PARTITIONS+=(168)
    PARTITION+=($(seq 0 167))
done

# for ((i=0; i<18; i++)); do
#     PARTITIONS+=(18)
#     PARTITION+=($(seq 0 17))
# done

# for ((i=0; i<24; i++)); do
#     PARTITIONS+=(24)
#     PARTITION+=($(seq 0 23))
# done

# for ((i=0; i<6; i++)); do
#     PARTITIONS+=(6)
#     PARTITION+=($(seq 0 5))
# done

# for ((i=0; i<6; i++)); do
#     PARTITIONS+=(6)
#     PARTITION+=($(seq 0 5))
# done

# for ((i=0; i<6; i++)); do
#     PARTITIONS+=(6)
#     PARTITION+=($(seq 0 5))
# done

# for ((i=0; i<12; i++)); do
#     PARTITIONS+=(12)
#     PARTITION+=($(seq 0 11))
# done

# for ((i=0; i<6; i++)); do
#     PARTITIONS+=(6)
#     PARTITION+=($(seq 0 5))
# done

# for ((i=0; i<84; i++)); do
#     PARTITIONS+=(84)
#     PARTITION+=($(seq 0 83))
# done



 



python /scratch/project_465001820/Spatialformer/data_preprocessing/xenium_5k_preprocess.py \
  --input_file "${OUTPUT[$SLURM_ARRAY_TASK_ID]}" \
  --workers 4 \
  --batch_size 1000000 \
  --save_batch_size 20000000 \
  --partitions ${PARTITIONS[$SLURM_ARRAY_TASK_ID]} \
  --partition ${PARTITION[$SLURM_ARRAY_TASK_ID]}