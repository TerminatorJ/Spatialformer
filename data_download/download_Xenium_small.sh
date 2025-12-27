#!/bin/bash
#SBATCH --partition=standard
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
#SBATCH --array=0-37
set -euo pipefail

source activate spatialformer

URLS=(
"https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hSkin_nondiseased_section_2_FFPE/Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hLiver_cancer_section_FFPE/Xenium_V1_hLiver_cancer_section_FFPE_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.9.0/Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE/Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Prostate_FFPE/Xenium_Prime_Human_Prostate_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Skin_FFPE/Xenium_Prime_Human_Skin_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Mouse_Brain_Coronal_FF/Xenium_Prime_Mouse_Brain_Coronal_FF_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Cervical_Cancer_FFPE/Xenium_Prime_Cervical_Cancer_FFPE_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE/Xenium_Prime_Ovarian_Cancer_FFPE_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Human_Ovary_FF/Xenium_Prime_Human_Ovary_FF_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.3.0/Xenium_V1_FFPE_Human_Breast_IDC_With_Addon/Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section/Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Pancreas_FFPE/Xenium_V1_human_Pancreas_FFPE_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.9.0/Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE/Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE/Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_xe_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hLymphNode_nondiseased_section/Xenium_V1_hLymphNode_nondiseased_section_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hLung_cancer_section/Xenium_V1_hLung_cancer_section_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_mouse_Colon_FF/Xenium_V1_mouse_Colon_FF_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hPancreas_nondiseased_section/Xenium_V1_hPancreas_nondiseased_section_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.0.2/Xenium_V1_FF_Mouse_Brain_Coronal/Xenium_V1_FF_Mouse_Brain_Coronal_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hHeart_nondiseased_section_FFPE/Xenium_V1_hHeart_nondiseased_section_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hSkin_Melanoma_Base_FFPE/Xenium_V1_hSkin_Melanoma_Base_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hColon_Non_diseased_Base_FFPE/Xenium_V1_hColon_Non_diseased_Base_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hColon_Non_diseased_Add_on_FFPE/Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.6.0/Xenium_V1_hColon_Cancer_Base_FFPE/Xenium_V1_hColon_Cancer_Base_FFPE_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.6.0/Xenium_V1_hColon_Cancer_Add_on_FFPE/Xenium_V1_hColon_Cancer_Add_on_FFPE_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.6.0/Xenium_V1_mouse_pup/Xenium_V1_mouse_pup_outs.zip"
"https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.3.0/Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE/Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon/Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon/Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE/Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE/Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs.zip"
"https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Lung_Cancer_Addon_FFPE/Xenium_V1_Human_Lung_Cancer_Addon_FFPE_outs.zip"
)

python download.py --url "${URLS[$SLURM_ARRAY_TASK_ID]}" 