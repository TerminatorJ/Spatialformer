import os 
import sys 
import h5py
import pandas as pd
import numpy as np
import scanpy as sc
import multiprocessing
import argparse
import itertools
from pathlib import Path
import random
import zarr
from scipy.sparse import csr_matrix
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]
sys.path.append(str(p_path))
sys.path.append(os.path.join(str(p_path), "utils"))
# from utils import *
import pickle
from utils import *
from tqdm import tqdm
import logging
import argparse
import glob
from xenium_5k_cell_feature import read_xenium_5k

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess(partitions : int = 6, 
               data_name : str = "Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs",
               transcript_threshold : int = 100,
               Condition : str = "Healthy",
               Tissues : str = "Lung",
               Species : str = "Human",
               Assay : str = "Xenium",
               datapath_name: str = "david_data",
               datapath: str = p_path 
               ):
    '''
     
    Preprocessing the raw dataset and send it to a h5ad file for further process
    Args:
        partitions: Number of partitions that are used to save the AnnData. An OOM error will raise if we concate all the transcript in one file
        data_name: The name of the dataset to distinguish different dataset we use.
        transcript_threshold: The minimum number of transcripts locates within the cells.
        condition: The healthy status of the sample, which can become optional: healthy, disease.
        Tissues: The tissue where the samples are collected from.
        Species: The species the sample belongs to.
        Assay: Which assay that was used to measure the gene expression in the spatial context.
        datapath_name: The name of the path that is used to store all the raw and processed data.
        
    '''
    
    #Getting the raw xenium dataset via the dataname
    if datapath_name == "david_data":
        raw_dir = os.path.join(datapath, datapath_name, data_name, "outs")
        data_dir = os.path.join(datapath, datapath_name)
        save_dir = os.path.join(datapath, datapath_name, data_name, "processed")
    else:
        raw_dir = os.path.join(datapath, "raw", data_name)
        data_dir = os.path.join(datapath, "processed", data_name)
        #saved path with training and validation dataset
        save_dir = os.path.join(datapath, "processed")
    os.makedirs(raw_dir, exist_ok = True)
    os.makedirs(data_dir, exist_ok = True)
    os.makedirs(save_dir, exist_ok = True)

    #find the cell_feature file
    prefix = "cell_feature_matrix"

    # Use glob to find any file starting with the prefix followed by anything
    search_pattern = os.path.join(raw_dir, f"{prefix}*")

    # This list will contain the full paths of all matching files
    found_files = glob.glob(search_pattern)

    if found_files:
        # We assume the first found file is the correct matrix
        matrix_file_path = found_files[0] 
        
        # 1. Print the file found
        print(f"Found matrix file: {matrix_file_path}")
        
        # 2. Extract the suffix
        # The os.path.basename isolates the file name (e.g., 'cell_feature_matrix.h5')
        filename = os.path.basename(matrix_file_path)
        
        # We can split the filename on the prefix to get the suffix part
        # We split only once (maxsplit=1) to ensure we capture '.zarr.zip'
        suffix = filename.split(prefix, 1)[1] 
        
        print(f" Extracted suffix: {suffix}")
        
    else:
        print(f"Error: No file found starting with '{prefix}' in '{raw_dir}'")
        matrix_file_path = None
        suffix = None
    
    if suffix == ".h5":
        adata = sc.read_10x_h5(f"{raw_dir}/cell_feature_matrix.h5") #10M
    elif suffix == ".zarr.zip":
        xenium_loader = read_xenium_5k(f"{raw_dir}/cell_feature_matrix.zarr.zip")
        
        adata = xenium_loader()
    #convert gene names to the columns and set gene id as index
    try:
        adata.var = adata.var.reset_index().rename(columns={'index': 'gene_name'}).set_index('gene_ids')
        adata.var.index.name = None
    except:
        adata.var = adata.var.reset_index().rename(columns={'feature_name': 'gene_name', "feature_id": "gene_ids"}).set_index('gene_ids')
        adata.var.index.name = None
    #transfer to sparse matrix
    adata.X = csr_matrix(adata.X)
    #adding additional information for the whole dataset
    adata.obs["Conditions"] = pd.Categorical([Condition for i in range(len(adata))])
    adata.obs["Tissues"] = pd.Categorical([Tissues for i in range(len(adata))])
    adata.obs["Species"] = pd.Categorical([Species for i in range(len(adata))])
    adata.obs["Assay"] = pd.Categorical([Assay for i in range(len(adata))])
    adata.obs["DataID"] = pd.Categorical([data_name for i in range(len(adata))])
    
    #filtering shoud be identified here, filtering the auxiliary genes Unassigned
    genes_mask = ~(adata.var["gene_name"].str.startswith('Neg') | 
                   adata.var["gene_name"].str.startswith('BLANK')| 
                   adata.var["gene_name"].str.startswith('Unassigned')|
                   adata.var["gene_name"].str.startswith('Deprecated')|
                   adata.var["gene_name"].str.startswith('Intergenic')|
                   adata.var["gene_name"].str.startswith('Total')|
                   adata.var["gene_name"].str.startswith('Human'))
    
    #using the transcript file to get the reference of the genes
    
    adata = adata[:, genes_mask]

    #merge all the .h5 file to a single file
    # List of input file paths
    matrix_name = data_name+"_gene_interaction"
    data_files = [os.path.join(os.path.abspath(save_dir),file) for file in os.listdir(save_dir) if data_name in file]
    matching_files = [file for file in data_files if matrix_name in file and "merge" not in file]
    # Create a new HDF5 file for merging
    # Load directly from partition files without merging
    print("Loading matrices directly from partition files...")
    cell_ids_needed = set(adata.obs.index) - {"bahfibck-1"}
    cell_id_to_matrix = {}  # Store matrices temporarily
    
    for h5_file_path in tqdm(matching_files, desc="Processing partition files"):
        with h5py.File(h5_file_path, 'r') as file:
            for cell_id in file.keys():
                if str(cell_id) in cell_ids_needed:
                    try:
                        int_matrix = read_h5(file, str(cell_id)).tocsr()
                        assert int_matrix.shape[0] == adata.n_vars, f"Matrix row count {int_matrix.shape[0]} does not match adata.n_vars {adata.n_vars} for cell {cell_id}"
                        cell_id_to_matrix[str(cell_id)] = int_matrix
                    except Exception as e:
                        print(f"Error loading {cell_id}: {e}")
                        continue
    # Add matrices to adata.uns in the order of adata.obs.index
    keep_id = []
    for cell_id in tqdm(list(adata.obs.index), desc="Adding matrices to adata.uns"):
        if cell_id != "bahfibck-1" and str(cell_id) in cell_id_to_matrix:
            adata.uns[cell_id] = cell_id_to_matrix[str(cell_id)]
            keep_id.append(cell_id)
        elif cell_id != "bahfibck-1":
            print(f"{cell_id} not in the h5 partition files")
    #saving the data into ".h5"
    adata = adata[keep_id, :]
    adata.write(f"{save_dir}/{data_name}.h5ad")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='embed all the info altogether')
    parser.add_argument('--partitions', type=int, default=1, help='The number of partitions that need to be integrated')
    parser.add_argument('--data_name', type=str, default=None, help='The datanames use to identify the raw file')
    parser.add_argument('--transcript_threshold', type=int, default=100, help='The number of transcripts locate within the cells')
    parser.add_argument('--condition', type=str, default="Healthy", help='The status of the sample')
    parser.add_argument('--tissues', type=str, default="Lung", help='The tissue where the sample is collected from')
    parser.add_argument('--species', type=str, default="Human", help='The species that the sample belongs to')
    parser.add_argument('--assay', type=str, default="Xenium", help='The technolegy that is used to measure the transcripts')
    parser.add_argument('--datapath_name', type=str, default="david_data", help= "The name of the path that is used to store all the raw and processed data")
    parser.add_argument('--datapath', type=str, default="/tmp/erda/Spatialformer/downloaded_data", help= "The path that is used to store all the processed data")
    args = parser.parse_args()
    
    preprocess(partitions = args.partitions, 
               data_name = args.data_name,
               transcript_threshold = args.transcript_threshold,
               Condition = args.condition,
               Tissues = args.tissues,
               Species = args.species,
               Assay = args.assay,
               datapath_name = args.datapath_name,
               datapath = args.datapath
               )

#how to read the data
#aa = sc.read_h5ad("/scratch/project_465001027/spatialformer/data/processed/Xenium_Preview_Human_Non_
# diseased_Lung_With_Add_on_FFPE_outs/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.h5ad")
# adata = sc.read_h5ad("/scratch/project_465001027/spatialformer/data/processed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.h5ad")

# THD0008
# python build_h5ad.py --partitions 3 --data_name relabel_output-XETG00048__0003392__THD0008__20230313__191400 --matrix_name THD0008_gene_interaction --condition Healthy --tissues Lung --species Human  --assay Xenium
# VUILD106
# python build_h5ad.py --partitions 6 --data_name relabel_output-XETG00048__0003392__VUILD106__20230313__191400 --matrix_name VUILD106_gene_interaction --condition Disease --tissues Lung --species Human --assay Xenium
# VUILD110
# python build_h5ad.py --partitions 6 --data_name relabel_output-XETG00048__0003392__VUILD110__20230313__191400 --matrix_name VUILD110_gene_interaction --condition Disease --tissues Lung --species Human --assay Xenium

# Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs
# python build_h5ad.py --partitions 5 --data_name Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs --matrix_name Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Skin --species Human --assay Xenium --transcript_threshold 30
# Xenium_V1_hLung_cancer_section_outs
# python build_h5ad.py --partitions 8 --data_name Xenium_V1_hLung_cancer_section_outs --matrix_name Xenium_V1_hLung_cancer_section_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Lung --species Human --assay Xenium --transcript_threshold 30
# Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs
# python build_h5ad.py --partitions 14 --data_name Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs --matrix_name Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Lung --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs
# python build_h5ad.py --partitions 2 --data_name Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs --matrix_name Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Brain --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon_outs
# python build_h5ad.py --partitions 3 --data_name Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon_outs --matrix_name Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Brain --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs
# python build_h5ad.py --partitions 10 --data_name Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs --matrix_name Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Pancreas --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_human_Pancreas_FFPE_outs
# python build_h5ad.py --partitions 5 --data_name Xenium_V1_human_Pancreas_FFPE_outs --matrix_name Xenium_V1_human_Pancreas_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Pancreas --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs
# python build_h5ad.py --partitions 2 --data_name Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs --matrix_name Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Skin --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs
# python build_h5ad.py --partitions 2 --data_name Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs --matrix_name Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Skin --species Human --assay Xenium --transcript_threshold 30
#Xenium_V1_hLiver_nondiseased_section_FFPE_outs
# python build_h5ad.py --partitions 12 --data_name Xenium_V1_hLiver_nondiseased_section_FFPE_outs --matrix_name Xenium_V1_hLiver_nondiseased_section_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Liver --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hHeart_nondiseased_section_FFPE_outs
# python build_h5ad.py --partitions 2 --data_name Xenium_V1_hHeart_nondiseased_section_FFPE_outs --matrix_name Xenium_V1_hHeart_nondiseased_section_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Heart --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hSkin_Melanoma_Base_FFPE_outs
# python build_h5ad.py --partitions 6 --data_name Xenium_V1_hSkin_Melanoma_Base_FFPE_outs --matrix_name Xenium_V1_hSkin_Melanoma_Base_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Skin --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hColon_Non_diseased_Base_FFPE_outs
# python build_h5ad.py --partitions 12 --data_name Xenium_V1_hColon_Non_diseased_Base_FFPE_outs --matrix_name Xenium_V1_hColon_Non_diseased_Base_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Colon --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs
# python build_h5ad.py --partitions 14 --data_name Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs --matrix_name Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Colon --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_humanLung_Cancer_FFPE_outs
# python build_h5ad.py --partitions 7 --data_name Xenium_V1_humanLung_Cancer_FFPE_outs --matrix_name Xenium_V1_humanLung_Cancer_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Lung --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs
# python build_h5ad.py --partitions 11 --data_name Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs --matrix_name Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Ovary --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_Human_Lung_Cancer_Addon_FFPE_outs
# python build_h5ad.py --partitions 8 --data_name Xenium_V1_Human_Lung_Cancer_Addon_FFPE_outs --matrix_name Xenium_V1_Human_Lung_Cancer_Addon_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Lung --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs
# python build_h5ad.py --partitions 9 --data_name Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs --matrix_name Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Pancreas --species Human --assay Xenium --transcript_threshold 30
#Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs
# python build_h5ad.py --partitions 7 --data_name Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs --matrix_name Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues BoneMarrow --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hBoneMarrow_nondiseased_section_outs
# python build_h5ad.py --partitions 1 --data_name Xenium_V1_hBoneMarrow_nondiseased_section_outs --matrix_name Xenium_V1_hBoneMarrow_nondiseased_section_outs_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues BoneMarrow --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hBone_nondiseased_section_outs
# python build_h5ad.py --partitions 1 --data_name Xenium_V1_hBone_nondiseased_section_outs --matrix_name Xenium_V1_hBone_nondiseased_section_outs_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Bone --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_FFPE_Human_Breast_ILC_With_Addon_outs
# python build_h5ad.py --partitions 18 --data_name Xenium_V1_FFPE_Human_Breast_ILC_With_Addon_outs --matrix_name Xenium_V1_FFPE_Human_Breast_ILC_With_Addon_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Breast --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_FFPE_Human_Breast_ILC_outs
# python build_h5ad.py --partitions 18 --data_name Xenium_V1_FFPE_Human_Breast_ILC_outs --matrix_name Xenium_V1_FFPE_Human_Breast_ILC_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Breast --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hKidney_cancer_section_outs
# python build_h5ad.py --partitions 2 --data_name Xenium_V1_hKidney_cancer_section_outs --matrix_name Xenium_V1_hKidney_cancer_section_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Kidney --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon_outs
# python build_h5ad.py --partitions 2 --data_name Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon_outs --matrix_name Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Brain --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs
# python build_h5ad.py --partitions 29 --data_name Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs --matrix_name Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Breast --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs
# python build_h5ad.py --partitions 38 --data_name Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs --matrix_name Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Tonsil --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hLiver_cancer_section_FFPE_outs
# python build_h5ad.py --partitions 8 --data_name Xenium_V1_hLiver_cancer_section_FFPE_outs --matrix_name Xenium_V1_hLiver_cancer_section_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Liver --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hColon_Cancer_Base_FFPE_outs
# python build_h5ad.py --partitions 28 --data_name Xenium_V1_hColon_Cancer_Base_FFPE_outs --matrix_name Xenium_V1_hColon_Cancer_Base_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Colon --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hLymphNode_nondiseased_section_outs
# python build_h5ad.py --partitions 17 --data_name Xenium_V1_hLymphNode_nondiseased_section_outs --matrix_name Xenium_V1_hLymphNode_nondiseased_section_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues LymphNode --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_outs
# python build_h5ad.py --partitions 18 --data_name Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_outs --matrix_name Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Colon --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_FFPE_Human_Breast_IDC_Big_2_outs
# python build_h5ad.py --partitions 42 --data_name Xenium_V1_FFPE_Human_Breast_IDC_Big_2_outs --matrix_name Xenium_V1_FFPE_Human_Breast_IDC_Big_2_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Breast --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs
# python build_h5ad.py --partitions 29 --data_name Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs --matrix_name Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Breast --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_FFPE_Human_Breast_IDC_outs
# python build_h5ad.py --partitions 28 --data_name Xenium_V1_FFPE_Human_Breast_IDC_outs --matrix_name Xenium_V1_FFPE_Human_Breast_IDC_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Breast --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hPancreas_nondiseased_section_outs
# python build_h5ad.py --partitions 6 --data_name Xenium_V1_hPancreas_nondiseased_section_outs --matrix_name Xenium_V1_hPancreas_nondiseased_section_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Pancreas --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hKidney_nondiseased_section_outs
# python build_h5ad.py --partitions 4 --data_name Xenium_V1_hKidney_nondiseased_section_outs --matrix_name Xenium_V1_hKidney_nondiseased_section_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Healthy --tissues Kidney --species Human --assay Xenium --transcript_threshold 30

#Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs
# python build_h5ad.py --partitions 26 --data_name Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs --matrix_name Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Lung --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_FFPE_Human_Breast_IDC_Big_1_outs
# python build_h5ad.py --partitions 42 --data_name Xenium_V1_FFPE_Human_Breast_IDC_Big_1_outs --matrix_name Xenium_V1_FFPE_Human_Breast_IDC_Big_1_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Breast --species Human --assay Xenium --transcript_threshold 30


#Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs
# python build_h5ad.py --partitions 26 --data_name Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs --matrix_name Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Lung --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs
# python build_h5ad.py --partitions 61 --data_name Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs --matrix_name Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Tonsil --species Human --assay Xenium --transcript_threshold 30


#Xenium_V1_hColon_Cancer_Add_on_FFPE_outs
# python build_h5ad.py --partitions 28 --data_name Xenium_V1_hColon_Cancer_Add_on_FFPE_outs --matrix_name Xenium_V1_hColon_Cancer_Add_on_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Colon --species Human --assay Xenium --transcript_threshold 30

#Xenium_V1_Human_Brain_GBM_FFPE_outs
# python build_h5ad.py --partitions 41 --data_name Xenium_V1_Human_Brain_GBM_FFPE_outs --matrix_name Xenium_V1_Human_Brain_GBM_FFPE_outs_gene_interaction --datapath_name pandata --datapath /tmp/erda/Spatialformer/downloaded_data --condition Disease --tissues Brain --species Human --assay Xenium --transcript_threshold 30
