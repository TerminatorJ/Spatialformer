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
from scipy.sparse import csr_matrix
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]

sys.path.append("p_path")
# from utils import *
import pickle
from utils import *
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess(partitions : int = 6, 
               data_name : str = "Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs",
               matrix_name : str = None,
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
        matrix_name: The datanames use to identify the gene-gene interaction file
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
    #Processing the genes and cells
    adata = sc.read_10x_h5(f"{raw_dir}/cell_feature_matrix.h5") #10M
    
    adata.var = adata.var.reset_index().rename(columns={'index': 'gene_name'}).set_index('gene_ids')
    adata.var.index.name = None
    #transfer to sparse matrix
    adata.X = csr_matrix(adata.X)
    #adding additional information for the whole dataset
    adata.obs["Conditions"] = pd.Categorical([Condition for i in range(len(adata))])
    adata.obs["Tissues"] = pd.Categorical([Tissues for i in range(len(adata))])
    adata.obs["Species"] = pd.Categorical([Species for i in range(len(adata))])
    adata.obs["Assay"] = pd.Categorical([Assay for i in range(len(adata))])
    adata.obs["DataID"] = pd.Categorical([data_name for i in range(len(adata))])
    #adding the filtering information
    #The cells that are filtered should be identified here according to the output of the transcript.csv
    # import pdb; pdb.set_trace()
    try:
        transcript_df = pd.read_csv(f"{raw_dir}/transcripts.csv") #2G
    except:
        transcript_df = pd.read_csv(f"{raw_dir}/transcripts.csv.gz") #2G
    transcript_df.rename(columns={'x_location': 'x', 'y_location':'y', 'z_location':'z', 'feature_name':'gene'}, inplace=True)
    value_counts = transcript_df['cell_id'].value_counts()
    # import pdb; pdb.set_trace()
    try:
        value_counts = value_counts.drop("UNASSIGNED")
    except:
        value_counts = value_counts.drop(-1)
    # import pdb; pdb.set_trace()
    kept_cell_id_unique = transcript_df[transcript_df['cell_id'].isin(value_counts.index[value_counts >= transcript_threshold])]['cell_id'].unique().astype(str)
    #filtering the cells match the threshold
    adata = adata[kept_cell_id_unique, :]
    
    #filtering shoud be identified here, filtering the auxiliary genes Unassigned
    genes_mask = ~(adata.var["gene_name"].str.startswith('Neg') | adata.var["gene_name"].str.startswith('BLANK')| adata.var["gene_name"].str.startswith('Unassigned'))
    adata = adata[:, genes_mask]

    #split the dataset and then attach the split tags
    adata = split_data(adata, train_proportion=0.64, test_proportion=0.2, validation_proportion=0.16)

    #getting the compartment information for the downstream verification
    #TODO: getting the nucleus and cytoplasm info
    # adata.obs["Compartments"] = 'nuclus'


    #merge all the .h5 file to a single file
    # List of input file paths
    # h5_file_paths = [h5_file_path.split(".")[0][:-1] + str(partition) +"."+h5_file_path.split(".")[1] for partition in range(1, partitions+1)]
    data_files = [os.path.join(os.path.abspath(save_dir),file) for file in os.listdir(save_dir) if data_name in file]
    # import pdb; pdb.set_trace()
    matching_files = [file for file in data_files if matrix_name in file and "merge" not in file]
    
    merge_file_path = os.path.join(data_dir, matrix_name + "_merged.h5")
    # import pdb; pdb.set_trace()
    # Create a new HDF5 file for merging
    if not os.path.exists(merge_file_path):
        print(f"{merge_file_path} is not exists, running the code to merge all the partitions")
        with h5py.File(merge_file_path, "w") as merged_file:
            for h5_file_path in tqdm(matching_files):
                with h5py.File(h5_file_path, "r") as input_file:
                    # Copy datasets from input file to merged file
                    for cell_id in tqdm(list(input_file.keys())):
                        # import pdb; pdb.set_trace()
                        old_grp = input_file[cell_id]
                        new_grp = merged_file.create_group(str(cell_id))
                        new_grp.create_dataset('data', data=list(old_grp["data"]))
                        new_grp.create_dataset('row', data=list(old_grp["row"]))
                        new_grp.create_dataset('col', data=list(old_grp["col"]))
                        new_grp.attrs['shape'] = old_grp.attrs['shape']
    print(f"Merged datasets from {len(matching_files)} files into {merge_file_path}")

    #open the h5 file and save the complete h5 file
    # import pdb; pdb.set_trace()
    keep_id = []
    with h5py.File(merge_file_path, 'r') as file:
        for cell_id in tqdm(list(adata.obs.index)):
            if cell_id != "bahfibck-1":
                try: 
                    # import pdb; pdb.set_trace()
                    int_matrix = read_h5(file, str(cell_id)).tocsr()
                    #merge the matrix into the Anndata file
                    adata.uns[cell_id] = int_matrix
                    keep_id.append(cell_id)
                except:
                    print(f"{cell_id} not in the h5 data")
                    continue
    #saving the data into ".h5"
    adata = adata[keep_id, :]
    # import pdb; pdb.set_trace()
    adata.write(f"{save_dir}/{data_name}.h5ad")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='embed all the info altogether')
    parser.add_argument('--partitions', type=int, default=1, help='The number of partitions that need to be integrated')
    parser.add_argument('--data_name', type=str, default=None, help='The datanames use to identify the raw file')
    parser.add_argument('--matrix_name', type=str, default=None, help='The datanames use to identify the gene-gene interaction file')
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
               matrix_name = args.matrix_name,
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
