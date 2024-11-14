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
               datapath_name: str = "david_data"
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
        raw_dir = os.path.join(p_path, datapath_name, data_name, "outs")
        data_dir = os.path.join(p_path, datapath_name)
        save_dir = os.path.join(p_path, datapath_name, data_name, "processed")
    else:
        raw_dir = os.path.join(p_path, datapath_name, "raw", data_name)
        data_dir = os.path.join(p_path, datapath_name)
        #saved path with training and validation dataset
        save_dir = os.path.join(p_path, datapath_name, "processed", data_name)
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
    transcript_df = pd.read_csv(f"{raw_dir}/transcripts.csv") #2G
    transcript_df.rename(columns={'x_location': 'x', 'y_location':'y', 'z_location':'z', 'feature_name':'gene'}, inplace=True)
    value_counts = transcript_df['cell_id'].value_counts()
    clean_value_counts = value_counts.drop("UNASSIGNED")
    kept_cell_id_unique = transcript_df[transcript_df['cell_id'].isin(clean_value_counts.index[clean_value_counts >= transcript_threshold])]['cell_id'].unique()
    #filtering the cells match the threshold
    adata = adata[kept_cell_id_unique, :]

    #filtering shoud be identified here, filtering the auxiliary genes
    genes_mask = ~(adata.var["gene_name"].str.startswith('Neg') | adata.var["gene_name"].str.startswith('BLANK'))
    adata = adata[:, genes_mask]

    #split the dataset and then attach the split tags
    adata = split_data(adata, train_proportion=0.64, test_proportion=0.2, validation_proportion=0.16)

    #getting the compartment information for the downstream verification
    #TODO: getting the nucleus and cytoplasm info
    # adata.obs["Compartments"] = 'nuclus'


    #merge all the .h5 file to a single file
    # List of input file paths
    # h5_file_paths = [h5_file_path.split(".")[0][:-1] + str(partition) +"."+h5_file_path.split(".")[1] for partition in range(1, partitions+1)]
    data_files = [os.path.join(os.path.abspath(data_dir),file) for file in os.listdir(data_dir)]
    matching_files = [file for file in data_files if matrix_name in file and "merge" not in file]
    merge_file_path = os.path.join(data_dir, matrix_name + "_merged.h5")
    # Create a new HDF5 file for merging
    if not os.path.exists(merge_file_path):
        print(f"{merge_file_path} is not exists, running the code to merge all the partitions")
        with h5py.File(merge_file_path, "w") as merged_file:
            for h5_file_path in tqdm(matching_files):
                with h5py.File(h5_file_path, "r") as input_file:
                    # Copy datasets from input file to merged file
                    for cell_id in tqdm(list(input_file.keys())):
                        old_grp = input_file[cell_id]
                        new_grp = merged_file.create_group(str(cell_id))
                        new_grp.create_dataset('data', data=list(old_grp["data"]))
                        new_grp.create_dataset('row', data=list(old_grp["row"]))
                        new_grp.create_dataset('col', data=list(old_grp["col"]))
                        new_grp.attrs['shape'] = old_grp.attrs['shape']

    print(f"Merged datasets from {len(matching_files)} files into {merge_file_path}")

    #Adding the matrix to the AnnData file.
    #TODO: adding the gene matrix to the anndata
    #do it while confirm the threshold settings
    #open the h5 file
    with h5py.File(merge_file_path, 'r') as file:
        for cell_id in tqdm(list(adata.obs.index)):
            int_matrix = read_h5(file, cell_id).tocsr()
            #merge the matrix into the Anndata file
            adata.uns[cell_id] = int_matrix
    #saving the data into ".h5"
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
    args = parser.parse_args()
    
    preprocess(partitions = args.partitions, 
               data_name = args.data_name,
               matrix_name = args.matrix_name,
               transcript_threshold = args.transcript_threshold,
               Condition = args.condition,
               Tissues = args.tissues,
               Species = args.species,
               Assay = args.assay
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



