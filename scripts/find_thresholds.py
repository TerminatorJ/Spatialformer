# %%
import os 
import sys 
import h5py
import pandas as pd
import numpy as np
import multiprocessing
import argparse
import itertools
from pathlib import Path
import random
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[0]
sys.path.append("p_path")
from process import KNN_Radius_Graph
import pickle
import os
from tqdm import tqdm
import dask.dataframe as dd

#we drop the dataset with too many cells to statistic the distribution of the data, we didn't statistic the datasets that stored with parquet format



def process(lung_annot_3D_tx, ID):
   
    #filtering the weired genes
    lung_annot_3D_tx = lung_annot_3D_tx[~(lung_annot_3D_tx['gene'].str.startswith('Neg') | lung_annot_3D_tx['gene'].str.startswith('BLANK'))]
    #filtering the transcripts
    value_counts = lung_annot_3D_tx['cell_id'].value_counts().reset_index()
    #get the number of unique genes for each cell
    gene_counts = lung_annot_3D_tx.groupby('cell_id')['gene'].nunique().reset_index(name='unique_gene_count')
    #get mean count of all gene in the same cell
    genet_counts = lung_annot_3D_tx.groupby(['cell_id', 'gene']).size().reset_index(name='count')
    mean_gene_count = genet_counts.groupby('cell_id')['count'].mean().reset_index(name='mean_gene_count')
    #merge all the results
    merged_result = value_counts.merge(gene_counts, on='cell_id').merge(mean_gene_count, on='cell_id')


    #set as index
    merged_result = merged_result[merged_result['cell_id'] != 'UNASSIGNED']
    merged_result.set_index('cell_id', inplace=True)
    merged_result.index.name = None
    #create a unique cell id
    # import pdb; pdb.set_trace()
    merged_result.index = ID + "_" + merged_result.index.astype(str)
    return merged_result

def process_parquet(ddf, ID):
    # import pdb; pdb.set_trace()
    # Filtering out weird genes
    ddf_filtered = ddf[~ddf['feature_name'].str.startswith('Neg') & ~ddf['feature_name'].str.startswith('BLANK')]

    # Filtering the transcripts: Get value counts for cell_id
    value_counts = ddf_filtered['cell_id'].value_counts().compute().reset_index()
    value_counts.columns = ['cell_id', 'total_counts']

    # Get the number of unique genes for each cell_id
    gene_counts = ddf_filtered.groupby('cell_id')['feature_name'].nunique().compute().reset_index(name='unique_gene_count')

    # Get the mean count of all genes in the same cell
    gene_occurrences = ddf_filtered.groupby(['cell_id', 'feature_name']).size().compute().reset_index(name='count')
    mean_gene_count = gene_occurrences.groupby('cell_id')['count'].mean().compute().reset_index(name='mean_gene_count')

    # Merge all the results
    merged_result = value_counts.merge(gene_counts, on='cell_id').merge(mean_gene_count, on='cell_id')



    #set as index
    merged_result = merged_result[merged_result['cell_id'] != 'UNASSIGNED']
    merged_result.set_index('cell_id', inplace=True)
    merged_result.index.name = None
    #create a unique cell id
    # import pdb; pdb.set_trace()
    merged_result.index = ID + "_" + merged_result.index.astype(str)
    return merged_result



def getdata(transcript_path, ID):

    try:
        lung_annot_3D_tx = pd.read_csv(os.path.join(transcript_path, "transcripts.csv"))
        lung_annot_3D_tx.rename(columns={'x_location': 'x', 'y_location':'y', 'z_location':'z', 'feature_name':'gene'}, inplace=True)
        merged_result = process(lung_annot_3D_tx, ID)
    except FileNotFoundError:
        try:
            print("The transcripts file is already compressed")
            lung_annot_3D_tx = pd.read_csv(os.path.join(transcript_path, "transcripts.csv.gz"), compression='gzip')
            lung_annot_3D_tx.rename(columns={'x_location': 'x', 'y_location':'y', 'z_location':'z', 'feature_name':'gene'}, inplace=True)
            merged_result = process(lung_annot_3D_tx, ID)
        except FileNotFoundError:
            print("loading the parquet file")
            # lung_annot_3D_tx = pd.read_parquet(os.path.join(transcript_path, "transcripts.parquet"), columns=["cell_id","feature_name"], engine='pyarrow')
            lung_annot_3D_tx = dd.read_parquet(os.path.join(transcript_path, "transcripts.parquet"), columns=["cell_id","feature_name"])
            process_parquet(lung_annot_3D_tx, ID)
            print("after loading the parquet file")
            merged_result = lung_annot_3D_tx.rename(columns={'feature_name':'gene'}, inplace=True)

        except Exception as e:
            print(f"An error occurred : {e} for {transcript_path}")
    except Exception as e:
        print(f"An error occurred: {e} for {transcript_path}")
    
 


    return merged_result



if __name__ == "__main__":
    root = "/tmp/erda/Spatialformer/downloaded_data/raw"

    names = [
    "Xenium_V1_hLung_cancer_section_outs",
    "Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs",
    "Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs",
    "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs",
    "Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon_outs",
    "Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon_outs",
    "Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs",
    "Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs",
    "Xenium_V1_human_Pancreas_FFPE_outs",
    "Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs",
    "Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs",
    "Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs",
    "Xenium_V1_hLiver_nondiseased_section_FFPE_outs",
    "Xenium_V1_hLiver_cancer_section_FFPE_outs",
    "Xenium_V1_hHeart_nondiseased_section_FFPE_outs",
    "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs",
    "Xenium_V1_hSkin_Melanoma_Base_FFPE_outs",
    "Xenium_V1_hColon_Non_diseased_Base_FFPE_outs",
    "Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs",
    "Xenium_V1_hColon_Cancer_Base_FFPE_outs"
    ]

    ids = [
        1,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27
    ]


    df_list = []
    for name,id in tqdm(zip(names, ids)):
        print(f"running {name}")
        transcript_path = os.path.join(root, name)
        ID = "Xenium_" + str(id)
        merged_result = getdata(transcript_path, ID)
        df_list.append(merged_result)

    
    # transcript_path = os.path.join(root, "Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs")
    # clean_value_counts = getdata(transcript_path, "Xenium_10")

    combined_df = pd.concat(df_list)
    combined_df.to_csv("/home/sxr280/Spatialformer/data/all48samples_info.csv")



# %%

