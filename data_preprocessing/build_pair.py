from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import numpy as np
import h5py
import os
import sys
sys.path.append("/scratch/project_465001027/Spatialformer/utils")
from utils import *
import pickle
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split


def save_pair_dataset(dataset, sample_cell_index, radius, num_workers, chunk, res, define_name):
    '''
    dataset: huggingface dataset
    index_path: the index build for huggingface dataset for fast retrive
    radius: the radius for getting the cell connection
    
    '''

    combined_dataset_all = concatenate_datasets([dataset["train"], dataset["test"], dataset["validation"]])
    print("building all the datasets from all the samples!!!")
    # import pdb; pdb.set_trace()
    num_sample = len(list(sample_cell_index.keys()))
    target_dirs = os.listdir("/scratch/project_465001027/Spatialformer/cache")
    bs = 6
    pair_files = [file for file in target_dirs if "pair" in file]
    file_left = [file for file in list(sample_cell_index.keys()) if "xenium_" + file + "_pair" not in pair_files]
    # import pdb; pdb.set_trace()
    for sample_name in tqdm(list(sample_cell_index.keys())[(chunk-1)*bs + res:min(chunk*bs, num_sample)]):
    # for sample_name in ['Xenium_V1_hColon_Non_diseased_Base_FFPE_outs', 'Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs', 'Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs'][2:3]:
        save_path = f"xenium_{sample_name}_pair"
        # import pdb; pdb.set_trace()
        if save_path not in target_dirs:

            # if save_path not in target_dirs:
            print(f"running {save_path}")
            # Filter the dataset for rows corresponding to the current sample_name
            sample_index = list(sample_cell_index[sample_name].values())
            sample_data = combined_dataset_all.select(sample_index)
            
            sparse_adjmtx,cell_ids = get_adj(sample_data, radius = radius, plot = False)
            # import pdb; pdb.set_trace()
            sample_data = sample_data.select_columns(["Full_Tokens","Gene_Gene_Matrix","Expression","Cell_Ids"])
            # import pdb;pdb.set_trace()
            sample_data = sample_data.map(binary_to_coo_matrix, num_proc = num_workers)
            sample_data = sample_data.remove_columns("Gene_Gene_Matrix")
            Pairs = GetPairs(sparse_adjmtx, num_workers = num_workers) #sample_index, (leftdataset, rightdataset), label
            
            # import pdb;pdb.set_trace()
            all_left_idxs = list(map(lambda x: x[0],Pairs.all_pairs))
            all_right_idxs = list(map(lambda x: x[1],Pairs.all_pairs))
            # import pdb; pdb.set_trace()
            all_labels = Pairs.all_labels
            all_left_dataset = sample_data.select(all_left_idxs)
            all_right_dataset = sample_data.select(all_right_idxs)
            # import pdb; pdb.set_trace()
            left_renamed = all_left_dataset.rename_columns({col: f'left_{col}' for col in all_left_dataset.column_names})
            right_renamed = all_right_dataset.rename_columns({col: f'right_{col}' for col in all_right_dataset.column_names})
            # import pdb; pdb.set_trace
            # Concatenate the two datasets
            combined_dataset = concatenate_datasets([left_renamed, right_renamed], axis=1)
            combined_dataset = combined_dataset.add_column("Labels", all_labels)
            # combined_datasets.append(combined_dataset)
            combined_dataset.save_to_disk(f"/scratch/project_465001027/Spatialformer/cache/xenium_{sample_name}_pair", num_proc = 32)
            # import pdb; pdb.set_trace()

if __name__ == "__main__":

    combined_dataset = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset4")  
    index_path = "/scratch/project_465001027/Spatialformer/data/sample_cell_index.pkl"
    sample_cell_index = get_index(combined_dataset, save_file = index_path)
    save_pair_dataset(
                    combined_dataset, 
                    sample_cell_index, 
                    radius = 30, 
                    num_workers = 32,
                    chunk = 9,
                    res = 1
                    )