from data_loader import create_single_data_loaders
import json
import sys
sys.path.append("/scratch/project_465001820/Spatialformer/scripts")
from train import manual_train_fm
import os
sys.path.append("/scratch/project_465001820/Spatialformer/utils")
from utils import *
import torch
import pandas as pd
from datasets import load_from_disk,concatenate_datasets,load_dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np


def load_model(model_ckp_path, device):
    get_file_path = lambda path, filename: os.path.join("/scratch/project_465001820/Spatialformer", path, filename)
    config_path = get_file_path("config", "_config_train_large_pair.json")
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    model = manual_train_fm(config = config)
    ckp = torch.load(model_ckp_path, map_location=torch.device(device))
    params = ckp["state_dict"]
    model.load_state_dict(params)
    model.eval()
    model.to(device)
    return model

# Download the checkpoint file to your own path
model_ckp_path = "/scratch/project_465001027/Spatialformer/output/checkpoints/step=0104000-train_total_loss=-2.3064-val_total_loss=0.0000.ckpt"
model = load_model(model_ckp_path, "cuda")
# Loading the sample dataset from the huggingface dataset repository
# If you have already downloaded the dataset, remotely, you can load the dataset locally
#This is the path, where you can load your dataset.
datapath = "/scratch/project_465001820/Spatialformer/cache/" #customize your own path here
dataname = "xenium_pandavid_dataset4"
combined_dataset = load_dataset(f"TerminatorJ/{dataname}", cache_dir = datapath, num_proc=8) #using multiprocess to get better performance

# Put all the splits together
combined_dataset_all = concatenate_datasets([combined_dataset["train"], combined_dataset["test"], combined_dataset["validation"]])

# Index the cell to make flash access possible
index_path = "/scratch/project_465001027/Spatialformer/data/sample_cell_index.pkl"
sample_cell_index = get_index(combined_dataset, save_file = index_path)


df = pd.read_excel("/scratch/project_465001820/Spatialformer/data/Supplementary_Table1.xlsx", sheet_name='Human_data')
dv_df = pd.read_excel("/scratch/project_465001820/Spatialformer/data/Supplementary_Table1.xlsx", sheet_name='David_dataset', header = 1)

def assign(sample):
    # Precompute lookup dictionaries for fast access
    dv_sample_to_status = dict(zip(dv_df["Sample"], dv_df["Status"]))
    df_strid_to_tissue = dict(zip(df["Str_ID"], df["Tissues"]))
    df_strid_to_condition = dict(zip(df["Str_ID"], df["Condition2"]))

    tissues = []
    conditions = []
    for sample_name in sample["Sample_Names"]:
        if sample_name in dv_sample_to_status:
            tissues.append("Lung")
            conditions.append(dv_sample_to_status[sample_name])
        else:
            tissues.append(df_strid_to_tissue[sample_name])
            conditions.append(df_strid_to_condition[sample_name])
    return {"Tissues": tissues, "Conditions": conditions}
#getting the tissues and condition information
combined_dataset_tmp = combined_dataset_all.map(assign, batched = True, batch_size = 300, num_proc = 32)


from huggingface_hub import login
login(token="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")

combined_dataset_tmp.push_to_hub("xenium_pandavid_dataset5")