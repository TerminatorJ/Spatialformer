#!/usr/bin/env python
# coding: utf-8

# # Cell-cell colocalization for Xenium breast sample

# We cannot use the script/fine-tune.py directly because the pipeline is built for the huggingface dataset as the input.     Any external new test dataset should be loaded via "tools.get_embeddings.py" function  
# - The breast cancer tumor microenvironment dataset can be downloaded via:  
# https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast.   
# - Publication DOI:   
# https://doi.org/10.1038/s41467-023-43458-x

# In[1]:


import sys
sys.path.append("/scratch/project_465001820/Spatialformer")
sys.path.append("/scratch/project_465001820/Spatialformer/scripts")
sys.path.append("/scratch/project_465001820/Spatialformer/spatialformer/")
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from scipy.spatial import KDTree
from utils.utils import GetPairs, get_adj, split_dataset
from fine_tune import FineTune
import json
from tools import embed_data


# Loading the Xenium count matrix, and cell metadata

# In[2]:


adata_breast = sc.read_10x_h5("/scratch/project_465001820/Spatialformer/data/Xenium/outs/cell_feature_matrix.h5")
cell_meta = pd.read_parquet("/scratch/project_465001820/Spatialformer/data/Xenium/outs/cells.parquet")


# In[3]:


#add the spatial coordinates to the anndata
adata_breast.obsm["spatial"] = cell_meta[["x_centroid", "y_centroid"]].values
#convert the data type of index into interger
adata_breast.obs.index = adata_breast.obs.index.astype(int)


# In[4]:


adata_breast.var["gene_name"] = adata_breast.var.index


# In[5]:


adata_breast.obsm["spatial"]


# Getting the matrix of the cell neighbors

# In[6]:


# Getting the asymmetry matrix
sparse_adj, cell_ids = get_adj(sample_dataset = None, anndata = adata_breast, radius = 5, plot = False, sym = True)
### Getting the cell pairs according to the distance
Pairs = GetPairs(sparse_adj, num_workers = 8) #assign the 1:1 negative to the positive


# In[7]:


all_pairs = Pairs.all_pairs
all_labels = Pairs.all_labels


# In[8]:


all_labels.shape


# Random select 500 cells for evaluating the model

# In[9]:


zero_shot_cell_size = 500
selected_pairs, selected_labels = split_dataset(all_pairs, all_labels, n_splits = 0, test_size = None, zero_shot_cell_size = zero_shot_cell_size)
# test_dataloader = data_prepare(sample_name, kfold, num_workers, batch_size, radius=r, test_size = None, zero_shot_cell_size = zero_shot_cell_size, split_mode = "random")


# In[10]:


config_path = "/scratch/project_465001820/Spatialformer/config/_config_fine_tune_probe.json"
with open(config_path, 'r') as json_file:
    config = json.load(json_file)
# model_ckp_path = "/scratch/project_465001820/Spatialformer/output/checkpoints/step=0096000-train_total_loss=-2.9351-val_total_loss=0.0000.ckpt"
model_ckp_path = "/scratch/project_465001820/Spatialformer/output/checkpoints/step=0044000-train_total_loss=-1.3226-val_total_loss=0.2488.ckpt"
r = 10


# Getting the dataloader

# In[11]:


test_dataloader = embed_data(adata_breast,
               tissue = "Breast", 
               condition = "Disease",
               method = "gene",
               model_ckp_path = model_ckp_path, 
               batch_size = 4,
               mode = "pair",
               only_loader = True,
               left_cell = selected_pairs[:,0],
               right_cell = selected_pairs[:,1],
               pair_label = selected_labels,
               num_workers = 8,
               reveal_name = False
               )


# Run the model to get the predictions

# In[12]:


sample_name = "breast_cancer"
fine_tune_mode = "zero_shot"
all_results = {}
Finetune = FineTune(config, model_ckp_path, sample_name, r, fine_tune_mode, wandb = True, strategy = "ddp")
probe_model = Finetune.probe_model
results = Finetune.test(probe_model, test_dataloader)
#tesing the model
all_results[r] = results[0]


# In[ ]:





# In[ ]:





# In[ ]:




