from tqdm import tqdm
import sys
sys.path.append("/scratch/project_465001820/Spatialformer")
import os
import scanpy as sc
import spatialformer as sp
import importlib
import torch
import numpy as np
importlib.reload(sp)


adata_subset_10fra = sc.read("/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/test.h5ad")

selected_cells = adata_subset_10fra.obs.index


batch_size = 64
tissue = "Lung"
condition = "Disease"
# model_ckp_path = "/scratch/project_465001027/Spatialformer/output/checkpoints/step=0096000-train_total_loss=-2.9351-val_total_loss=0.0000.ckpt"
# model_ckp_path = "/scratch/project_465001027/Spatialformer/output/checkpoints/step=0044000-train_total_loss=-1.3226-val_total_loss=0.2488.ckpt"
# model_ckp_path = "/scratch/project_465001027/Spatialformer/output/checkpoints/step=0100000-train_total_loss=-2.2727-val_total_loss=0.0000.ckpt"
#pairwise
model_ckp_path = "/scratch/project_465001820/Spatialformer/output/checkpoints/stepstep=0176000-traintrain_total_loss=-2.8414-valval_total_loss=0.0000.ckpt"
method = "cls" #getting the cls token embeddings


all_key_cells = []
all_query_cells = []
for i, query in enumerate(selected_cells):
    key_cells = list(selected_cells)
    querys = len(selected_cells)*[query]
    all_key_cells += key_cells
    all_query_cells += querys

_, paired = sp.tl.embed_data(adata = adata_subset_10fra, 
                            tissue = tissue,
                            condition = condition,
                            method = method,
                            model_ckp_path = model_ckp_path, 
                            batch_size = batch_size,
                            mode = "pair",
                            threshold = 0.7,
                            num_workers = 32,
                            gene_median_path = "/scratch/project_465001820/Spatialformer/data/gene_median.pkl",
                            left_cell = all_key_cells,
                            right_cell = all_query_cells,
                            resume_before_5k = False,
                            max_len=500
                            )
paired_rsl = torch.cat([i.detach().cpu() for i in paired]).numpy()

np.save("/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/paired_rsl", paired_rsl)




