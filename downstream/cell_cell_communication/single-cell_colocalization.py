import sys
sys.path.append("/scratch/project_465001027/Spatialformer")
import scanpy as sc
import spatialformer as sp
import random
import numpy as np
from tqdm import tqdm
# covid_data = sc.read_h5ad("/scratch/project_465001027/Spatialformer/downstream/cell_cell_communication/data/covid_subsampled.h5ad")
# model_ckp_path = "/scratch/project_465001027/Spatialformer/spatialformer/checkpoint/step=0015000-train_total_loss=-0.5154-val_total_loss=0.0000v2.ckpt"
# batch_size = 8
# tissue = "Lung"
# condition = "Disease"

#testing the single cell mode
# embeddings = sp.tl.embed_data(covid_data, 
#                               tissue,
#                               condition,
#                             model_ckp_path, 
#                             batch_size,
#                             mode = "single",
#                             )
#tesing the pair cell mode
# import pdb; pdb.set_trace()

# left_cells = np.random.choice(covid_data.obs.index, size=10, replace=False)

adata = sc.read("/scratch/project_465001027/Spatialformer/spatialformer/tools/test.h5ad")

selected_cells = adata.obs.index


batch_size = 16
tissue = "Lung"
condition = "Disease"
# model_ckp_path = "/scratch/project_465001027/Spatialformer/output/checkpoints/step=0096000-train_total_loss=-2.9351-val_total_loss=0.0000.ckpt"
model_ckp_path = "/scratch/project_465001027/Spatialformer/output/checkpoints/step=0044000-train_total_loss=-1.3226-val_total_loss=0.2488.ckpt"
# model_ckp_path = "/scratch/project_465001027/Spatialformer/output/checkpoints/step=0100000-train_total_loss=-2.2727-val_total_loss=0.0000.ckpt"
method = "cls" #getting the cls token embeddings
# import pdb; pdb.set_trace()
#create the empty array 
pred = np.zeros((len(selected_cells), len(selected_cells), 2))
# right_cells = np.random.choice(covid_data.obs.index, size=10, replace=False)
for i, query in tqdm(enumerate(selected_cells)):
    key_cells = list(selected_cells)
    querys = len(selected_cells)*[query]
    # import pdb; pdb.set_trace()
    _,paired = sp.tl.embed_data(adata, 
                                  tissue,
                                  condition,
                                   method,
                                model_ckp_path, 
                                batch_size,
                                mode = "pair",
                                threshold = 0.7,
                                num_workers = 16,
                                left_cell = querys,
                                right_cell = key_cells
                               )
    all_paired = np.vstack(paired)
    pred[i] = all_paired
    # import pdb; pdb.set_trace()

np.save("/scratch/project_465001027/Spatialformer/downstream/cell_cell_communication/data/pred2.npy", pred)

#testing the gene-gene colocalization 
# covid_data = covid_data[:16,:]
# embed_adata = sp.tl.embed_data(covid_data, 
#                               tissue,
#                               condition,
#                             model_ckp_path, 
#                             batch_size,
#                             mode = "single",
#                             threshold = 0.8
#                             )

import pdb; pdb.set_trace()



