import sys
sys.path.append("/scratch/project_465001027/Spatialformer")
import scanpy as sc
import spatialformer as sp

covid_data = sc.read_h5ad("/scratch/project_465001027/Spatialformer/downstream/cell_cell_communication/data/covid_subsampled.h5ad")
model_ckp_path = "/scratch/project_465001027/Spatialformer/spatialformer/checkpoint/step=0015000-train_total_loss=-0.5154-val_total_loss=0.0000v2.ckpt"
batch_size = 8
tissue = "Lung"
condition = "Disease"

#testing the single cell mode
# embeddings = sp.tl.embed_data(covid_data, 
#                               tissue,
#                               condition,
#                             model_ckp_path, 
#                             batch_size,
#                             mode = "single",
#                             )
#tesing the pair cell mode
# left_cell = ["483188-0-0-1"]
# right_cell = ["144-0-1-0-1"]
# embeddings, pair_results = sp.tl.embed_data(covid_data, 
#                               tissue,
#                               condition,
#                             model_ckp_path, 
#                             batch_size,
#                             mode = "pair",
#                             left_cell = left_cell,
#                             right_cell = right_cell
#                             )
#testing the gene-gene colocalization 
# covid_data = covid_data[:16,:]
embed_adata = sp.tl.embed_data(covid_data, 
                              tissue,
                              condition,
                            model_ckp_path, 
                            batch_size,
                            mode = "single",
                            threshold = 0.8
                            )

import pdb; pdb.set_trace()



