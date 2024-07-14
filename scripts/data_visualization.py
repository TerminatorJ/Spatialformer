import scanpy as sc
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, triu
import scipy.sparse as sp
import torch
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import umap
import umap.plot
import pickle

#this code was used to generate the sample * features
'''
def count_cell(adata):
    count = len(adata.uns)
    return count

disease_path1 = "/scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD106__20230313__191400/processed/relabel_output-XETG00048__0003392__VUILD106__20230313__191400.h5ad"
disease_adata1 = sc.read_h5ad(disease_path1)

disease_path2 = "/scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD110__20230313__191400/processed/relabel_output-XETG00048__0003392__VUILD110__20230313__191400.h5ad"
disease_adata2 = sc.read_h5ad(disease_path2)

healthy_path1 = "/scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/processed/relabel_output-XETG00048__0003392__THD0008__20230313__191400.h5ad"
healthy_adata1 = sc.read_h5ad(healthy_path1)

healthy_path2 = "/scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__THD0011__20230313__191400/processed/relabel_output-XETG00048__0003400__THD0011__20230313__191400.h5ad"
healthy_adata2 = sc.read_h5ad(healthy_path2)

adata_lst = [disease_adata1, disease_adata2, healthy_adata1, healthy_adata2]

ref_gene = sorted(disease_adata1.var["gene_name"].unique())
# # ref_gene = sorted(disease_adata1.var["gene_name"].unique())
num_genes = len(ref_gene)
# # ref_matrix = sp.csr_matrix((num_genes, num_genes))

# # # Step 2: Extract upper triangle indices (excluding diagonal)
upper_triangle_indices = np.triu_indices(num_genes, k=1)
feature_length = len(upper_triangle_indices[0])

n_samples = np.sum([count_cell(adata) for adata in adata_lst])
# # # print(n_samples)
# # # # # Step 3: Initialize a sparse data matrix to store feature vectors for each sample
data_matrix = sp.lil_matrix((n_samples, feature_length))  



# # Initialize lists to store indices and values for the sparse tensor
indices = []
values = []

color_map = []
cell_count = 0

for i, adata in tqdm(enumerate(adata_lst)):
    sample_type = "disease" if i < 2 else "healthy"
    for cell_id in tqdm(adata.uns.keys(), leave=False):
        cell_count += 1
        this_matrix = adata.uns[cell_id]
        
        # Convert the matrix to a PyTorch tensor and move it to GPU
        this_matrix_tensor = torch.tensor(this_matrix.toarray(), dtype=torch.float32, device='cuda')
        
        # Get the upper triangle indices
        upper_triangle_indices = np.triu_indices(num_genes, k=1)
        
        # Extract upper triangle values
        # for idx, (row_idx, col_idx) in enumerate(zip(upper_triangle_indices[0], upper_triangle_indices[1])):
        row_indices_tensor = torch.tensor(upper_triangle_indices[0], dtype=torch.long, device=this_matrix_tensor.device)
        col_indices_tensor = torch.tensor(upper_triangle_indices[1], dtype=torch.long, device=this_matrix_tensor.device)
        upper_triangle_values = this_matrix_tensor[row_indices_tensor, col_indices_tensor]
        data_matrix[cell_count -1,:] = upper_triangle_values.detach().cpu().numpy()

        color_map.append(sample_type)

Convert indices and values to tensors
indices = torch.tensor(indices, dtype=torch.long).t().contiguous()
values = torch.tensor(values, dtype=torch.float32)


csr_matrix = data_matrix.tocsr()

sp.save_npz("./data_matrix.npz", csr_matrix)

'''


#the following code was used to generate umap

data_matrix = load_npz("./data_matrix.npz")
color_map = pickle.load(open("./color_map.pkl","rb"))
color_map = np.array(color_map)
print("Doing PCA")
svd = TruncatedSVD(n_components=50, random_state=42)  # Reduce to 50 dimensions first
svd_result = svd.fit_transform(data_matrix)
print("after dimention reduction", svd_result.shape)



samples = 30000

# If you want to concatenate them into one array
combined_result = np.concatenate((svd_result[:samples], svd_result[-samples:]), axis=0)

combined_color = np.concatenate((color_map[:samples], color_map[-samples:]), axis = 0)



# n_neighbors = [5,10,15,20]
# min_dist = [0.1, 0.25, 0.5, 1]
n_neighbors = [10]
min_dist = [0.5]

for n in n_neighbors:
    for m in min_dist:
        print("running", m, n)
        mapper = umap.UMAP(n_neighbors = n, min_dist = m).fit(combined_result)
        print("to fit the data")
        # import pdb; pdb.set_trace()

        fig = umap.plot.points(mapper, labels=combined_color, color_key_cmap='Set1')
        fig.figure.savefig(f"/scratch/project_465001027/spatialformer/figure/gene_gene_4samples_{n}_{m}.png", dpi = 300)




    