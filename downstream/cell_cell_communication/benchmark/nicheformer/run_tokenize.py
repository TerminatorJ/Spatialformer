import scanpy as sc
import anndata as ad
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
import numba
import math
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
import pyarrow.parquet as pq
import pyarrow
import os
import pandas as pd


def sf_normalize(X):
    # print("normaliza")
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    # avoid zero devision error
    counts += counts == 0.
    # normalize to 10000. counts
    scaling_factor = 10000. / counts
    # import pdb; pdb.set_trace()
    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X

# @numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(x: np.array, max_seq_len: int = -1, aux_tokens: int = 30):
    # import pdb; pdb.set_trace()
    scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    # print("x", x)
    # import pdb; pdb.set_trace()
    for i, cell in enumerate(x):
        # import pdb; pdb.set_trace()
        # print("x cell",cell)
        cell = np.array(cell)[0]
        nonzero_mask = np.nonzero(cell)[0]    
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len] 
        # print("sorted_indices", sorted_indices)
        sorted_indices = sorted_indices + aux_tokens # we reserve some tokens for padding etc (just in case)
        # print("sorted_indices2", sorted_indices)
        if max_seq_len:
            scores = np.zeros(max_seq_len, dtype=np.int32)
        else:
            scores = np.zeros_like(cell, dtype=np.int32)
        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)
        # print("scores", scores)
        
        scores_final[i, :] = scores
        # print("score final", scores_final)
        
    return scores_final


def tokenize_data(x: np.array, median_counts_per_gene: np.array, max_seq_len: int = None):
    """Tokenize the input gene vector to a vector of 32-bit integers."""
    # print("start tokenization")
    # import pdb; pdb.set_trace()
    x = np.nan_to_num(x) # is NaN values, fill with 0s
    x = sf_normalize(x)

    median_counts_per_gene += median_counts_per_gene == 0
    out = x / median_counts_per_gene.reshape((1, -1))
    # print("out", out)
    scores_final = _sub_tokenize_data(out, 4096, 30)

    return scores_final.astype('i4')
def reorder(ref, query): 
    """Getting the adata with full gene vocabulary"""
    intersecting_genes = ref.var.index.isin(query.var["gene_ids"])
    combined_data = np.zeros((query.shape[0], ref.shape[1]))
    new_adata = ad.AnnData(X=csr_matrix(combined_data))
    new_adata[:, intersecting_genes] = query.X
    return new_adata

def run_in_batch(batch_input):
    batch_data, obs_tokens_batch, batch_num, xenium_mean, OUT_PATH = batch_input
    # print("batch_data", batch_data)
    # print("obs_tokens_batch", obs_tokens_batch)
    # import pdb; pdb.set_trace()
    tokenized = tokenize_data(batch_data, xenium_mean, 4096)
    # print("after getting tokenization:", tokenized)

    obs_tokens_batch = obs_tokens_batch[['assay', 'specie', 'modality', 'idx']]
#     # concatenate dataframes
    # import pdb; pdb.set_trace()
    obs_tokens_batch['X'] = [tokenized[i, :] for i in range(tokenized.shape[0])]
    return obs_tokens_batch
# #     # mix spatial and dissociate data
    
    


def run_AIO(adata):
    #reset the index
    adata.obs.reset_index(drop=True, inplace=True)
    global xenium_mean
    global OUT_PATH
    xenium_mean = np.load("/scratch/project_465001027/nicheformer/data/model_means/xenium_mean_script.npy")
    xenium_mean = np.nan_to_num(xenium_mean)
    rounded_values = np.where((xenium_mean % 1) >= 0.5, np.ceil(xenium_mean), np.floor(xenium_mean))
    xenium_mean = np.where(xenium_mean == 0, 1, rounded_values)
    OUT_PATH = '/scratch/project_465001027/Spatialformer/downstream/cell_cell_communication/benchmark/nicheformer/data'

    #split into batch
    obs_adata = adata.obs
    print('n_obs: ', obs_adata.shape[0])
    N_BATCHES = math.ceil(obs_adata.shape[0] / 10_00)
    print('N_BATCHES: ', N_BATCHES)
    batch_indices = np.array_split(obs_adata.index, N_BATCHES)
    chunk_len = len(batch_indices[0])
    print('chunk_len: ', chunk_len)

    obs_adata = obs_adata.reset_index().rename(columns={'index':'idx'})
    obs_adata['idx'] = obs_adata['idx'].astype('i8')

    results = []
    # N_BATCHES = 20
    # chunk_len = 20
    # batch_data = adata.X[0*chunk_len:chunk_len*(0+1)]
    # obs_tokens_batch = obs_adata.iloc[0*chunk_len:chunk_len*(0+1)].copy()
    # print("before running batch sample:", batch_data)
    # batch_input = (batch_data, obs_tokens_batch, 0, xenium_mean, OUT_PATH)
    # run_in_batch(batch_input)
    with ProcessPoolExecutor(max_workers=32) as executor:
        # Processing in batches
        for batch_num in tqdm(range(N_BATCHES)):
            batch_data = adata.X[batch_num*chunk_len:chunk_len*(batch_num+1)]
            obs_tokens_batch = obs_adata.iloc[batch_num*chunk_len:chunk_len*(batch_num+1)].copy()
            # print("before running batch sample:", batch_data)
            batch_input = (batch_data, obs_tokens_batch, batch_num, xenium_mean, OUT_PATH)
            # Submit the batch processing to the executor
            # print(batch)
            future = executor.submit(run_in_batch, batch_input)
            # print(future.result())
            results.append(future.result())
    for i,result in enumerate(results):
        total_table = pyarrow.Table.from_pandas(result)
        # print(total_table)
        print("saving the data")
        pq.write_table(total_table, f'{os.path.join(OUT_PATH)}/tokens-{i}.parquet',
                        row_group_size=1024,)
        

if __name__ == "__main__":
    ref = sc.read_h5ad("/scratch/project_465001027/nicheformer/data/model_means/model.h5ad")
    data_dir = "/scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/outs/cell_feature_matrix"
    adata = sc.read_10x_mtx(data_dir,  # The directory containing the files
                            var_names='gene_symbols',  # Use 'gene_ids' if you prefer the IDs
                            cache=True) 

    new_adata = reorder(ref, adata)
    new_adata.obs["assay"] = 9
    new_adata.obs["modality"] = 4
    new_adata.obs["specie"] = 5

    #run in multiprocesses
    print("run in main thread")
    # import pdb; pdb.set_trace()
    # print(new_adata.X)
    run_AIO(new_adata)



