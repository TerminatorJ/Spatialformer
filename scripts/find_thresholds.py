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

lung_annot_3D_tx = pd.read_csv("/scratch/project_465001027/nicheformer/src/nicheformer/data/raw/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/transcripts.csv")
lung_annot_3D_tx.rename(columns={'x_location': 'x', 'y_location':'y', 'z_location':'z', 'feature_name':'gene'}, inplace=True)


threshold = 30
radius = 5
number_cell = 100
pair_threshold = 5
#segment 1
# pair_threshold_range = [2, 3]
#segment 2
# pair_threshold_range = [5, 7]
#segment 3
# pair_threshold_range = [10, 15]
# pair_threshold_range = [2, 3, 5, 7, 10]
# radius_range = [2, 3, 5, 7, 10]
# pair_threshold_range = [5, 7, 10]
# radius_range = [5, 7, 10]

#filtering the cells
value_counts = lung_annot_3D_tx['cell_id'].value_counts()
clean_value_counts = value_counts.drop("UNASSIGNED")
lung_annot_3D_tx_filtered = lung_annot_3D_tx[lung_annot_3D_tx['cell_id'].isin(clean_value_counts.index[clean_value_counts >= threshold])]
#filtering the transcripts
genes = list(lung_annot_3D_tx_filtered[~(lung_annot_3D_tx_filtered['gene'].str.startswith('Neg') | lung_annot_3D_tx_filtered['gene'].str.startswith('BLANK'))]["gene"].unique())

def save_to_hdf5(cell_id, gene_matrix):
    # import pdb; pdb.set_trace()
    file_path = os.path.join(save_path, "gene_matrices.h5")
    with h5py.File(file_path, "a") as f:
        f.create_dataset(str(cell_id), data=gene_matrix)
        
def calculate_func(cell_id, pair_threshold, radius):
    # import pdb;pdb.set_trace()
    self_threshold = pair_threshold
    print("running:",cell_id, ",", pair_threshold, ",", self_threshold)
    gene_int = {}
    transcript_int = {}
    data_graph = KNN_Radius_Graph(radius=radius, dataset=lung_annot_3D_tx_filtered, is_3D=True, cell_ID = cell_id, ref_gene = genes)
    gene_matrix, trans_matrix = data_graph.get_gene_matrix(pair_threshold = pair_threshold, self_threshold = pair_threshold, plot = False)
    #for gene level
    ig_allsum = np.triu(gene_matrix).sum()
    ig_rowsum = np.triu(gene_matrix).sum(axis = 1)
    ig_mean = ig_rowsum.mean()
    ig_mean_nonzero = np.mean(ig_rowsum[ig_rowsum != 0])
    gene_int[str(pair_threshold)+"_"+str(radius)] = [ig_allsum, ig_mean, ig_mean_nonzero]
    #for transcript level
    it_allsum = np.triu(trans_matrix).sum()
    it_rowsum = np.triu(trans_matrix).sum(axis = 1)
    it_mean = it_rowsum.mean()
    it_mean_nonzero = np.mean(it_rowsum[it_rowsum != 0])
    transcript_int[str(pair_threshold)+"_"+str(radius)] = [it_allsum, it_mean, it_mean_nonzero]
    
    return gene_int, transcript_int

# cell_ids = list(lung_annot_3D_tx_filtered['cell_id'].unique())
# random_cell_ids = random.sample(cell_ids, number_cell)
# pool = multiprocessing.Pool(processes=10)
# input_combinations = list(itertools.product(random_cell_ids, pair_threshold_range, radius_range))
# # pool.imap_unordered(calculate_func, cell_id_list)
# results = pool.starmap_async(calculate_func, input_combinations)
# pool.close()
# pool.join()
# results_comp = results.get()
# pickle.dump(results_comp, open("/scratch/project_465001027/spatialformer/data/density_thredshold3.pkl", "wb"))

#save to .h5 
# file_path = os.path.join(save_path, "gene_matrices.h5")
# with h5py.File(file_path, "a") as f:
#     for cell_id, gene_matrix in zip(cell_id_list, results.get()):
#         import pdb; pdb.set_trace()
#         f.create_dataset(str(cell_id), data=gene_matrix)


#testing
cell_id = list(lung_annot_3D_tx_filtered['cell_id'].unique())[0]
data_graph = KNN_Radius_Graph(radius=radius, dataset=lung_annot_3D_tx_filtered, is_3D=True, cell_ID = cell_id, ref_gene = genes)
gene_binary_matrix, gene_freq_matrix, trans_matrix = data_graph.get_gene_matrix(pair_threshold = 2, self_threshold = 2, plot = True) #more than 2 transcripts should be selected
import pdb; pdb.set_trace()
#for gene level
ig_allsum = np.triu(gene_binary_matrix).sum()
ig_rowsum = np.triu(gene_binary_matrix).sum(axis = 1)
ig_mean = ig_rowsum.mean()
ig_mean_nonzero = np.mean(ig_rowsum[ig_rowsum != 0])
#for transcript level
it_allsum = np.triu(trans_matrix).sum()
it_rowsum = np.triu(trans_matrix).sum(axis = 1)
it_mean = it_rowsum.mean()
it_mean_nonzero = np.mean(it_rowsum[it_rowsum != 0])

# import pdb; pdb.set_trace()


#todo, setting the overall genes as the reference







# %%

