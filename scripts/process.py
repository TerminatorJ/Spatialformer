"""transform npy file into torch_geometric.data.Data"""

import glob
import json
import os
import numpy as np 
import torch
import threading
from tqdm import tqdm
import pandas as pd 
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import multiprocessing as mp 
from multiprocessing import Process
from scipy.spatial import distance_matrix 
from scipy.spatial import KDTree 
import networkx as nx 
import community as community_louvain
from scipy.linalg import block_diag
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]
sys.path.append(os.path.join(p_path, "utils"))
from utils import unique_list_mapping_to_one_hot, one_graph_splits_nx
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


class KNN_Radius_Graph(object):
    def __init__(self, 
                 radius: float, 
                 dataset: pd.DataFrame, 
                 is_3D: bool = False,
                 cell_ID: str = 'cell_ID',
                 gene_column: str = 'gene',
                 ref_gene = None
                 ) -> None:
        """\
            KNN radius Graph to find the nearest neighbors of each node in the graph.
        
        Args: 
            radius: the radius of spatial transcript neighbor threshold
            dataset_name: the name of the dataset
            is_3D: check the dimension of transcript coordinates
            cell_ID: cell ID, "31-0"
            gene_column: "gene" : column name of transcript dataset
        """    
        self.radius = radius 
        self.dataset = dataset
        self.is_3D = is_3D
        self.cell_ID = cell_ID
        self.gene_column = gene_column
        self.gene_list = self.gene_list()
        self.selected_cell_data = self._data_process()
        self.selected_genes = None
        self.ref_gene = ref_gene

        
    def _data_process(self) -> pd.DataFrame:
        """
            Find the data of a target cell
        Args: 
            dataset: dataset
            cell_ID: the ID of the target cell
            
        Returns:
            pd.DataFrame: the data of the target cell
            
        """
        return self.dataset[self.dataset['cell_id'] == self.cell_ID]
    
    def gene_list(self) -> List[str]:
        """
           Find the gene list of the dataset.
        
        Args: 
            dataset: dataset
        Returns:
            gene_list: the list of genes in the dataset
        """
        return sorted(self.dataset['gene'].unique().tolist())
        
    def get_gene_matrix(self, pair_threshold, self_threshold, plot) -> np.array:
        """
           Find the edge index of the graph of a selected cell.
        
        Args:
            pair_threshold: The thredshold number of different transcripts that in the subgroups of transcripts
            self_threshold: The thredshold number of same transcripts that in the subgroups of transcripts
        Returns:
            gene_matrix: the edge index of the graph of a selected cell
        """
        
        #getting all the vertexes
        selected_data = self.selected_cell_data
        if self.is_3D is True:
            r_c = np.array(selected_data[['x', 'y', 'z']])

        else:
            r_c = np.array(selected_data[['x', 'y']])
        total_transcript = len(r_c)
        #building the KD trees
        kdtree = KDTree(r_c)
        G = nx.Graph()
        # Add all nodes to the graph initially
        for i in range(len(r_c)):
            G.add_node(i)
        for i, x in enumerate(r_c):
            idx = kdtree.query_ball_point(x, self.radius)
            for j in idx:
                if i < j:
                    G.add_edge(i, j)
        #generate the symmetric matrix
        adj_matrix = nx.to_numpy_array(G, nodelist=range(len(r_c)))
        # rows, cols = np.where(adj_matrix == 1)
        # edge_index = np.array([rows, cols])
        
        #using louvain method to find the clusters
        partition = community_louvain.best_partition(G)
        # Extract subgroups based on the detected communities
        communities = {}
        for node, comm in partition.items():
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(node)
        if plot:
            # Sort nodes within each community and create a new ordering
            new_order = []
            for comm in sorted(communities.keys()):
                new_order.extend(sorted(communities[comm]))
            # Permute the adjacency matrix according to the new node ordering
            permuted_adj_matrix = adj_matrix[np.ix_(new_order, new_order)]
            # Extract blocks
            blocks = []
            start = 0
            for comm in sorted(communities.keys()):
                size = len(communities[comm])
                block = permuted_adj_matrix[start:start+size, start:start+size]
                blocks.append(block)
                start += size
            # Construct the block-diagonal matrix
            block_diag_matrix = block_diag(*blocks)
            plt.figure(figsize=(10, 8))
            sns.heatmap(block_diag_matrix, annot=False, cmap='viridis', cbar=True)
            plt.title('Block-Diagonal Matrix Heatmap')
            plt.savefig(f"/scratch/project_465001027/spatialformer/figure/{self.cell_ID}_transcripts_heatmap.png", dpi = 300)
        
        #get genes of current cell
        selected_genes = np.sort(self.node_type()) #ordered references for every transcripts
        #convert the communities from index to gene
        gene_pair_counter = {}
        gene_pair_set = set()
        for comm in sorted(communities.keys()):
            transcript_index = communities[comm]
            genes_comm = selected_genes[transcript_index]
            genes_counter = dict(Counter(genes_comm))
            for idx_l, gene_l in enumerate(genes_comm):
                if genes_counter[gene_l] >= self_threshold:
                    for idx_r, gene_r in enumerate(genes_comm[idx_l + 1:]):
                        if gene_l != gene_r:
                            if genes_counter[gene_r] >= pair_threshold:
                                sorted_pair = sorted([gene_l, gene_r])
                                pair_str = sorted_pair[0] + "_" + sorted_pair[1]
                                gene_pair_set.add(pair_str)
                                gene_pair_counter[pair_str] = genes_counter[gene_l] + genes_counter[gene_r]
        # unique_genes =  list(sorted(set(selected_genes))) 
        unique_genes = list(sorted(self.ref_gene))
        gene_num = len(unique_genes)
        
        gene_binary_matrix = np.zeros((gene_num, gene_num))
        gene_freq_matrix = np.zeros((gene_num, gene_num))
        for pair_str in gene_pair_set:
            gene1_idx = unique_genes.index(pair_str.split("_")[0])
            gene2_idx = unique_genes.index(pair_str.split("_")[1])
            gene_binary_matrix[gene1_idx, gene2_idx] = 1  
        for pair_str, count in gene_pair_counter.items():
            gene1_idx = unique_genes.index(pair_str.split("_")[0])
            gene2_idx = unique_genes.index(pair_str.split("_")[1])
            gene_freq_matrix[gene1_idx, gene2_idx] = count/total_transcript #normalized by overall counts of different cells
            #get the symetric matrix
        gene_binary_matrix = gene_binary_matrix + gene_binary_matrix.T
        gene_freq_matrix = gene_freq_matrix + gene_freq_matrix.T
        if plot:
            plt.figure(figsize=(20, 8))

            # Plot for gene_binary_matrix
            plt.subplot(1, 2, 1)
            sns.heatmap(gene_binary_matrix, annot=False, cmap='viridis', cbar=True)
            plt.title('Block-Diagonal Matrix Heatmap for genes (Binary)')

            # Plot for gene_freq_matrix
            plt.subplot(1, 2, 2)
            sns.heatmap(gene_freq_matrix, annot=False, cmap='viridis', cbar=True)
            plt.title('Block-Diagonal Matrix Heatmap for genes (Frequency)')

            plt.savefig(f"/scratch/project_465001027/spatialformer/figure/{self.cell_ID}_gene_combined_heatmap.png", dpi=300)

        #TODO, insert the genes within the reference list
        
        return gene_binary_matrix, gene_freq_matrix, adj_matrix
   
    
    def node_type(self):
        """
            Node type: transcript type: Gene name
        """
        return np.array(self.selected_cell_data[self.gene_column])
    
    def node_spatial(self):
        if self.is_3D is True:
            return np.array(self.selected_cell_data[['x', 'y', 'z']]).T
        else:
            return np.array(self.selected_cell_data[['x', 'y']]).T
    
    def graph_label(self):
        return self.selected_cell_data[self.selected_cell_data['cell_ID'] == self.cell_ID]['cell_type'].unique()[0]



# class Spatial_exp:
#     def __init__(self, 
#                  dataset: pd.DataFrame,
                 
#                  ):
        