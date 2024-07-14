"""
utils.py for focus
"""
import os
import pandas as pd 
import numpy as np
from communities.algorithms import louvain_method
import networkx as nx
import torch
import torch_geometric
from torch_geometric.utils import to_undirected, to_dense_adj, remove_self_loops
import random
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from scipy.sparse import coo_matrix


def unique_list_mapping_to_one_hot(unique_list: List, target_list: List)-> np.array:
    """\
        Convert a list of Unique list to one hot vector.
    """
    unique_elements = sorted(set(unique_list))
    element_to_index = {element: index for index, element in enumerate(unique_elements)}
    
    one_hot_encodings = []
    for target_element in target_list:
        if target_element not in element_to_index:
            raise ValueError("Target element not found in unique list.")
        
        one_hot_vector = [0] * len(unique_elements)
        target_index = element_to_index[target_element]
        one_hot_vector[target_index] = 1
        one_hot_encodings.append(one_hot_vector)
    return np.array(one_hot_encodings)


def find_subcellular_domains(cell_data: pd.DataFrame,
                             transcript_data: pd.DataFrame) -> pd.DataFrame:
    """\
    Find the subcellular domains of a cell.
    
    Args:
        cell_data: pd.DataFrame
            columns: "cell_boundaries", "nucleus_boundaries"
        transcript_data: pd.DataFrame
            columns: "x", "y", "gene", 
        
    Returns:
        subcellular_domains: the subcellular domains of a cell
    """
    pass

def one_graph_splits(data, idx: int = 0):
    """\
        Return: bool, whether edge is intra subgraph
    """
    edge_index = data.edge_index
    undirected_dege_index = to_undirected(edge_index)
    try:
        adj = to_dense_adj(undirected_dege_index).cpu().numpy()[0]
    except:
        adj = np.zeros((data.num_nodes, data.num_nodes), dtype=int)
    subgraphs = louvain_method(adj)[0]
    node_group = torch.zeros(data.num_nodes, dtype=torch.long)
    for i in range(len(subgraphs)):
        for node in subgraphs[i]:
            node_group[node] = i     
    intra_edge = torch.tensor([node_group[edge_index[0][i]]==node_group[edge_index[1][i]] for i in range(data.num_edges)], dtype=torch.bool)
    return intra_edge, node_group

def one_graph_splits_nx_save(args):
    graph, idx, dataset_name, save_path = args 
    edge_mask, node_group, idx = one_graph_splits_nx(graph, idx)
    with open('{}/one_graph_mask/{}_{}.mat'.format(save_path, dataset_name, idx), 'wb') as edge_mask_file:
        torch.save(edge_mask, edge_mask_file)
    with open('{}/one_graph_split/{}_{}.mat'.format(save_path, dataset_name, idx), 'wb') as node_group_file:
        torch.save(node_group, node_group_file)
    

    

def one_graph_splits_nx(graph,  idx: int = 0, seed: int = 42):
    '''
    output: bool, whether edge is intra subgraph
    '''
    
    edge_index = graph.edge_index
    x = graph.x
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    try:
        nx_partitions = nx.algorithms.community.louvain_communities(g, seed=seed)
    except:
        return one_graph_splits(graph, idx)
    node_group = torch.zeros(data.num_nodes, dtype=torch.long)
    for i in range(len(nx_partitions)):
        for node in nx_partitions[i]:
            node_group[node] = i
    intra_edge = torch.tensor([node_group[edge_index[0][i]]==node_group[edge_index[1][i]] for i in range(data.num_edges)], dtype=torch.bool)
    return intra_edge, node_group, idx



def one_graph_splits_feature(graph,  seed: int = 42):
    '''
    output: bool, whther edge is intra subgraph
    '''
    edge_index = graph.edge_index
    x = graph.x
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    try:
        partition = (set(torch.nonzero((x>0).view(-1)).view(-1).numpy()), set(torch.nonzero((x==0).view(-1)).view(-1).numpy()))
        nx_partitions = partition
    except:
        return one_graph_splits(graph)
    node_group = torch.zeros(data.num_nodes, dtype=torch.long)
    for i in range(len(nx_partitions)):
        for node in nx_partitions[i]:
            node_group[node] = i       
    intra_edge = torch.tensor([node_group[edge_index[0][i]]==node_group[edge_index[1][i]] for i in range(data.num_edges)], dtype=torch.bool)
    return intra_edge, node_group

def multi_graph_split_nx(data_list: List) -> List:
    mask_list = []
    split_list = []
    for data in data_list:
        intra_edge, node_group = one_graph_splits_nx(data)
        mask_list.append(intra_edge)
        split_list.append(node_group)
    return mask_list, split_list

def multi_graph_split(data_list: List) -> List:
    result = []
    for data in data_list:
        intra_edge, _ = one_graph_splits(data)
        result.append(intra_edge)
    return result


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def split_data(adata, train_proportion=0.7, test_proportion=0.2, validation_proportion=0.1):
    """
    Splits the data in an AnnData object into training, testing, and validation sets.
    
    Parameters:
    adata (AnnData): The annotated data matrix.
    train_proportion (float): Proportion of the data to be used for training.
    test_proportion (float): Proportion of the data to be used for testing.
    validation_proportion (float): Proportion of the data to be used for validation.
    
    Returns:
    AnnData: The AnnData object with an additional column in `adata.obs` indicating the split.
    """
    # Ensure the proportions sum to 1
    assert train_proportion + test_proportion + validation_proportion == 1.0, "Proportions must sum to 1."

    # Get the total number of cells
    total_cells = len(adata.obs)

    # Determine the number of cells for each split
    num_train = int(total_cells * train_proportion)
    num_test = int(total_cells * test_proportion)
    num_validation = total_cells - num_train - num_test

    # Shuffle the cells
    np.random.seed(42)  # For reproducibility
    shuffled_indices = np.random.permutation(total_cells)

    # Assign cells to each split
    train_indices = shuffled_indices[:num_train]
    test_indices = shuffled_indices[num_train:num_train + num_test]
    validation_indices = shuffled_indices[num_train + num_test:]
    adata.obs["Split"] = "train"
    # Ensure 'Split' is a Categorical column
    if not pd.api.types.is_categorical_dtype(adata.obs["Split"]):
        adata.obs["Split"] = adata.obs["Split"].astype("category")
    # Add new categories if not already present
    new_categories = ["test", "validation"]
    adata.obs["Split"] = adata.obs["Split"].cat.add_categories(new_categories)
    # Assign "test","validation" to the specified indices
    adata.obs.iloc[test_indices, adata.obs.columns.get_loc("Split")] = "test"
    adata.obs.iloc[validation_indices, adata.obs.columns.get_loc("Split")] = "validation"

    return adata

def binary_to_coo_matrix(binary_matrix : np.array):

    # Find the indices where the elements are non-zero
    row, col = np.nonzero(binary_matrix)

    # Gather the non-zero elements. Since it's a binary matrix, these will all be 1s.
    data = binary_matrix[row, col]

    # Create the COO format sparse matrix
    sparse_matrix = coo_matrix((data, (row, col)), shape=binary_matrix.shape)

    return sparse_matrix

def coo_to_binary_matrix(group_shape, data, row, col):
    # Create an empty binary matrix with the same shape as the sparse matrix
    binary_matrix = np.zeros(group_shape, dtype=int)

    # Fill in the ones at the indices where the sparse matrix has non-zero elements
    # import pdb; pdb.set_trace()
    if len(data) > 0:
        binary_matrix[row,col] = data

    return binary_matrix

def read_h5(file_object, cell_id):
    # print("running the cell_id:", cell_id)
    # Get the data
    grp = file_object[cell_id]
    grp_shape = grp.attrs["shape"]
    row = grp['row'][:]
    col = grp['col'][:]
    data = grp['data'][:]
    sparse_matrix = coo_matrix((data, (row, col)), shape=grp_shape)
    # int_matrix = coo_to_binary_matrix(grp_shape, data, row, col)
    return sparse_matrix



def complete_masking(batch, p, n_tokens):
    
    padding_token = 0
    cls_token = 2
    mask_token = 353

    indices = batch['indices']
    # import pdb; pdb.set_trace()
    # indices = torch.where(indices == 0, torch.tensor(padding_token), indices) # 0 is originally the padding token, we change it to 1
    # batch['indices'] = indices

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p) # mask indices with probability p, represent mask as 0 (15%), 85% as 1, without padding tokens
    
    # mask = torch.where(mask == 1, mask_token, mask)
    
    masked_indices = indices * mask # masked_indices, mute masked sites
    #embedding the mask token
    masked_indices = torch.where(masked_indices == 0, mask_token, masked_indices)

    masked_indices = torch.where(indices != padding_token, masked_indices, indices) # we just mask non-padding indices
    mask = torch.where(indices == padding_token, torch.tensor(padding_token), mask) # the mask sequence with the padding tokens
    # so we make the mask of all PAD tokens to be 1 so that it's not taken into account in the loss computation
    
    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_indices = torch.where(indices != cls_token, masked_indices, indices) # same with CLS, no CLS token can be masked
    mask = torch.where(indices == cls_token, torch.tensor(padding_token), mask) # we change the mask so that it doesn't mask any CLS token
    #setting the 0 to the mask tokens
    mask = torch.where(mask == 0, mask_token, mask)

    mask = torch.where(indices == padding_token, torch.tensor(padding_token), mask)
   


    
    # 80% of masked indices are masked
    # 10% of masked indices are a random token
    # 10% of masked indices are the real token
    # import pdb; pdb.set_trace()
    #10 means the start token of the real gene names
    random_tokens = torch.randint(10, n_tokens, size=masked_indices.shape, device=masked_indices.device)
    random_tokens = random_tokens * torch.bernoulli(torch.ones_like(random_tokens) * 0.1).type(torch.int64) 
    random_tokens = torch.where(random_tokens == 0, mask_token, random_tokens)
    masked_indices = torch.where(masked_indices == mask_token, random_tokens, masked_indices) # put random tokens just in the previously masked tokens

    same_tokens = indices.clone()
    same_tokens = same_tokens * torch.bernoulli(torch.ones_like(same_tokens) * 0.1).type(torch.int64)
    same_tokens = torch.where(same_tokens == 0, mask_token, same_tokens)
    masked_indices = torch.where(masked_indices == mask_token, same_tokens, masked_indices) # put same tokens just in the previously masked tokens
    masked_indices = torch.where(indices != padding_token, masked_indices, indices) # don't mask the padding sites
    batch['masked_indices'] = masked_indices
    batch['mask'] = mask

    return batch





