"""
utils.py for spatialformer
"""
import os
import pandas as pd 
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from communities.algorithms import louvain_method
import networkx as nx
import torch
import time
from tqdm import tqdm
from sklearn.model_selection import KFold

from concurrent.futures import ProcessPoolExecutor
from datasets.distributed import split_dataset_by_node
# import torch_geometric
# from torch_geometric.utils import to_undirected, to_dense_adj, remove_self_loops
import random
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from scipy.sparse import coo_matrix
from typing import Dict, Optional, Union
from scipy.spatial import KDTree 
import matplotlib.pyplot as plt
import pickle
import random
from torch.utils.data import Dataset, IterableDataset
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from datasets import load_from_disk, interleave_datasets, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType

# import logging


def get_adj(sample_dataset = None, anndata = None, radius = 10, plot = False, sym = True):

    if anndata is None:
        r_c = np.array(list(zip(sample_dataset["centroid_x"],sample_dataset["centroid_y"])))
        cell_ids = sample_dataset["Cell_Ids"]
    elif anndata is not None:
        # Getting the coordinate from the anndata
        r_c = np.array(list(zip(anndata.obsm["spatial"][:,0], anndata.obsm["spatial"][:,1])))
        # Getting the cell id
        cell_ids = anndata.obs.index

    
    if sym:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    kdtree = KDTree(r_c)
    # Add all nodes to the graph initially
    for i in range(len(r_c)):
        G.add_node(i)
    for i, x in enumerate(r_c):
        idx = kdtree.query_ball_point(x, radius)
        for j in idx:
            if i < j:
                G.add_edge(i, j)
    sparse_adj_matrix = nx.to_scipy_sparse_array(G, nodelist=range(len(r_c)))
    sparse_adj = sparse_adj_matrix.tocoo()

    if plot:
        x = r_c[:,0]
        y = r_c[:,1]
        # Now, for plotting
        plt.figure(figsize=(8, 8))
        pos = {i: (r_c[i][0], r_c[i][1]) for i in range(len(r_c))}  # Position of each node

        # Draw the graph with edges
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=5, font_size=1)

        # Add title and labels
        plt.title('Graph of Cells with Connections')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid()
        plt.axis('auto')
        plt.gca().set_aspect(1.0, adjustable='datalim') 
        plt.savefig(f"/scratch/project_465001027/Spatialformer/Figure/selected_coord_{radius}.png", dpi=1000)
        plt.show()


    return sparse_adj, cell_ids





def build_index(dataset):
    # Step 2: Construct the index mapping
    sample_cell_index = {}
    # import pdb; pdb.set_trace()
    cell_ids = dataset.select_columns(["Cell_Ids"])
    sample_names = dataset.select_columns(["Sample_Names"])
    for index in tqdm(range(len(cell_ids))):
        cell_id = cell_ids[index]["Cell_Ids"]  # Access the cell_id for the current row
        sample_name = sample_names[index]["Sample_Names"]
        sample_cell_index.setdefault(sample_name, {}).setdefault(cell_id, index)
    return sample_cell_index

def get_index(dataset, save_file):
    if not os.path.exists(save_file):
        sample_cell_index = build_index(dataset) #{sample: {cell : index}}
        #saving the index
        pickle.dump(sample_cell_index, open(save_file, "wb"))
    else:
        sample_cell_index = pickle.load(open(save_file, "rb"))

    return sample_cell_index



class CustomIterableDataset:
    def __init__(self, datapath):
        '''
        datapath: cache path
        split: which split to access ('train' or 'test')
        shuffle: whether to shuffle the dataset
        '''
        all_files = os.listdir(datapath)  # Corrected method name
        self.datasets_paths = [os.path.join(datapath, file) for file in all_files if file.endswith("pair")]

    def load_dataset(self, path):
        dataset = load_from_disk(path)
        return dataset

    def get_all(self):
        # Iterate through datasets
        train_iters = []
        test_iters = []
        for datasets_path in tqdm(self.datasets_paths[:2]):

            # import pdb; pdb.set_trace()
            try:
                dataset = self.load_dataset(datasets_path)
            except FileNotFoundError:
                print(f"{datasets_path} is not a valid dataset")
            # Split the dataset into train/test
            split_dataset = dataset.train_test_split(test_size=0.005, seed=42)

            # Convert to IterableDataset
            train_iter_dataset = split_dataset["train"].to_iterable_dataset(num_shards=64)
            train_iter_dataset = train_iter_dataset.shuffle(buffer_size=10_000, seed=42)
            test_iter_dataset = split_dataset["test"].to_iterable_dataset(num_shards=64)

            # if self.shuffle:
            #     # import pdb; pdb.set_trace()
            #     iter_dataset = iter_dataset.shuffle(buffer_size=10_000, seed=42)
            train_iters.append(train_iter_dataset)
            test_iters.append(test_iter_dataset)
        # import pdb; pdb.set_trace()
        all_train_iter_dataset = concatenate_datasets(train_iters)
        #shuffle the train and test iterable dataset, make the batch contain different samples
        # all_train_iter_dataset = all_train_iter_dataset.shuffle(buffer_size=10_000, seed=42)

        all_test_iter_dataset = concatenate_datasets(test_iters)
        # all_test_iter_dataset = all_test_iter_dataset.shuffle(buffer_size=10_000, seed=42)
        # train_interleave_dataset = interleave_datasets(train_iters)
        # test_interleave_dataset = interleave_datasets(test_iters)
        return all_train_iter_dataset, all_test_iter_dataset

# dataloader = torch.utils.data.DataLoader(ids, num_workers=4)



class NonRedundantSampler:
    def __init__(self, total_samples, batch_size):
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.indices = np.random.permutation(total_samples)  # Shuffle indices
        self.current_index = 0  # Track the index for sampling

    def sample(self):
        # Check if we've finished all indices
        if self.current_index >= self.total_samples:
            self.indices = np.random.permutation(self.total_samples)  # Reshuffle
            self.current_index = 0  # Reset index for the new sample

        # Determine the end of the current batch
        end_index = min(self.current_index + self.batch_size, self.total_samples)
        batch_indices = self.indices[self.current_index:end_index]

        # Update the current index
        self.current_index = end_index

        return batch_indices


class DynamicHuggingFaceDataset(IterableDataset):
    def __init__(self, datapath, split):
        '''
        datapath: cache path
        split: which split to access ('train' or 'test')
        shuffle: whether to shuffle the dataset
        '''
        # self.current_index = resume_index
        all_files = os.listdir(datapath)  # Corrected method name
        # self.datasets_paths = [os.path.join(datapath, file) for file in all_files if file.endswith("pair")] # train all the slides
        self.datasets_paths = [os.path.join(datapath, file) for file in all_files if (("TIL" in file) & ("pair" in file)) or (("THD" in file) & ("pair" in file)) or (("VU" in file) & ("pair" in file))]
        print("number of training slides:", len(self.datasets_paths))
        # self.datasets_paths = [os.path.join(datapath, file) for file in all_files if file == "xenium_VUILD102LF_pair"]
        self.split = split
        self.yield_counter = 0 
    def load_dataset(self, path):
        dataset = load_from_disk(path)
        return dataset
    def __len__(self):
        counter = 0
        for path in self.datasets_paths:
            dataset = self.load_dataset(path)
            counter += dataset.shape[0]
        return counter

    def __iter__(self):
        
        rank = torch.distributed.get_rank()  # Get current process rank
        # random.seed(rank)
        dynamic_seed = rank + int(time.time()/1000) 
        random.seed(dynamic_seed)

        while True:  # Continuous iteration
            import pdb; pdb.set_trace()
            datasets_path = random.choice(self.datasets_paths)  #  choose a dataset path
            print("current datasets:", datasets_path)
            if "THD0008" not in datasets_path:
                try:
                    dataset = self.load_dataset(datasets_path)
                except FileNotFoundError:
                    print(f"{datasets_path} is not a valid dataset")
                    continue
                
                # Optionally limit the dataset size, or select a subset if needed
                # dataset = dataset.select(range(1000))  # Load a subset for processing
                # Determine the total number of samples available
                # left_sample = 5000
                total_samples = len(dataset)
                n_samples = min(5000, total_samples)  # Ensure not to exceed available samples

                # Randomly select 5000 samples
                selected_indices = np.random.choice(total_samples, size=n_samples, replace=False)
                dataset = dataset.select(selected_indices)  # Load a subset for processing
                split_dataset = dataset.train_test_split(test_size=0.001, seed=42)

                if self.split == "train":
                    # import pdb; pdb.set_trace()
                    print(split_dataset["train"])
                    # iter_dataset = split_dataset["train"].to_iterable_dataset(num_shards=64).shuffle(buffer_size=10_000)#for randomization
                    iter_dataset = split_dataset["train"].to_iterable_dataset()#
                elif self.split == "val":
                    # import pdb; pdb.set_trace()
                    print(split_dataset["test"])
                    # iter_dataset = split_dataset["test"].to_iterable_dataset(num_shards=1)
                    iter_dataset = split_dataset["test"].to_iterable_dataset()

                # Yield samples from the current dataset
                for sample in iter_dataset:
                    # import pdb; pdb.set_trace()
                    yield sample

class DynamicHuggingFaceDataset2(IterableDataset):
    def __init__(self, datapath, split, gpu_num):
        '''
        datapath: cache path
        split: which split to access ('train' or 'test')
        shuffle: whether to shuffle the dataset
        '''
        # self.current_index = resume_index
        all_files = os.listdir(datapath)  # Corrected method name
        self.datasets_paths = [os.path.join(datapath, file) for file in all_files if file.endswith("pair")] # train all the slides
        # self.datasets_paths = [os.path.join(datapath, file) for file in all_files if (("TIL" in file) & ("pair" in file)) or (("THD" in file) & ("pair" in file)) or (("VU" in file) & ("pair" in file))]
        print("number of training slides:", len(self.datasets_paths))
        counter = 0
        for path in self.datasets_paths:
            dataset = self.load_dataset(path)
            
            try:
                counter += dataset.shape[0]
            except:
                counter += dataset["train"].shape[0]
        self.datasize = counter
        # self.datasets_paths = [os.path.join(datapath, file) for file in all_files if file == "xenium_VUILD102LF_pair"]
        self.split = split
        self.gpu_num = gpu_num
    def load_dataset(self, path):
        dataset = load_from_disk(path)
        return dataset
    def __len__(self):
        return int(self.datasize/self.gpu_num)

    def __iter__(self):
        
        rank = torch.distributed.get_rank()  # Get current process rank
        # random.seed(rank)
        # dynamic_seed = rank + int(time.time()/1000) 
        # random.seed(dynamic_seed)
        # world_size = torch.distributed.get_world_size()  # Total number of processes (GPUs)

        for datasets_path in self.datasets_paths:
            # import pdb; pdb.set_trace()
            print("current datasets:", datasets_path)
            if "THD0008" not in datasets_path:
                try:
                    dataset = self.load_dataset(datasets_path)["train"]
                except FileNotFoundError:
                    print(f"{datasets_path} is not a valid dataset")
                    continue
                
                split_dataset = dataset.train_test_split(test_size=0.001, seed=42)
                data_subset = split_dataset['train'] if self.split == 'train' else split_dataset['test']
                # Implement sharding
                num_samples = len(data_subset)
                samples_per_worker = num_samples // self.gpu_num
                # Indices for the current worker
                start_idx = rank * samples_per_worker
                end_idx = start_idx + samples_per_worker if rank != self.gpu_num - 1 else num_samples
                # Slice the dataset for the current worker
                print(f"rank:{rank} is selecting: from {start_idx} to {end_idx}. total: {num_samples}, samples per worker: {samples_per_worker}")
                sliced_dataset = data_subset.select(range(start_idx, end_idx))
                if self.split == "val":
                    iter_dataset = sliced_dataset.to_iterable_dataset()
                elif self.split == "train":
                    iter_dataset = sliced_dataset.to_iterable_dataset()
                
                # Yield samples from the current dataset
                for sample in iter_dataset:
                    # import pdb; pdb.set_trace()
                    yield sample
 

       
class DynamicHuggingFaceDatasetFast(IterableDataset):
    def __init__(self, sample_dataset, pair_index, labelget_adjs, split):
        '''
        huggingface dataset, this is used for fine-tuning
        '''
        self.sample_dataset  = sample_dataset
        self.pair_index = pair_index
        self.labels = labels
        self.split = split
        self.gpus = torch.cuda.device_count()
        self.num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        self.gpu_num = self.gpus*self.num_nodes
    def load_dataset(self, path):
        dataset = load_from_disk(path)
        return dataset

    def get_iter(self, rand_seed):
        # import pdb; pdb.set_trace()
        np.random.seed(rand_seed)
        shuffled_indices = np.random.permutation(self.pair_index.shape[0])
        left_idxs = self.pair_index[:,0][shuffled_indices]
        right_idxs = self.pair_index[:,1][shuffled_indices]
        left_dataset = self.sample_dataset.select(left_idxs)
        right_dataset = self.sample_dataset.select(right_idxs)

        self.pair_index = self.pair_index[shuffled_indices]
        self.labels = self.labels[shuffled_indices]
        return left_dataset, right_dataset
    def __len__(self):
        return int(self.pair_index.shape[0]/self.gpu_num)
    def __iter__(self):
        
        rank = torch.distributed.get_rank()  # Get current process rank
        # random.seed(rank)
        # random_value = random.randint(1, 1000)
        dynamic_seed = rank + int(time.time()) 
        random.seed(dynamic_seed)
        np.random.seed(dynamic_seed)
        # left_dataset, right_dataset = self.get_iter(dynamic_seed)
        # import pdb; pdb.set_trace()
        if self.split != "test":
            while True:
                # import pdb; pdb.set_trace()
                total_samples = len(self.labels)
                n_samples = min(5000, total_samples)
                selected_indices = np.random.choice(total_samples, size=n_samples, replace=False)
                left_idxs = self.pair_index[:,0][selected_indices]
                right_idxs = self.pair_index[:,1][selected_indices]

                left_dataset = self.sample_dataset.select(left_idxs)
                right_dataset =  self.sample_dataset.select(right_idxs)
                selected_pairs = self.pair_index[selected_indices]
                selected_labels = self.labels[selected_indices]
                left_iter_dataset = left_dataset.to_iterable_dataset(num_shards=64)
                right_iter_dataset = right_dataset.to_iterable_dataset(num_shards=64)
                for left_sample,right_sample,(left_idx, right_idx), label in zip(left_iter_dataset, right_iter_dataset, selected_pairs, selected_labels):

                    yield left_sample, right_sample, label, left_idx, right_idx
        else:
            # import pdb; pdb.set_trace()
            # Implement sharding
            num_samples = self.pair_index.shape[0]
            samples_per_worker = num_samples // self.gpu_num
            # Indices for the current worker
            start_idx = rank * samples_per_worker
            end_idx = start_idx + samples_per_worker if rank != self.gpu_num - 1 else num_samples
            # Slice the dataset for the current worker
            print(f"rank:{rank} is selecting: from {start_idx} to {end_idx}. total: {num_samples}, samples per worker: {samples_per_worker}")
            # import pdb; pdb.set_trace()
            left_idxs = self.pair_index[:,0][start_idx:end_idx]
            right_idxs = self.pair_index[:,1][start_idx:end_idx]

            left_dataset = self.sample_dataset.select(left_idxs)
            right_dataset =  self.sample_dataset.select(right_idxs)
            left_iter_dataset = left_dataset.to_iterable_dataset()
            right_iter_dataset = right_dataset.to_iterable_dataset()

            #subset the index and labels
            subset_index = self.pair_index[start_idx:end_idx]
            subset_labels = self.labels[start_idx:end_idx]
            for left_sample,right_sample,(left_idx, right_idx), label in zip(left_iter_dataset, right_iter_dataset, subset_index, subset_labels):

                yield left_sample, right_sample, label, left_idx, right_idx

        
class DynamicHuggingFaceDatasetEval(IterableDataset):
    def __init__(self, datapath, kfold = False, cur_fold = False, split = False):
        '''
        huggingface dataset
        '''
        self.datapath  = datapath
        self.kfold = kfold
        self.cur_fold = cur_fold
        self.split = split
        self.dataset = load_from_disk(datapath)

    def kfold_split(self, dataset):
        sample_num = len(dataset)  # Use len() for datasets

        kf = KFold(n_splits=self.kfold, shuffle=True, random_state=42)
        for fold_index, (train_index, test_index) in enumerate(kf.split(range(sample_num))):
            if self.cur_fold == fold_index:
                train_dataset = dataset.select(train_index)
                test_dataset = dataset.select(test_index)
                print(f"Fold {self.cur_fold}, Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
                return train_dataset, test_dataset

        raise ValueError("Current fold index is out of bounds.")
    def __len__(self):

        return len(self.dataset)

    
    def __iter__(self):

        try:
            # Get the total number
            total_number = len(self.dataset)

            # Shuffle the cells
            np.random.seed(42)  # For reproducibility
            shuffled_indices = np.random.permutation(total_number)
            if self.kfold:
                train_dataset, test_dataset = self.kfold_split(self.dataset)
                if self.split == "train":
                    yield from train_dataset
                elif self.split == "test":
                    yield from test_dataset
            else:
                #WARNING: if you use the to_iterable_dataset method, you won't get the whole datasets as the right order.
                dataset_shuffle = self.dataset.select(shuffled_indices)
                iter_dataset = dataset_shuffle.to_iterable_dataset(num_shards=64)
                yield from iter_dataset 

        except FileNotFoundError:
            print(f"{self.datapath} is not a valid dataset")

        

        


class GetPairs(Dataset):
    def __init__(self, adjacency_matrix:coo_matrix, num_workers:int):
        self.adj_matrix = adjacency_matrix
        self.adj_matrix_csr = self.adj_matrix.tocsr()
        self.num_workers = num_workers
        self.num_nodes = adjacency_matrix.shape[0]
        # Create positive pairs (edges from adjacency matrix)
        self.positive_pairs = np.column_stack((adjacency_matrix.row, adjacency_matrix.col))

        # Get the number of positive pairs
        num_positive = len(self.positive_pairs)

        # Create negative pairs (we will sample after ensuring all nodes are covered)
        # self.negative_pairs = self.create_negative_pairs(num_positive)
        self.negative_pairs = self.create_even_negative_pairs(batch_size=1000, num_workers=num_workers) #create even pos and neg pairs by nodes
        #make sure all nodes included
        # import pdb; pdb.set_trace()
        positive_covered_nodes = {node for pair in self.positive_pairs for node in pair}
        negative_covered_nodes = {node for pair in self.negative_pairs for node in pair}
        # import pdb; pdb.set_trace()
        all_node = positive_covered_nodes.union(negative_covered_nodes)
        nodes_left_num = self.num_nodes - len(all_node)
        # import pdb; pdb.set_trace()
        if self.num_nodes != len(all_node):
            print(f"ERROR: There are {nodes_left_num} nodes not included")
        print(f"The total number of pairs: \npositive pair:{len(self.positive_pairs)}\nnegative pair:{len(self.negative_pairs)}")

        self.all_pairs = np.concatenate([self.positive_pairs, self.negative_pairs])
        self.all_labels = np.concatenate([np.ones(len(self.positive_pairs)), np.zeros(len(self.negative_pairs))])


    def get_reverse(self, pairs):
        reversed_pairs = np.array([[b, a] for a, b in pairs])
        return reversed_pairs

    def select_one_connection_per_node(self, adjacency_matrix):
        # Create a list to hold the positive pairs
        positive_pairs = []
        
        # Assuming adjacency_matrix is in COO format
        rows, cols = adjacency_matrix.row, adjacency_matrix.col
        
        # Dictionary to store one connection per node
        seen_nodes = {}
        
        for row, col in zip(rows, cols):
            if row not in seen_nodes:
                seen_nodes[row] = col  # Take the first connection for this node
                positive_pairs.append((row, col))  # Each connection to store
            
        
        return np.array(positive_pairs)
    def create_negative_pairs(self, num_positive):
        """Generate negative pairs (non-edges) and ensure they are balanced with positive pairs."""
        negative_pairs = []
        # possible_pairs = set((i, j) for i in range(self.num_nodes) for j in range(self.num_nodes) if i != j)
        covered_nodes = {node for pair in self.positive_pairs for node in pair}
        uncovered_nodes = [i for i in range(self.num_nodes) if i not in covered_nodes]
        # Create set of existing positive pairs for quick lookup
        positive_set = set(map(tuple, self.positive_pairs))
        gap_num = num_positive - len(negative_pairs)
        # import pdb; pdb.set_trace()
        # Step 2: Pair remaining uncovered nodes with covered nodes
        if gap_num != 0 :
            for idx, uncovered_node in enumerate(uncovered_nodes):
                if idx < gap_num:  # Ensure we do not exceed covered nodes count
                    # import pdb; pdb.set_trace()
                    # covered_node = list(covered_nodes)[idx]
                    covered_node = random.sample(covered_nodes, 1)[0]
                    pair = (covered_node, uncovered_node)
                    negative_pairs.append(pair)
        gap_num = num_positive - len(negative_pairs)
        add_nodes = {node for pair in negative_pairs for node in pair}
        covered_nodes = covered_nodes.union(add_nodes)
        uncovered_nodes = [i for i in range(self.num_nodes) if i not in covered_nodes]
        # import pdb; pdb.set_trace()
        # Step 3: Pair within the positive pair
        if gap_num != 0 :
            for pair in combinations(covered_nodes, 2):
                if len(negative_pairs) < num_positive:
                    if pair not in positive_set:
                        negative_pairs.append(pair)  # Append the uncovered pair
                else:
                    break
        add_nodes = {node for pair in negative_pairs for node in pair}
        covered_nodes = covered_nodes.union(add_nodes)
        uncovered_nodes = [i for i in range(self.num_nodes) if i not in covered_nodes]
        gap_num = num_positive - len(negative_pairs)                

        #if still not enough
        if gap_num != 0:
            negative_pairs += negative_pairs[:gap_num]
        # import pdb; pdb.set_trace()
        assert num_positive == len(negative_pairs), "The positive and negative pairs should be balanced, please check your codes!!!"
        assert not bool(set(negative_pairs) & positive_set), "The positive pair should not have overlap pairs with negative pairs"
        return np.array(negative_pairs)
    def get_negative_candidates_for_node(self, node1):
        # Get all nodes that are not connected and not itself
        # import pdb;pdb.set_trace()
        # print(node1)
        candidates = np.where(self.adj_matrix_csr[[node1]].toarray()[0] == 0)[0]
        # print(candidates)
        candidates = candidates[candidates != node1]
        return candidates
        
    def generate_negative_pairs_for_batch(self, batch):
        negative_pairs = []
        random.seed(42)
        for node1, num in batch:

            potential_negatives = self.get_negative_candidates_for_node(node1)

            negative_pair = [(node1,neg) for neg in random.sample(list(potential_negatives), k=num)]
            negative_pairs += negative_pair
        return negative_pairs

    def create_even_negative_pairs(self, batch_size=1000, num_workers=4):
        """Generate negative pairs (non-edges) and ensure they are balanced with positive pairs."""
        negative_pairs = []
        # import pdb; pdb.set_trace()
        # first_pos_node1 = set([pair[0] for pair in self.positive_pairs])
        unique_elements, counts = np.unique(self.positive_pairs[:,0], return_counts=True)
        # Combine into a dictionary for easier readability
        first_pos_node1_dict = dict(zip(unique_elements, counts))
        # import pdb; pdb.set_trace()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Processing in batches
            for start in tqdm(range(0, len(first_pos_node1_dict), batch_size)):
            # for start in tqdm(range(0, 2000, batch_size)):
                end = min(start + batch_size, len(first_pos_node1_dict))
                batch = list(first_pos_node1_dict.items())[start:end]
                # Submit the batch processing to the executor
                # print(batch)
                future = executor.submit(self.generate_negative_pairs_for_batch, batch)
                negative_pairs.extend(future.result())  # Collect the result
                # print(future.result())
        # Ensure the count of positive and negative pairs is the same
        if len(negative_pairs) < len(self.positive_pairs):
            print("Warning: Fewer negative pairs than positive pairs.")
        # import pdb; pdb.set_trace()
        return np.array(negative_pairs)
    


def split_dataset(all_pairs, all_labels, n_splits, split_mode = "random", test_size=None,zero_shot_cell_size=None):
    """
    test_size: number of test samples we want to cover, default=None.
    split_mode: random means randomly select the edges using the upper triangle matrix without diagonal values.
                leave_cell_out randomly select the cells and their edges.
    zero_shot_cell_size: if you want to test the zero_shot ability, you should define this cell number 
    """
    # For the sake of demonstration, we'll create a placeholder array for our pairs
    np.random.seed(42)
    # Initialize KFold
    
    pair_num = all_pairs.shape[0]

    
    
    if n_splits:
        if split_mode == "random":
            kf = KFold(n_splits=n_splits, shuffle=False)
            # Prepare to store the train/test pairs
            train_test_splits = {}
            train_test_labels = {}
            #getting the positive pairs
            pos_pairs = all_pairs[: int(pair_num/2)]
            pos_labels = all_labels[: int(pair_num/2)]
            #get the negative pairs
            neg_pairs = all_pairs[int(pair_num/2):]
            neg_labels = all_labels[int(pair_num/2):]
            for fold, (train_index, test_index) in enumerate(kf.split(pos_pairs)):
                #getting train and test pairs
                train_shuffled_indices = np.random.permutation(2*len(train_index)) #for pos and neg shuffle
                test_shuffled_indices = np.random.permutation(2*len(test_index))
                # import pdb; pdb.set_trace()
                pos_train_pairs = pos_pairs[train_index]
                neg_train_pairs = neg_pairs[train_index]

                pos_test_pairs = pos_pairs[test_index[:test_size]]
                neg_test_pairs = neg_pairs[test_index[:test_size]]
                #getting train and test labels
                pos_train_labels = pos_labels[train_index]
                neg_train_labels = neg_labels[train_index]

                pos_test_labels = pos_labels[test_index[:test_size]]
                neg_test_labels = neg_labels[test_index[:test_size]]

                # import pdb; pdb.set_trace()
                #combine all the pairs and labels
                train_pairs = np.vstack([pos_train_pairs, neg_train_pairs])
                train_pairs = train_pairs[train_shuffled_indices]
                test_pairs = np.vstack([pos_test_pairs, neg_test_pairs])
                test_pairs = test_pairs[test_shuffled_indices]


                train_labels = np.hstack([pos_train_labels, neg_train_labels])
                train_labels = train_labels[train_shuffled_indices]
                test_labels = np.hstack([pos_test_labels, neg_test_labels])
                test_labels = test_labels[test_shuffled_indices]

                train_test_splits[fold] = (train_pairs, test_pairs)
                train_test_labels[fold] = (train_labels, test_labels)

                # You can also print the shapes or any other information:
                print(f"Train size: {train_pairs.shape[0]}, Test size: {test_pairs.shape[0]}")
            return train_test_splits, train_test_labels
        elif split_mode == "leave_cell_out":
            #getting all the query cells
            # import pdb; pdb.set_trace()
            cell_ids = np.unique(all_pairs[:,0])
            kf = KFold(n_splits=10, shuffle=False) #to make sure the test cell should be 10%
            train_test_splits = {}
            train_test_labels = {}
            for fold, (train_index, test_index) in enumerate(kf.split(cell_ids)):
                # import pdb; pdb.set_trace()
                if fold < n_splits:
                    train_cell_ids = cell_ids[train_index]
                    test_cell_ids = cell_ids[test_index]
                    # import pdb; pdb.set_trace()
                    #getting the test pairs 
                    test_pairs = all_pairs[np.isin(all_pairs[:,0], test_cell_ids)]
                    test_labels = all_labels[np.isin(all_pairs[:,0], test_cell_ids)]
                    # import pdb; pdb.set_trace()
                    #getting the train pairs and exclude the edges that link to the test nodes
                    #only filter the test cell_id in the pos pairs

                    #first filter out the query cells belong to the test cell ids
                    train_pairs = all_pairs[~np.isin(all_pairs[:,0], test_cell_ids)] #getting the potential positive pairs
                    train_labels = all_labels[~np.isin(all_pairs[:,0], test_cell_ids)]
                    #filter out the key cells that belong to the test cell ids, only filter by the positive pairs - the 1/2 cells
                    mid_point = len(train_pairs) // 2 #split to two half avoid data imbalance of pos and neg after filtering
                    train_pos_pairs = train_pairs[:mid_point, :]
                    train_neg_pairs = train_pairs[mid_point:, :]
                    pos_mask = ~np.isin(train_pos_pairs[:,1], test_cell_ids) #the pos key should not in test
                    neg_mask = ~np.isin(train_neg_pairs[:,1], test_cell_ids) #the neg key should not in test
                    half_mask = pos_mask & neg_mask
                    full_mask = np.hstack([half_mask, half_mask])
                    # import pdb; pdb.set_trace()
                    # kept_mask = ~np.isin(train_pairs[:,1], test_cell_ids)
                    # import pdb; pdb.set_trace()
                    train_pairs = train_pairs[full_mask]
                    train_labels = train_labels[full_mask]
                    assert not np.isin(train_pairs.flatten(), test_cell_ids).any(), "ERROR: The test cell ids should not in the training cell ids"
                    #shuffle the pairs and labels
                    train_shuffled_indices = np.random.permutation(len(train_pairs))
                    test_shuffled_indices = np.random.permutation(len(test_pairs))
                    # import pdb; pdb.set_trace()
                    train_pairs = train_pairs[train_shuffled_indices]
                    train_labels = train_labels[train_shuffled_indices]
                    test_pairs = test_pairs[test_shuffled_indices]
                    test_labels = test_labels[test_shuffled_indices]


                    # import pdb; pdb.set_trace()
                    train_test_splits[fold] = (train_pairs, test_pairs)
                    train_test_labels[fold] = (train_labels, test_labels)
                else:
                    break
                print(f"Train cell number: {train_cell_ids.shape[0]}, Test cell number: {test_cell_ids.shape[0]}")
                print(f"Train size: {train_pairs.shape[0]}, Test size: {test_pairs.shape[0]}")
            return train_test_splits, train_test_labels
                
    else:
        # import pdb; pdb.set_trace()
        
        
        if zero_shot_cell_size != -1:
            #getting the test dataset based on the cell_ids
            cell_ids = np.unique(all_pairs[:,0])
            selected_cell_ids = np.random.choice(cell_ids, size = zero_shot_cell_size, replace=False)

            selected_pairs = all_pairs[np.isin(all_pairs[:,0], selected_cell_ids)] #select positive and negative pairs according to cells
            selected_labels = all_labels[np.isin(all_pairs[:,0], selected_cell_ids)]

            #shuffling the data
            train_shuffled_indices = np.random.permutation(len(selected_pairs))
            selected_pairs = selected_pairs[train_shuffled_indices]
            selected_labels = selected_labels[train_shuffled_indices]
            return selected_pairs, selected_labels
        else:
            return all_pairs, all_labels



class BalanceDataset(Dataset):
    def __init__(self, dataset, pairs, labels):
        self.pairs = pairs
        self.labels = labels
        # self.sample_cell_index = sample_cell_index
        # self.sample_name = sample_name
        self.dataset = dataset
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        left_idx = int(self.pairs[:,0][idx])
        right_idx = int(self.pairs[:,1][idx])
        # sample_cell_index = list(self.sample_cell_index[self.sample_name].values())
        left_dataset = self.dataset.select([left_idx])
        right_dataset = self.dataset.select([right_idx])

        return left_dataset, right_dataset, self.labels[idx]



class CustomDataCollator2(object):
    def __init__(self, directionality, context_length = 1000, padding_idx=0, special_token_num = 4, n_bins = 51, sep_token = 1949, cls_token = 1):
        self.context_length = context_length
        self.padding_idx = padding_idx
        self.directionality = directionality
        self.special_token_num = special_token_num
        self.n_bins = n_bins
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pair_labels = []
        self.pair1_length = None
        self.last_mtx_length = None
        self.batch = None
        self.full_tokens = None
        self.full_exp = None
        self.norm_exp = None
        self.gg_mtx_p = None
    def __call__(self, batch):
        '''
        batch is the index and label, we need to get the rows of dataset
        '''
        #Getting the rows from the datasets
        # import pdb; pdb.set_trace()
        # filtered_batch = [item for item in batch if len(item[0]) < 500]
        pair_labels =  torch.tensor([label for left, right, label in batch])
        #define which data you want to extract from the dataset
        
        self.full_tokens = torch.full((len(batch), self.context_length), self.padding_idx, dtype=torch.int)#, device = device)
        self.token_type_ids =  torch.full((len(batch), self.context_length), self.padding_idx, dtype=torch.int)#, device = device)
        self.full_exp = torch.ones((len(batch), self.context_length), dtype=torch.int)#, device = device)
        self.norm_exp = torch.ones((len(batch), self.context_length),dtype=torch.float)#, device = device)
        self.gg_mtx = torch.full((len(batch), self.context_length, self.context_length), self.padding_idx, dtype=torch.float)#, device = device)
        #fill the data 

        self.one_site(batch)

       
        

        # Filtering: Keep samples with at least 5 non-padding elements
        valid_mask = (self.full_tokens != self.padding_idx)  # Create a mask for non-padding elements
        non_padding_counts = valid_mask.sum(dim=1)  # Count non-padding elements for each sample
        # Filter based on the count of non-padding elements
        non_zero_indices = non_padding_counts >= 5  # Keep only samples with at least 5 non-padding elements
        # Now filter gg_mtx to exclude 2D matrices that are all zeros
        gg_mtx_nonzero_mask = (self.gg_mtx.sum(dim=(1, 2)) != 0)  # Check if sum across the last two dimensions is not zero
        non_zero_indices = non_zero_indices & gg_mtx_nonzero_mask  # Combine both conditions



        # Check if there are valid samples
        if not non_zero_indices.any():
            print("Skipping batch: no valid samples with at least 5 non-padding elements.")
            return None 

        # Apply the filter to all relevant tensors
        self.full_tokens = self.full_tokens[non_zero_indices]
        self.gg_mtx = self.gg_mtx[non_zero_indices]
        self.token_type_ids = self.token_type_ids[non_zero_indices]
        self.full_exp = self.full_exp[non_zero_indices]
        self.norm_exp = self.norm_exp[non_zero_indices]
        pair_labels = pair_labels[non_zero_indices]



        # Pad sequences after filling the data
        attention_masks = (self.full_tokens != self.padding_idx).bool()

        # pair_labels =  torch.tensor(self.pair_labels)
        # import pdb; pdb.set_trace()
        return {
            'adjmtx': self.gg_mtx,
            'indices': self.full_tokens,
            'attention_mask': attention_masks,
            'normalized_exp': self.norm_exp,
            "Expression": self.full_exp,
            "pair_label": pair_labels,
            "token_type_ids": self.token_type_ids
        }
    def filldata(self, sample_index, batch, side):

        full_tokens = torch.tensor(batch["Full_Tokens"][0])
        gg_mtx = torch.tensor(batch["Gene_Gene_Matrix"][0])
        raw_exp = torch.tensor(batch["Expression"][0][0])

        #make sure all the data shorter than the context_length
        if len(full_tokens) > self.context_length:
            full_tokens = full_tokens[:self.context_length]
        if gg_mtx.shape[0] > self.context_length:
            gg_mtx = gg_mtx[:self.context_length, :self.context_length]
        if len(raw_exp) > self.context_length:
            raw_exp = raw_exp[:self.context_length]
        # import pdb; pdb.set_trace()
        self.full_exp[sample_index,1: raw_exp.size(0)+1] = raw_exp #1 for cls token
        # import pdb; pdb.set_trace()
        cls_site = 1
        sep_site = 1
        current_size = gg_mtx.shape[0]
        if side == 0: #for the left pair
            # import pdb; pdb.set_trace()
            prefix_length = cls_site + self.special_token_num
            self.pair1_length = cls_site + full_tokens.size(0)-(4-self.special_token_num)
            #adding the cls token first
            self.full_tokens[sample_index,0] = self.cls_token
            #adding the tokens 
            self.full_tokens[sample_index,cls_site:self.pair1_length] = full_tokens[4-self.special_token_num:] #add the special token for the left and right sequence
            #adding the sep in the middle of pair
            self.full_tokens[sample_index, self.pair1_length] = self.sep_token
            #adding the gene pair matrix
            self.last_mtx_length = gg_mtx.shape[0]
            self.gg_mtx[sample_index, prefix_length:(prefix_length+current_size), prefix_length:(prefix_length+current_size)] = gg_mtx

            self.token_type_ids[sample_index, :self.pair1_length+cls_site] = 1
        elif side == 1: #for the right pair
            # import pdb; pdb.set_trace()
            prefix_length = cls_site + self.special_token_num
            
            #adding the right cell
            start = (self.pair1_length + sep_site)
            seq_length = full_tokens.size(0)-(4-self.special_token_num)
            self.full_tokens[sample_index, start:(start + seq_length)] = full_tokens[4-self.special_token_num:]
            #add the sep to the right end
            self.full_tokens[sample_index, (start + seq_length): (start + seq_length + sep_site) ] = self.sep_token
            #add the gene pairs to the right cell
            self.gg_mtx[sample_index, (prefix_length + self.last_mtx_length + cls_site) : (prefix_length + self.last_mtx_length + cls_site + current_size), (prefix_length + self.last_mtx_length + cls_site) : (prefix_length + self.last_mtx_length + cls_site + current_size)] = gg_mtx

            self.token_type_ids[sample_index, start:(start + seq_length + sep_site)] = 2
        # import pdb; pdb.set_trace()

    def rebuild_adj(self, data, row, col, shape):
        # import pdb; pdb.set_trace()
        sparse_matrix = coo_matrix((data, (row, col)), shape=shape)
        adj = sparse_matrix.toarray()
        return adj


    def one_site(self, dataset_batch):

        for i, (left_dataset, right_dataset, label) in enumerate(dataset_batch):
            # import pdb; pdb.set_trace()
            self.filldata(i, left_dataset, side = 0)
           
            self.filldata(i, right_dataset, side = 1)
            # import pdb; pdb.set_trace()
            # self.pair_labels.append(label)
            




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


class Lora:
    def __init__(self, lora_config = None):
        self.lora_config = lora_config

    def wrapper(self, model = None):

        lora_config = LoraConfig(
                r=self.lora_config["r"], # Rank
                lora_alpha=self.lora_config["lora_alpha"],
                target_modules=self.lora_config["target_modules"],
                modules_to_save=self.lora_config["modules_to_save"],
                lora_dropout=0.05,
                bias="none"
            )

        peft_model = get_peft_model(model, lora_config)

        return peft_model
    @staticmethod
    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        import pdb; pdb.set_trace()
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        import pdb; pdb.set_trace()
        print(f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")




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

def extract_coo_components(example):
    """Extract COO components from example dictionary"""
    adj_mtx = np.array(example["Gene_Gene_Matrix"])
    row, col = np.nonzero(adj_mtx)
    data = adj_mtx[row, col]
    shape = adj_mtx.shape
    return {"row": row, "col": col, "data": data, "shape": shape}

def coo_to_binary_matrix(group_shape, data, row, col):
    # Create an empty binary matrix with the same shape as the sparse matrix
    binary_matrix = np.zeros(group_shape, dtype=int)

    # Fill in the ones at the indices where the sparse matrix has non-zero elements
    # import pdb; pdb.set_trace()
    if len(data) > 0:
        binary_matrix[row,col] = data

    return binary_matrix

def read_h5(file_object, cell_id):
    # Get the data
    # import pdb; pdb.set_trace()
    grp = file_object[cell_id]
    grp_shape = grp.attrs["shape"]
    row = grp['row'][:]
    col = grp['col'][:]
    data = grp['data'][:]
    sparse_matrix = coo_matrix((data, (row, col)), shape=grp_shape)
    # int_matrix = coo_to_binary_matrix(grp_shape, data, row, col)
    return sparse_matrix



def uniform_quantile_global(values: torch.Tensor):
    # Flatten the tensor to treat it as a single list of values
    flattened_values = values.flatten()
    

    # Compute quantiles for each unique value
    bins = torch.quantile(values ,torch.linspace(0, 1, 100))
    # value_to_quantile = dict(zip(unique_vals, quantiles))
    left_digits = np.digitize(values.numpy(), bins)
    # Map each value in the original flattened tensor to its corresponding quantile
    #for the diagnal values add a extremly small value
    # Define an extremely small value
    epsilon = 1e-10
    # Create an identity matrix of the same size
    identity_matrix = torch.eye(left_digits.shape[0])

    # Convert flattened_values to a NumPy array for mapping
    p = 1 - torch.eye(left_digits.shape[0])
    # import pdb; pdb.set_trace()
    left_digits =  (torch.from_numpy(left_digits) * p / left_digits.max()) + epsilon * identity_matrix #the diagnal values are distinguishable to 0 in order to easy for masking

    # import pdb; pdb.set_trace()
    return left_digits




def _digitize(x: np.ndarray, bins: np.ndarray, side="one") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    # import pdb; pdb.set_trace()
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    # TODO: use torch.quantile and torch.bucketize
    # import pdb; pdb.set_trace()
    if row.max() == 0:
        print(
            "The input data contains row of zeros. Please make sure this is expected."
        )
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        # import pdb; pdb.set_trace()
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)/_digitize(row, bins).max()

    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)



def complete_masking(batch, p, n_tokens, cls_token, mask_token, sep_token, pad_token):
    '''
    This is used to mask the tokens for the mask language model head.
    '''
    # padding_token = 0
    # cls_token = 1
    # mask_token = 2
    
    nomasked_token = 999
    indices = batch['indices']

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p) # mask indices with probability p, represent mask as 0 (15%), 85% as 1
    
    
    masked_indices = indices * mask # masked_indices, mute masked sites
    #embedding the mask token
    masked_indices = torch.where(masked_indices == 0, mask_token, masked_indices)

    masked_indices = torch.where(indices != pad_token, masked_indices, indices) # we just mask non-padding indices
    mask = torch.where(indices == pad_token, nomasked_token, mask) # the mask sequence with the padding tokens
    # so we make the mask of all PAD tokens to be 0 so that it's not taken into account in the loss computation

    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_indices = torch.where(indices != cls_token, masked_indices, indices) # same with CLS, no CLS token can be masked
    mask = torch.where(indices == cls_token, nomasked_token, mask) # we change the mask so that it doesn't mask any CLS token
    masked_indices = torch.where(indices != sep_token, masked_indices, indices) # same with SEP, no SEP token can be masked
    mask = torch.where(indices == sep_token, nomasked_token, mask) # we change the mask so that it doesn't mask any SEP token
    # import pdb; pdb.set_trace()
    #setting the 0 to the mask tokens
    mask = torch.where(mask == 0, mask_token, mask)

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
    masked_indices = torch.where(indices != pad_token, masked_indices, indices) # don't mask the padding sites
    batch['masked_indices'] = masked_indices
    batch['mask'] = mask

    return batch


def complete_edge_masking(dis_mtx, p):
    '''
    This can be used to mask the edges of the distance of gene pairs
    '''
    padding_token = 0
    cls_token = 1
    mask_token = 2
    # import pdb; pdb.set_trace()
    # Fetch the distance matrix - ground truth

    # Create a mask for the distance matrix with probability p
    mask = 1 - torch.bernoulli(torch.ones_like(dis_mtx) * p).int() #set 15% edges for masking as 0

    masked_dis_mtx = dis_mtx * mask #set mask tag to the whole matrix, including cls and padding sites
    # Apply the mask to the distances
    masked_dis_mtx = torch.where(masked_dis_mtx == 0, torch.tensor(mask_token), dis_mtx)  # step 1: Represent masked distances with mask token
    masked_dis_mtx = torch.where(dis_mtx != padding_token, masked_dis_mtx, dis_mtx) # step 2: we just mask non-padding indices, 
    mask = torch.where(dis_mtx == padding_token, torch.tensor(padding_token), mask) # step 3: the mask sequence with the padding tokens
    # Handling padding and CLS tokens
    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_dis_mtx = torch.where(dis_mtx != cls_token, masked_dis_mtx, dis_mtx) # step 4: same with CLS, no CLS token can be masked
    mask = torch.where(dis_mtx == cls_token, torch.tensor(padding_token), mask) # we change the mask so that it doesn't mask any CLS token
    #setting the 0 to the mask tokens
    mask = torch.where(mask == 0, mask_token, mask)

    mask = torch.where(dis_mtx == padding_token, torch.tensor(padding_token), mask)


    # 80% of masked indices are masked
    # 10% of masked indices are a random token
    # 10% of masked indices are the real token
    # import pdb; pdb.set_trace()
    #10 means the start token of the real gene names
    # random_dis = torch.rand(size=masked_dis_mtx.shape, device=masked_dis_mtx.device) #because the distance has been normalized to 0-1, the random distance should be 0-1
    # random_dis = random_dis * torch.bernoulli(torch.ones_like(random_dis) * 0.1).type(torch.int64) #apply 10%
    # random_dis = torch.where(random_dis == 0, mask_token, random_dis)
    # masked_dis_mtx = torch.where(masked_dis_mtx == mask_token, random_dis, masked_dis_mtx) # put random tokens just in the previously masked tokens

    # same_dis_mtx = dis_mtx.clone()
    # same_dis = same_dis_mtx * torch.bernoulli(torch.ones_like(same_dis_mtx) * 0.1).type(torch.int64)
    # same_dis = torch.where(same_dis == 0, mask_token, same_dis_mtx)
    # masked_dis_mtx = torch.where(masked_dis_mtx == mask_token, same_dis, masked_dis_mtx) # put same tokens just in the previously masked tokens
    # masked_dis_mtx = torch.where(masked_dis_mtx != padding_token, masked_dis_mtx, dis_mtx) # don't mask the padding sites



    # batch['masked_dis_mtx'] = masked_dis_mtx
    # batch['mask_2d'] = mask

    return masked_dis_mtx, mask

def categorical_2d_masking(batch, p = 0.5):
    '''
    The input of this fuction should be a binary co-occurrency matrix
    This can be used to mask the edges of the distance of gene pairs
    '''
    import pdb; pdb.set_trace()
    padding_token = 0
    cls_token = 1
    mask_token = 2
    # import pdb; pdb.set_trace()
    # Fetch the distance matrix - ground truth, which is a binaray matrix
    co_mtx = batch['Gene_Gene_Matrix']
    nco_mtx = 1 - batch['Gene_Gene_Matrix']

    masked_co_mtx, co_mask = complete_edge_masking(co_mtx, p)
    masked_nco_mtx, nco_mask = complete_edge_masking(nco_mtx, p)

    masked_adj_mtx = co_mtx * masked_co_mtx + nco_mtx * masked_nco_mtx
    mask = co_mtx * co_mask + nco_mtx * nco_mask


    batch['masked_adj_mtx'] = masked_adj_mtx
    batch['mask_2d'] = mask

    return batch



def stat_test(all_infos, vocab):
    pair_diff = {}
    pair_rank = {}
    pair_cellpair = {}
    for cell_pair in all_infos.keys():
        token_pairs = all_infos[cell_pair]["combination_tokens"]
        diffs = all_infos[cell_pair]["diff"]
        rank = all_infos[cell_pair]["combination_indexs"]
        for i, (token_pair, diff) in enumerate(zip(token_pairs, diffs)):
            pairrank = rank[i]
            rev_pair = tuple([token_pair[1], token_pair[0]])
            if rev_pair in pair_diff.keys():
                pair_diff.setdefault(rev_pair,[]).append(diff)
                pair_rank.setdefault(rev_pair,[]).append(pairrank)
                pair_cellpair.setdefault(rev_pair,[]).append(cell_pair)
            else:
                pair_diff.setdefault(tuple(token_pair),[]).append(diff)
                pair_rank.setdefault(tuple(token_pair),[]).append(pairrank)
                pair_cellpair.setdefault(tuple(token_pair),[]).append(cell_pair)
    #calculating the mean for each pair
    pairs = []
    gene1 = []
    gene2 = []
    top20_cp = []
    top20_diff = []
    mean_diffs = []
    mean_ranks = []
    support_num = []
    t_stats = []
    p_values = []
    threshold = 0
    for pair,diffs in tqdm(pair_diff.items()):
        
        ranks = pair_rank[pair]
        mean_diff = np.mean(diffs)
        # Get the indices of values in the top 20% of diffs
        percentile_20 = np.percentile(diffs, 20)
        
        # Find indices of all values that are <= the 20th percentile
        top_20_percent_indices = np.where(diffs <= percentile_20)[0]
        
        # Now you have both indices and values
        top_20_percent_values = np.array(diffs)[top_20_percent_indices]
        #sort the values
        sorted_indices = np.argsort(top_20_percent_values)
        sorted_diff = top_20_percent_values[sorted_indices]
        cell_pairs = np.array(pair_cellpair[pair])[top_20_percent_indices]
        sorted_cellpairs = cell_pairs[sorted_indices]
        # import pdb; pdb.set_trace()
        top20_cp.append(sorted_cellpairs)
        top20_diff.append(sorted_diff)
        
        t_stat, p_value = stats.ttest_1samp(diffs, popmean=threshold, alternative='less')
        mean_rank = np.mean(ranks)
        #transfer the pair to gene symbol
        p_values.append(p_value)
        t_stats.append(t_stat)
        support_num.append(len(diffs))
        gene_pair = (vocab[pair[0]], vocab[pair[1]])
        gene1.append(vocab[pair[0]])
        gene2.append(vocab[pair[1]])
        pairs.append(gene_pair)
        mean_diffs.append(mean_diff)
        mean_ranks.append(mean_rank)

    

    
    pair_df = pd.DataFrame({"gene_pair": pairs, 
                            "gene1": gene1,
                            "gene2": gene2,
                            "Deduction": mean_diffs,
                            "rank_mean": mean_ranks, 
                            "stat": t_stats,
                            "P_value": p_values,
                            "support_num": support_num,
                           "top20_cellpairs": top20_cp,
                           "top20_diff": top20_diff})
    return pair_df
def ovlp_database(pair_df, database_path1, database_path2):
    '''
    Finding the overlapping between the pair-wise genes and the ligand-receptor gene pairs
    '''
    #loading the table of the database from cellchat
    database1 = pd.read_csv(database_path1)
    database2 = pd.read_csv(database_path2)
    # Create a function to check for overlaps
    def check_overlap_for_cellchat(interaction):
        for i,top_pair in enumerate(pair_df["gene_pair"]):
            if (top_pair[0] in interaction.split("_")) and (top_pair[1] in interaction.split("_")):
                if top_pair[0] != top_pair[1]: #filter
                    return int(i)  # Return the top pair if found
        return None  # Return None if no pairs match
    def check_overlap_for_cellnest(row):
        lr_pair = [row["Ligand"], row["Receptor"]]
        for i,top_pair in enumerate(pair_df["gene_pair"]):
            if (top_pair[0] in lr_pair) and (top_pair[1] in lr_pair):
                if top_pair[0] != top_pair[1]: #filter
                    return int(i)  # Return the top pair if found
        return None  # Return None if no pairs match

    # Apply the function to the column to extract overlapping pairs
    ovlp_lr_idx1 = database1["interaction.interaction_name"].apply(check_overlap_for_cellchat)
    ovlp_lr_idx2 = database2[["Ligand", "Receptor"]].apply(check_overlap_for_cellnest, axis=1)
    ovlp_lr_idx = list(ovlp_lr_idx1) + list(ovlp_lr_idx2)
    ovlp_lr_idx_series = pd.Series(ovlp_lr_idx)
    ovlp_lr_idx = np.unique(ovlp_lr_idx_series.dropna().tolist())
    # FDR p_value correction
    
    pair_df["ligand_receptor"] = False
    pair_df.loc[ovlp_lr_idx, "ligand_receptor"] = True

    # Transform the "Deduction" column into "10log(1-D)" and keep sign
    pair_df['10log(1-D)'] = -10*np.log(1-np.abs(pair_df['Deduction'])) / np.log(2)  # Apply -log transformation
    pair_df['symbol'] = [1 if i >0 else -1 for i in pair_df['Deduction']]
    pair_df['10log(1-D)'] = pair_df['10log(1-D)']*pair_df['symbol']
    return pair_df
def filter_df(pair_df):
    '''
    Filter out the significant gene pairs
    '''
    filtered_df = pair_df[(pair_df["P_value"] < 0.05) & (pair_df["support_num"] > 50) & (pair_df["gene1"] != "<SEP>") & (pair_df["gene2"] != "<SEP>") & (pair_df["symbol"] == -1)]
    reject, pvals_corrected, _, _ = multipletests(filtered_df["P_value"], alpha=0.05, method='fdr_bh')
    filtered_df["adj_P_value"] = pvals_corrected
    filtered_df = filtered_df[filtered_df["adj_P_value"] < 0.05]
    ranked_df = filtered_df.sort_values(by="Deduction", ascending=True)
    
    # Optionally, reset the index if you want a cleaner DataFrame
    return ranked_df



def filter_state_dict_by_shape(src_sd, tgt_model):
    tgt_sd = tgt_model.state_dict()
    new_sd = OrderedDict()

    loaded, skipped = [], []

    for k, v in src_sd.items():
        if k in tgt_sd and tgt_sd[k].shape == v.shape:
            new_sd[k] = v
            loaded.append(k)
        else:
            skipped.append(k)

    return new_sd, loaded, skipped

def load_original_into_flash(
    flash_model,
    original_ckpt_path,
    device="cpu",
    strict=False,
):
    ckpt = torch.load(original_ckpt_path, map_location=device)
    src_sd = ckpt.get("state_dict", ckpt)

    # 1️⃣ Remove known incompatible modules explicitly
    blacklist_prefixes = (
        "encoder.emb_proj",
        "encoder.bpp_feature_proj",
        "encoder.bpp_convnet",
        "classifier_head",        # different output dim
        "embeddings",             # vocab size mismatch
        "spatialembeds.emb",      # vocab size mismatch
        "adjprojector",           # architecture changed
    )

    filtered_src = {
        k: v for k, v in src_sd.items()
        if not k.startswith(blacklist_prefixes)
    }

    # 2️⃣ Match by key + shape
    matched_sd, loaded, skipped = filter_state_dict_by_shape(
        filtered_src, flash_model
    )

    # 3️⃣ Load
    flash_model.load_state_dict(matched_sd, strict=False)

    print(f"✅ Loaded {len(loaded)} tensors")
    print(f"⚠️ Skipped {len(skipped)} tensors")

    if strict:
        print("\nSkipped keys:")
        for k in skipped:
            print("  ", k)

    return flash_model

def load_partial_embeddings(src_emb, tgt_emb):
    n = min(src_emb.weight.shape[0], tgt_emb.weight.shape[0])
    tgt_emb.weight.data[:n].copy_(src_emb.weight.data[:n])