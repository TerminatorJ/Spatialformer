import os
import pandas as pd 
import numpy as np
from communities.algorithms import louvain_method
import networkx as nx
import torch
from tqdm import tqdm
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from scipy.sparse import coo_matrix
from typing import Dict, Union
from scipy.spatial import KDTree 
import matplotlib.pyplot as plt
import pickle
import random
from torch.utils.data import Dataset, IterableDataset
from itertools import combinations
from sklearn.model_selection import train_test_split
from datasets import load_from_disk, concatenate_datasets



def get_adj(sample_dataset, radius = 10, plot = False):
    r_c = np.array(list(zip(sample_dataset["centroid_x"],sample_dataset["centroid_y"])))
    cell_ids = sample_dataset["Cell_Ids"]
    # r_c = np.array(df[['centroid_x', 'centroid_y']])
    G = nx.Graph()
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
class DynamicHuggingFaceDatasetEval(IterableDataset):
    def __init__(self, datapath):
        '''
        huggingface dataset
        '''
        self.datapath  = datapath
    def load_dataset(self, path):
        dataset = load_from_disk(path)
        return dataset

    def __iter__(self):

        try:
            dataset = self.load_dataset(self.datapath)
        except FileNotFoundError:
            print(f"{self.datapath} is not a valid dataset")

        iter_dataset = dataset.to_iterable_dataset(num_shards=64).shuffle(buffer_size=10_000, seed=42)

        yield from iter_dataset 

class DynamicHuggingFaceDataset(IterableDataset):
    def __init__(self, datapath, split):
        '''
        datapath: cache path
        split: which split to access ('train' or 'test')
        shuffle: whether to shuffle the dataset
        '''
        all_files = os.listdir(datapath)  # Corrected method name
        self.datasets_paths = [os.path.join(datapath, file) for file in all_files if file.endswith("pair")]
        self.split = split
    def load_dataset(self, path):
        dataset = load_from_disk(path)
        return dataset

    def __iter__(self):
        # Iterate through datasets
        for datasets_path in tqdm(self.datasets_paths):
        # for i in tqdm(range(0, len(self.datasets_paths))):
            # if i + 1 < len(self.datasets_paths):
            #     pairs_path = [self.datasets_paths[i], self.datasets_paths[i + 1]]
            # else:
            #     pairs_path = [self.datasets_paths[i]]

            # pair_dataset = []
            # for path in pairs_path:
            try:
                dataset = self.load_dataset(datasets_path)
            except FileNotFoundError:
                print(f"{datasets_path} is not a valid dataset")
                continue
                # dataset = dataset[:10]###delete
                # Split the dataset into train/test
            # dataset = dataset.select(range(100))  ##delete
            split_dataset = dataset.train_test_split(test_size=0.001, seed=42)
            # split_dataset = dataset.train_test_split(test_size=0.001)
                # import pdb; pdb.set_trace()
                ##   
            print(split_dataset)
                # Convert to IterableDataset
            if self.split == "train":
                iter_dataset = split_dataset["train"].to_iterable_dataset(num_shards=64)
                    # pair_dataset.append(iter_dataset)
                iter_dataset = iter_dataset.shuffle(buffer_size=10_000, seed=42)###
                # iter_dataset = split_dataset_by_node(iter_dataset, world_size=64, rank=0)
            elif self.split == "test":
                iter_dataset = split_dataset["test"].to_iterable_dataset(num_shards=8)##8
                # iter_dataset = split_dataset_by_node(iter_dataset, world_size=64, rank=0)
            yield from iter_dataset 
            # pair_dataset.append(iter_dataset)
        # if pair_dataset:
        #     if self.split == "train":
        #         iter_pair_dataset = concatenate_datasets(pair_dataset)
        #         iter_pair_dataset = iter_pair_dataset.shuffle(buffer_size=10_000, seed=42)
        #         yield from iter_pair_dataset 
        #     elif self.split == "test":
        #         iter_pair_dataset = concatenate_datasets(pair_dataset)
        #         yield from iter_pair_dataset
        # else:
        #     print("No valid datasets found in the current pair.")
 


class GetPairs(Dataset):
    def __init__(self, adjacency_matrix:coo_matrix):
        self.adj_matrix = adjacency_matrix
        self.num_nodes = adjacency_matrix.shape[0]
        # Create positive pairs (edges from adjacency matrix)
        self.positive_pairs = np.column_stack((adjacency_matrix.row, adjacency_matrix.col))

        # Get the number of positive pairs
        num_positive = len(self.positive_pairs)

        # Create negative pairs (we will sample after ensuring all nodes are covered)
        self.negative_pairs = self.create_negative_pairs(num_positive)
        #make sure all nodes included
        # import pdb; pdb.set_trace()
        positive_covered_nodes = {node for pair in self.positive_pairs for node in pair}
        negative_covered_nodes = {node for pair in self.negative_pairs for node in pair}
        # import pdb; pdb.set_trace()
        all_node = positive_covered_nodes.union(negative_covered_nodes)
        # import pdb; pdb.set_trace()
        assert self.num_nodes == len(all_node), "ERROR: There are some nodes won't be sampled"
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
    

    indices = batch['indices']
    # import pdb; pdb.set_trace()
    # indices = torch.where(indices == 0, torch.tensor(padding_token), indices) # 0 is originally the padding token, we change it to 1
    # batch['indices'] = indices

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p) # mask indices with probability p, represent mask as 0 (15%), 85% as 1, without padding tokens
    
    # mask = torch.where(mask == 1, mask_token, mask)
    
    masked_indices = indices * mask # masked_indices, mute masked sites
    #embedding the mask token
    masked_indices = torch.where(masked_indices == 0, mask_token, masked_indices)

    masked_indices = torch.where(indices != pad_token, masked_indices, indices) # we just mask non-padding indices
    mask = torch.where(indices == pad_token, torch.tensor(pad_token), mask) # the mask sequence with the padding tokens
    # so we make the mask of all PAD tokens to be 1 so that it's not taken into account in the loss computation
    # import pdb; pdb.set_trace()
    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_indices = torch.where(indices != cls_token, masked_indices, indices) # same with CLS, no CLS token can be masked
    mask = torch.where(indices == cls_token, torch.tensor(pad_token), mask) # we change the mask so that it doesn't mask any CLS token
    masked_indices = torch.where(indices != sep_token, masked_indices, indices) # same with SEP, no SEP token can be masked
    mask = torch.where(indices == sep_token, torch.tensor(pad_token), mask) # we change the mask so that it doesn't mask any SEP token

    #setting the 0 to the mask tokens
    mask = torch.where(mask == 0, mask_token, mask)

    mask = torch.where(indices == pad_token, torch.tensor(pad_token), mask)

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







