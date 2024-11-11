import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
from scipy.sparse import coo_matrix
sys.path.append("/scratch/project_465001027/Spatialformer/utils")

class BalancedPairDataset(Dataset):
    def __init__(self, adjacency_matrix:coo_matrix):
        self.adj_matrix = adjacency_matrix
        self.num_nodes = adjacency_matrix.shape[0]
        # Create positive pairs (edges from adjacency matrix)
        self.positive_pairs = np.column_stack((adjacency_matrix.row, adjacency_matrix.col))
        # self.positive_pairs = np.array(np.where(self.adj_matrix == 1)).T
        # import pdb; pdb.set_trace()
        # Get the number of positive pairs
        num_positive = len(self.positive_pairs)

        # Create negative pairs (we will sample after ensuring all nodes are covered)
        self.negative_pairs = self.create_negative_pairs(num_positive)
        #make sure all nodes included
        covered_nodes = {node for pair in self.positive_pairs for node in pair}
        covered_nodes = {node for pair in self.negative_pairs for node in pair}
        all_node = covered_nodes.union(covered_nodes)
        assert self.num_nodes == len(all_node), "ERROR: There are some nodes won't be sampled"
        print(f"The total number of pairs: \npositive pair:{len(self.positive_pairs)}\nnegative pair:{len(self.negative_pairs)}")
        # Combine pairs for the dataset
        self.pairs = np.concatenate([self.positive_pairs, self.negative_pairs])
        self.labels = np.concatenate([np.ones(len(self.positive_pairs)), np.zeros(len(self.negative_pairs))])

    def create_negative_pairs(self, num_positive):
        """Generate negative pairs (non-edges) and ensure they are balanced with positive pairs."""
        negative_pairs = []
        possible_pairs = set((i, j) for i in range(self.num_nodes) for j in range(self.num_nodes) if i != j)
        covered_nodes = {node for pair in self.positive_pairs for node in pair}
        # Create set of existing positive pairs for quick lookup
        positive_set = set(map(tuple, self.positive_pairs))
        # import pdb; pdb.set_trace()
        # Create negative pairs by sampling non-existing edges
        for pair in possible_pairs:
            for node in pair:
                #first add the independent node
                if node not in covered_nodes and len(negative_pairs) < num_positive and pair not in negative_pairs:
                    negative_pairs.append(pair)


        # If we do not have enough negative pairs, warning or adjustment can be added here
        if len(negative_pairs) < num_positive:
            for pair in possible_pairs:
                if pair not in negative_pairs and pair not in positive_set and len(negative_pairs) < num_positive:
                    negative_pairs.append(pair)
        #if still not enough
        if len(negative_pairs) < num_positive:
            gap_num = num_positive - len(negative_pairs)
            negative_pairs += negative_pairs[:gap_num]
        print(f"positive pair: {num_positive}, negative pair: {len(negative_pairs)}")
        assert num_positive == len(negative_pairs), "The positive and negative pairs should be balanced, please check your codes!!!"

        return np.array(negative_pairs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


#13 mins for filtering the sample
# data = np.array([1, 1, 1])
# row = np.array([0, 1, 2])
# col = np.array([1, 2, 0])
# coo_matrix_example = coo_matrix((data, (row, col)), shape=(3, 3))
# dataset = BalancedPairDataset(coo_matrix_example)
# import pdb; pdb.set_trace()
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# for batch in dataloader:
#     pairs, labels = batch
#     print("Pairs:", pairs)
#     print("Labels:", labels)
