import os
import logging
import json
import pickle
from pathlib import Path
from torch_geometric.data import Data
from .GraphSAGE import *
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
# Ensure these functions and classes are defined or imported
# from your_module import load_and_preprocess_data, build_graph_for_sample, MyTrainer


class GraphDataset(Dataset):
    def __init__(self, filename):
        # Load the data list from a Parquet file
        # import pdb; pdb.set_trace()
        self.data_list = pickle.load(open(f"{filename}/graph_list.pkl", "rb"))  # Correct file path for parquet
        self.all_patterns = pickle.load(open(f"{filename}/patterns.pkl", "rb"))
        # import pdb; pdb.set_trace()
        # Store the number of samples
        self.num_samples = len(self.data_list)  # Use node_features for the number of samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        x = self.data_list[idx].x.type(torch.float32)
        edge_index = self.data_list[idx]['edge_index'].type(torch.long).contiguous()  # Ensure it is of shape [2, num_edges]
        patterns = self.all_patterns[idx]
        return Data(x=x, edge_index=edge_index), patterns



class GraphSAGEModule:
    def __init__(self):
        self.subgraphs = None
        self.config = {}
        self.directory_path = None
        self.model = None
        self.trainer = None
        self.index_to_gene = None
    def get_subgraph(self, 
                     transcript_file: str = None,
                     postfix: str = None,
                     threshold: int = 10):
        """
        Employing the GraphSAGE algorithm to get the transcript embeddings.

        This method processes the provided transcript file to generate a 
        graph representation using the GraphSAGE algorithm.

        Args:
            transcript_file (str): The path to the transcript file 
                to be processed.
            postfix (str, optional): A postfix for differentiating files. Defaults to None.
            threshold (int, optional): The threshold for calculating the transcript 
                neighbors as "d" in the paper. Defaults to 10.

        Returns:
            None

        Raises:
            AssertionError: If transcript_file is not provided.
        """

        assert transcript_file, "Please input the transcript file!"

        logging.info("Running GraphSAGE to get the spatial embeddings")
        logging.info("The subgraph will be saved in the same path as the transcript file")

        filename = Path(transcript_file).name
        self.directory_path = Path(transcript_file).parent
        subgraph_name = f"subgraph_data_{postfix}.pt" if postfix else "subgraph_data.pt"
        subgraph_path = self.directory_path / subgraph_name
        self.index_to_gene, gene_to_index = index2gene(transcript_file, qv = False)
        if subgraph_path.exists():
            logging.info(f"The subgraph has been generated, please check {subgraph_path}")
            self.subgraphs = torch.load(subgraph_path)
        else:
            dataset = load_and_preprocess_data(transcript_file)
            self.subgraphs = build_graph_for_sample(dataset, sample_id=postfix, threshold=threshold)
            torch.save(self.subgraphs, subgraph_path)
            logging.info("The subgraph has been generated, please run the command again!")

    def train_model(self,
                    subgraphs=None,
                    feature_num = None,
                    hidden_dim = None,
                    output_dim = None,
                    total_step = None,
                    strategy =  None,
                    lr = None,
                    batch_size = None):
        """
        Train the model on the available subgraphs.

        Args:
            config_path (str): The path to the configuration file.
            subgraphs: A pre-defined list of subgraphs to be used. Defaults to None.
        
        Returns:
            model: The trained model.
        
        Raises:
            AssertionError: If `get_subgraph` hasn't been run.
        """

        assert self.subgraphs, "You need to run 'get_subgraph' first"

        if subgraphs:
            joint_x = torch.cat([subgraph.x for subgraph in subgraphs], dim=0)
            offset = 0
            edge_lists = []
            for subgraph in subgraphs:
                edge_lists.append(subgraph.edge_index + offset)
                offset += subgraph.x.size(0)
            joint_edge_index = torch.cat(edge_lists, dim=1)
            self.subgraphs = Data(x=joint_x, edge_index=joint_edge_index)

        logging.info("Building the dataloader")
        print("Training the model")

        
        self.config["feature_num"] = feature_num
        self.config["hidden_dim"] = hidden_dim
        self.config["output_dim"] = output_dim
        self.config["total_step"] = total_step
        self.config["strategy"] = strategy
        self.config["lr"] = lr
        self.config["batch_size"] = batch_size
        
        trainer = MyTrainer(self.config, self.subgraphs)
        self.trainer = trainer
        trainer.train()
        model = trainer.plmodel
        self.model = model
        return model

    def get_embedding(self,
                      checkpoint: str = None,
                      token_path: str = None):
        """
        Obtain the embeddings from the trained model.

        Args:
            checkpoint (str): Path to the model checkpoint.
            token_path (str): Path to the token data.

        Returns:
            None
        """
        import pdb; pdb.set_trace()
        # Ensure index_to_gene and config['output_dim'] are defined or passed in as arguments
        embeddings = self.trainer.get_embedding(self.index_to_gene, checkpoint, token_path, self.config["feature_num"], self.config["output_dim"])
        pickle.dump(embeddings, open(self.directory_path / "gene_embeddings_GraphSAGE.pkl", "wb"))
        return embeddings
    
    def load_pretrained_model(self,
                            vocab_path,
                            hidden_dim,
                            output_dim,
                            batch_size,
                            checkpoint,
                            device
                            ):
        vocab_list = index2gene(vocab_path)
        self.config["feature_num"] = len(vocab_list)
        self.config["hidden_dim"] = hidden_dim
        self.config["output_dim"] = output_dim
        self.config["total_step"] = 10086
        self.config["strategy"] = "ddp"
        self.config["lr"] = 0.001
        self.config["batch_size"] = batch_size
        
        trainer = MyTrainer(self.config, len(vocab_list), None)
        self.trainer = trainer
        plmodel = trainer.plmodel
        ckp = torch.load(checkpoint, map_location=torch.device(device))
        params = ckp["state_dict"]
        plmodel.load_state_dict(params)
        return plmodel
    # def save_to_parquet(self, data_list, filename):
    #     edge_lists = [d.edge_index.t().numpy() for d in data_list]
    #     node_features = [d.x.numpy() for d in data_list]
        
    #     table = pa.Table.from_arrays(
    #         [pa.array(edge_lists), pa.array(node_features)],
    #         names=['edge_list', 'node_features']
    #     )
    #     pq.write_table(table, filename)
    def save_to_parquet(self, data_list, filename):
        #only save the graph with edge
        # import pdb; pdb.set_trace()
        data_list = [data for data in data_list if len(data.edge_index) > 0]
        pickle.dump(data_list, open(f"{filename}/graph_list.pkl", "wb"))
        # edge_sources = []  # This will hold source nodes
        # edge_targets = []  # This will hold target nodes
        # node_features = []

        # for data in data_list:
        #     edge_index = data.edge_index.numpy()  # Assuming edge_index is a 2D tensor [2, num_edges]
        #     if len(edge_index) > 0:
        #         # Collect sources and targets
        #         edge_sources.append(edge_index[0])  # First row: source
        #         edge_targets.append(edge_index[1])  # Second row: target

        #         node_features.append(data.x.numpy())  # Load node features

        # # Create PyArrow Table with two arrays for edge_source and edge_target
        # edge_sources_pa = pa.array([item for sublist in edge_sources for item in sublist])  # Flatten to 1D
        # edge_targets_pa = pa.array([item for sublist in edge_targets for item in sublist])  # Flatten to 1D
        # import pdb; pdb.set_trace()
        # node_features_pa = pa.array([node_feature for node_feature in node_features])

        # import pdb; pdb.set_trace()
        # # Creating the table with separate source and target lists
        # table = pa.Table.from_arrays(
        #     [edge_sources_pa, edge_targets_pa, node_features_pa],
        #     names=['edge_source', 'edge_target', 'node_features']
        # )

        # pq.write_table(table, f"{filename}/graph_data.arrow")

    def build_sc_graph(self, vocab_list, single_gene_df, batch_size, threshold):
        # import pdb; pdb.set_trace()
        try:
            r_c = np.array(single_gene_df[['x_location', 'y_location', 'z_location']])
        except:
            r_c = np.array(single_gene_df[['x', 'y', 'z']])
        gene_labels = single_gene_df['feature_name'].values 

        # Convert vocab dictionary keys to a sorted list
        # Initialize the OneHotEncoder with the specific categories
        encoder = OneHotEncoder(categories=[vocab_list])
        # Transform the gene_labels into one-hot encoded format
        one_hot_labels = encoder.fit_transform(gene_labels.reshape(-1, 1))

        # import pdb; pdb.set_trace()
        kdtree = KDTree(r_c)
        G = nx.Graph()
        # import pdb; pdb.set_trace()
        for i in range(len(r_c)):
            G.add_node(i, feature=one_hot_labels[i])
        # import pdb; pdb.set_trace()
        num_nodes = len(r_c)
    
        # print("building graph in batch")
        for start_idx in range(0, num_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_nodes)
            batch_r_c = r_c[start_idx:end_idx]
            edges_to_add = []
            for i, x in enumerate(batch_r_c, start=start_idx):
                # import pdb; pdb.set_trace()
                neighbors_idx = kdtree.query_ball_point(x, threshold)
                for j in neighbors_idx:
                    if i < j:
                        edges_to_add.append((i, j))
            G.add_edges_from(edges_to_add)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()   
        # import pdb; pdb.set_trace()
        x = torch.tensor(one_hot_labels.toarray(), dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        return data
    def load_parquet(self, filename, batch_size, split):
        # Load the Parquet file
        graph_dataset = GraphDataset(filename)
        if split:
            # Set the random seed for reproducibility
            torch.manual_seed(42)

            # Define the sizes of each split
            total_size = len(graph_dataset)
            train_size = int(0.7 * total_size)  # 70% for training
            val_size = int(0.15 * total_size)    # 15% for validation
            test_size = total_size - train_size - val_size  # The rest for testing

            # Split the dataset
            train_dataset, val_dataset, test_dataset = random_split(graph_dataset, [train_size, val_size, test_size])

            # Create DataLoaders for each dataset split
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            return (train_dataloader, val_dataloader, test_dataloader)
        else:
            graph_dataloader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True)
        return graph_dataloader
    def build_graph(self,
                    data_path,
                    vocab_path,
                    batch_size,
                    graph_path,
                    threshold,
                    split
                    ):
        if not os.path.exists(f"{graph_path}/graph_list.pkl"):
            print(f"loading {data_path}")
            try:
                dataset = pd.read_csv(data_path)
            except:
                dataset = pd.read_csv(data_path + ".gz")
            print("after loading the data")
            if "qv" in dataset.columns:
                dataset = dataset[dataset['qv'] >= 20]
            dataset = dataset[~(dataset['feature_name'].str.startswith('Neg') | dataset['feature_name'].str.startswith('BLANK') | dataset['feature_name'].str.startswith('Unassigned'))]
            vocab_list = index2gene(vocab_path)
            # dataset = dataset[:100]
            # import pdb; pdb.set_trace()
            grouped = dataset.groupby(['feature_name', 'cell_id'])
            all_graphs = []
            all_patterns = []
            # Iterate over each unique cell type
            for (feature_name, cell_id), group_df in tqdm(grouped):
                # import pdb; pdb.set_trace()
                data = self.build_sc_graph(vocab_list, group_df, batch_size, threshold)
                all_graphs.append(data)
                patterns = list(group_df["pattern"])
                all_patterns.extend(patterns)


            print("saving the graph")
            self.save_to_parquet(all_graphs, graph_path)

            pickle.dump(all_patterns, open(f"{graph_path}/patterns.pkl", "wb"))
            graph_dataloader = self.load_parquet(graph_path, batch_size, split)
        else:
            #loading the save parquet file
            print("loading the saved data")
            # all_graphs = self.load_parquet(graph_path)
            graph_dataloader = self.load_parquet(graph_path, batch_size, split)

        return graph_dataloader
    def train_model(self, 
                    model,
                    lr,
                    strategy,
                    output_dim,
                    output_dir,
                    train_dataloader,
                    val_dataloader):
        self.trainer = PatternTrainer(model, lr, strategy, output_dim, output_dir, train_dataloader, val_dataloader, None)
        print("training the model")
        self.trainer.train()
    def test_model(self,
                   test_dataloader,
                   ckp_path
                   ):
        self.trainer = PatternTrainer(test_dataset=test_dataloader)
        self.trainer.evaluation(ckp_path)

    # def train_pattern_model(self,dataloader):
    #     PatternTrainer()




