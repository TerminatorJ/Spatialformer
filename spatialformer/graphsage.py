import os
import logging
import json
import pickle
from pathlib import Path
import torch
from torch_geometric.data import Data
from .GraphSAGE import *

# Ensure these functions and classes are defined or imported
# from your_module import load_and_preprocess_data, build_graph_for_sample, MyTrainer

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