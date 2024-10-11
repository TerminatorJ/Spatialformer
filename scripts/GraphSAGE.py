import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import pdb; pdb.set_trace()
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, negative_sampling, add_self_loops
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import to_networkx, from_networkx
from scipy.spatial import KDTree
from torchmetrics.classification import BinaryAccuracy

from pytorch_lightning.loggers import WandbLogger
import networkx as nx
import logging
from tqdm import tqdm
import json
import random
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import argparse
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
import pickle
import os
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

path = Path(os.getcwd())
parent_dir = path.parent
# import pdb; pdb.set_trace()
data_dir = os.path.join(parent_dir, "david_data")
model_path = os.path.join(parent_dir, "output", "GraphSAGE_model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Step 1: Load and preprocess data
def load_and_preprocess_data(filepath):
    dataset = pd.read_csv(filepath)
    dataset = dataset[dataset['qv'] >= 20]
    dataset = dataset[~(dataset['feature_name'].str.startswith('Neg') | dataset['feature_name'].str.startswith('BLANK') | dataset['feature_name'].str.startswith('Unassigned'))]
    #only use partial data
    # dataset = dataset.iloc[:10000]
    return dataset

def index2gene(filepath):
    dataset = pd.read_csv(filepath)
    dataset = dataset[dataset['qv'] >= 20]
    dataset = dataset[~(dataset['feature_name'].str.startswith('Neg') | dataset['feature_name'].str.startswith('BLANK') | dataset['feature_name'].str.startswith('Unassigned'))]

    gene_labels = dataset['feature_name'].values 
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_labels = encoder.fit_transform(gene_labels.reshape(-1, 1))
    gene_names = encoder.categories_[0]
    gene_to_index = {gene_name: index for index, gene_name in enumerate(gene_names)}
    # If needed, reverse the dictionary to go from index to gene name
    # import pdb;pdb.set_trace()
    index_to_gene = {index: gene_name for index, gene_name in enumerate(gene_names)}
    return index_to_gene, gene_to_index


def build_graph_for_sample(data, threshold=3.0, batch_size=100, sample_id = None):
    # import pdb; pdb.set_trace()
    r_c = np.array(data[['x_location', 'y_location', 'z_location']])
    gene_labels = data['feature_name'].values 
    # import pdb; pdb.set_trace()
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_labels = encoder.fit_transform(gene_labels.reshape(-1, 1))
    # import pdb; pdb.set_trace()
    kdtree = KDTree(r_c)
    G = nx.Graph()
    # import pdb; pdb.set_trace()
    for i in range(len(r_c)):
        G.add_node(i, feature=one_hot_labels[i])
    # import pdb; pdb.set_trace()
    num_nodes = len(r_c)
    
    print("building graph in batch")
    for start_idx in tqdm(range(0, num_nodes, batch_size)):
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
            
    # Batch add edges
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    edge_index = torch.tensor(list(G.edges)).t().contiguous()   
    x = torch.tensor(one_hot_labels, dtype=torch.float)

    num_nodes = x.size(0)

    root_nodes = torch.tensor(random.sample(range(num_nodes), min(5000, num_nodes)))

    # import pdb; pdb.set_trace()
    print("creating the subgraph...")
    subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
        node_idx=root_nodes,
        num_hops=3,
        edge_index=edge_index,
        relabel_nodes=True
    )
    x_subgraph = x[subgraph_nodes]
    # import pdb; pdb.set_trace()
    data = Data(x=x_subgraph, edge_index=subgraph_edge_index)
    

    print("saving the graph")
    torch.save(data, f'{data_dir}/subgraph_data_{sample_id}.pt')



# Step 2: Create subgraphs
def create_subgraph(data, num_root_nodes=5000, num_neighbors=[20, 10, 10]):
    num_nodes = data.x.size(0)
    root_nodes = torch.tensor(random.sample(range(num_nodes), min(num_root_nodes, num_nodes)))

    subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
        node_idx=root_nodes,
        num_hops=len(num_neighbors),
        edge_index=data.edge_index,
        relabel_nodes=True
    )

    x_subgraph = data.x[subgraph_nodes]

    components = [c for c in nx.connected_components(G) if len(c) >= 10]
    G = G.subgraph(set.union(*map(set, components)))


    return Data(x=x_subgraph, edge_index=subgraph_edge_index)


# Step 4: Define and train the 2-hop GraphSAGE Model
class TwoHopGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TwoHopGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
def filter_component(data):
    # import pdb; pdb.set_trace()
    print("subgraph to networkx...")
    sub_G = to_networkx(data, to_undirected=True, node_attrs=['x'])
    print("filtering by components...")
    components = [c for c in nx.connected_components(sub_G) if len(c) >= 10]
    sub_G_f = sub_G.subgraph(set.union(*map(set, components)))

    data_filtered = from_networkx(sub_G_f, group_node_attrs=['x'])
    return data_filtered



# Define your LightningModule
class GraphSAGEModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr, train_dataset, batch_size):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.accuracy = BinaryAccuracy() 
        self.lr = lr
        self.train_dataset = train_dataset
        self.batch_size = batch_size
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def training_step(self, batch, batch_idx):
        z = self(batch.x, batch.edge_index)  # Pass through the model

        # Use positive samples from existing edges
        pos_edge_index = batch.edge_index
        
        # Generate negative samples
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=batch.x.size(0),
            num_neg_samples=pos_edge_index.size(1)
        )

        # Compute dot product of embeddings for positive and negative samples
        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

        # Concatenate all outputs and create labels
        all_out = torch.cat([pos_out, neg_out])
        all_labels = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        
        # Define the binary classification loss
        loss = F.binary_cross_entropy_with_logits(all_out, all_labels)
        
        # Evaluate accuracy
        preds = torch.sigmoid(all_out) > 0.5
        acc = self.accuracy(preds, all_labels.int())

        # Log loss and accuracy
        self.log('train_loss', loss, sync_dist=True, reduce_fx='mean', prog_bar=True, batch_size=batch.x.size(0))
        self.log('train_acc', acc, sync_dist=True, reduce_fx='mean', prog_bar=True, batch_size=batch.x.size(0))


        return loss

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=0.001)
    
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        return optimizer
    
    def train_dataloader(self):
    
        loader = NeighborLoader(
            data=self.train_dataset,
            num_neighbors=[20, 10],  # 20 first-hop, 10 second-hop neighbors
            batch_size=self.batch_size,  # Example batch size for root nodes
            shuffle=True,
            num_workers=4  # Set based on your CPU availability
        )
        return loader
    






class MyTrainer:
    def __init__(self, config, train_dataset):
        self.config = config
        self.plmodel = GraphSAGEModel(
            input_dim=config["feature_num"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            lr = config["lr"],
            train_dataset = train_dataset,
            batch_size = config["batch_size"]
            )
        self.output_dir = "/home/sxr280/Spatialformer/output/GraphSAGE_model"
        self.train_dataset = train_dataset
        self.gpus = torch.cuda.device_count()
        self.trainer = None

    def make_callback(self):
        # Callbacks
        callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(self.output_dir, "GraphSAGE_model", "checkpoints"),
            filename=f"{{step:07d}}-{{train_loss:.4f}}-{{val_loss:.4f}}",
            every_n_train_steps=20000,
            save_top_k=-1,
            # every_n_epochs=1,
            monitor='train_loss',
            save_on_train_epoch_end=False
        ), LearningRateMonitor(logging_interval="step"),
        # EarlyStopping(monitor = "val_loss", min_delta = 0.00, verbose = True, mode = "min")
        ]

        return callbacks
    def set_trainer(self):
        self.logger = WandbLogger(project = "Spaformer", 
                                  name = "GraphSAGE", 
                                  log_model = "all", 
                                  save_dir = self.output_dir)
        # self.logger = CSVLogger("/home/sxr280/Spatialformer/output", name="my_experiment")
        
        self.trainer = pl.Trainer(
            accelerator="auto",
            devices=self.gpus,
            max_steps=self.config["total_step"],
            val_check_interval = 0.1,
            default_root_dir=self.output_dir,
            callbacks=self.make_callback(),
            log_every_n_steps=50,
            logger=self.logger,
            precision='bf16',
            strategy = self.config['strategy'],
            num_nodes = 1
        )
    def resume_train(self, ckp, train_loader, val_loader):
        self.logger = WandbLogger(project = "Spaformer", 
                                  name = "GraphSAGE", 
                                  log_model = "all", 
                                  save_dir = self.output_dir)
        # import pdb; pdb.set_trace()
        logging.info("resuming the training ...")
        self.trainer = pl.Trainer(
            accelerator="auto",
            devices=self.gpus,
            strategy = self.config['strategy'],
            num_nodes = 1,
            val_check_interval = 0.1,
            gradient_clip_val = 1,
            logger=self.logger,
            default_root_dir=self.output_dir,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            precision='bf16',
            callbacks=self.make_callback(),
            max_steps=self.config["total_step"], 
            resume_from_checkpoint=ckp,
            accumulate_grad_batches = self.config['accumulate_grad_batches'])
        self.trainer.fit(self.plmodel, train_loader)


    def train(self):
        # import pdb; pdb.set_trace()
        self.set_trainer()
        self.trainer.fit(self.plmodel)
    def get_embedding(self, index_to_gene, ckp_path, batch_size, token_path, output_dim):
        '''
        Getting the gene embeddings that merge from the transcripts
        '''
        model = self.plmodel
        ckp = torch.load(ckp_path)
        params = ckp["state_dict"]
        model.load_state_dict(params)
        
        model.eval()

        gene_embeds = {}

        #loading the token path
        with open(os.path.join(token_path), 'r') as json_file:
            token_config = json.load(json_file)
        token_num = np.max([j for i,j in token_config.items()]) + 1
        pretrained_embeddings = torch.rand(token_num, output_dim)



        # import pdb;pdb.set_trace()
        # Ensure no gradient tracking during evaluation
        with torch.no_grad():
            indices = torch.argmax(self.train_dataset.x, axis=1)
            genes = [index_to_gene[indice.item()] for indice in indices]
            # Generate embeddings for all nodes
            embeddings = model(self.train_dataset.x, self.train_dataset.edge_index)
            # Group embeddings by gene
            for i, gene in enumerate(genes):
                if gene not in gene_embeds:
                    gene_embeds[gene] = []
                gene_embeds[gene].append(embeddings[i])
                
        # gene_embed = {gene: torch.mean(torch.stack(embeds), dim=0) for gene, embeds in gene_embeds.items()}
        #transfer gene to embedding by token ids
        for gene, embeds in gene_embeds.items():
            # import pdb; pdb.set_trace()
            pretrained_embeddings[token_config[gene]] = torch.mean(torch.stack(embeds), dim=0)

        #settign the padding as 0
        pretrained_embeddings[0] = 0
        # import pdb; pdb.set_trace()
        return pretrained_embeddings


        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='calculate the gene graph')
    parser.add_argument('--save_graph', action = 'store_true', help='only save the graph for each sample')
    args = parser.parse_args()



    sample_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if "transcripts.csv" in file]
    index_to_gene, gene_to_index = index2gene(sample_files[0])
    if args.save_graph:
        #saving all the intermediate data
        # import pdb; pdb.set_trace()
        # build_graph_for_sample(load_and_preprocess_data(sample_files[0]), sample_id = sample_files[0].split("__")[-3])
        all_samples = [build_graph_for_sample(load_and_preprocess_data(sample_file), sample_id = sample_file.split("__")[-3]) for sample_file in sample_files if f'subgraph_data_{sample_file.split("__")[-3]}.pt' not in os.listdir(data_dir)]
        # import pdb; pdb.set_trace()
    else:
        print("WARNNING: please make sure you have already save the graph for each sample")
        subgraphs = [torch.load(os.path.join(data_dir,f"subgraph_data_{sample_file.split('__')[-3]}.pt")) for sample_file in sample_files]
        # subgraphs = 10*[torch.load("/home/sxr280/Spatialformer/david_data/subgraph_data_VUILD104MF.pt")]
        # subgraphs = [filter_component(subgraph) for subgraph in subgraphs]
        #exclude the sample with only 342 genes
        subgraphs = [j for i,j in enumerate(subgraphs) if i!=11]
    # import pdb; pdb.set_trace()
    print("building the full graph")
    # Combine subgraphs into a joint large graph
    joint_x = torch.cat([subgraph.x for subgraph in subgraphs], dim=0)
    offset = 0
    edge_lists = []
    for subgraph in subgraphs:
        edge_lists.append(subgraph.edge_index + offset)
        offset += subgraph.x.size(0)
    # import pdb; pdb.set_trace()
    joint_edge_index = torch.cat(edge_lists, dim=1)
    # import pdb; pdb.set_trace()
    joint_graph = Data(x=joint_x, edge_index=joint_edge_index)
    # import pdb; pdb.set_trace()
    print("building the dataloader")
    # Step 2: Use NeighborLoader for specific sampling strategies
    
    # import pdb; pdb.set_trace()

    
    print("training the model")
    # Example execution
    with open(os.path.join("/home/sxr280/Spatialformer/config/_config_graphsave.json"), 'r') as json_file:
        config = json.load(json_file)
    trainer = MyTrainer(config, joint_graph)
    # trainer.train()
    # torch.save(model, f'{model_path}/model_{current_time}.pt')
    #getting the embeddings
    embeddings = trainer.get_embedding(index_to_gene, "/home/sxr280/Spatialformer/output/GraphSAGE_model/GraphSAGE_model/checkpoints/step=0900000-train_total_loss=0.0000-val_total_loss=0.0000.ckpt", 32,
                                       "/home/sxr280/Spatialformer/tokenizer/token.json", config["output_dim"])
    pickle.dump(embeddings, open("/home/sxr280/Spatialformer/data/gene_embeddings_GraphSAGE.pkl", "wb"))
    


    



    
