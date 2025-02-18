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
from torch_geometric.nn import global_mean_pool
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import MulticlassAccuracy
from pytorch_lightning.loggers import WandbLogger, CSVLogger
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
# data_dir = os.path.join(parent_dir, "david_data")

model_path = os.path.join(parent_dir, "output", "GraphSAGE_model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Step 1: Load and preprocess data
def load_and_preprocess_data(filepath):
    print(f"loading {filepath}")
    try:
        dataset = pd.read_csv(filepath)
    except:
        dataset = pd.read_csv(filepath + ".gz")
    print("after loading the data")
    if "qv" in dataset.columns:
        dataset = dataset[dataset['qv'] >= 20]
    dataset = dataset[~(dataset['feature_name'].str.startswith('Neg') | dataset['feature_name'].str.startswith('BLANK') | dataset['feature_name'].str.startswith('Unassigned'))]
    #only use partial data
    # Ensure that we work with distinct cell IDs
    unique_cell_ids = dataset['cell_id'].unique()
    # Randomly sample 20,000 unique cell_ids
    # Make sure that there are at least 20,000 unique cell IDs before sampling
    if len(unique_cell_ids) >= 20000:
        sampled_cell_ids = pd.Series(unique_cell_ids).sample(n=20000, random_state=42)
    else:
        sampled_cell_ids = pd.Series(unique_cell_ids)
    # import pdb; pdb.set_trace()
    # Select only the rows where `cell_id` is in the sampled set
    sampled_dataset = dataset[dataset['cell_id'].isin(sampled_cell_ids)]
    # dataset = dataset.iloc[:10000]
    return sampled_dataset

def index2gene(filepath):
    #get the reference gene vocabulary
    with open(filepath, "r") as f:
        vocab = json.load(f)
    vocab_list = list(vocab.keys())[27:]
    return vocab_list


def build_graph_for_sample(data, threshold=3.0, batch_size=100, sample_id = None):
    # import pdb; pdb.set_trace()
    r_c = np.array(data[['x_location', 'y_location', 'z_location']])
    gene_labels = data['feature_name'].values 
    # Convert vocab dictionary keys to a sorted list
    # Initialize the OneHotEncoder with the specific categories
    encoder = OneHotEncoder(categories=[vocab_list], sparse_output=False)
    # Transform the gene_labels into one-hot encoded format
    one_hot_labels = encoder.fit_transform(gene_labels.reshape(-1, 1))
    # import pdb; pdb.set_trace()
    
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
    # import pdb; pdb.set_trace()
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
    # import pdb; pdb.set_trace()
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
    


class PatternGraphSAGEModel(pl.LightningModule):
    def __init__(self, model, output_dim, nn_hidden_layer_dim, lr):
        super(PatternGraphSAGEModel, self).__init__()
        self.model = model
        self.fc = nn.Linear(512, 2)  # 5 classes
        self.hidden_layer = nn.Linear(output_dim, nn_hidden_layer_dim)

        # self.label_str = ["random", "extranuclear", "perinuclear", "pericellular", "intranuclear"]
        self.label_str = ["random", "non-random"]
        self.accuracy = MulticlassAccuracy(num_classes=len(self.label_str))

        self.lr = lr

    def forward(self, x, edge_index, batch_indice):
        x = self.model(x, edge_index)
        # x = F.relu(x)
        # Activation is not required before the hidden layer
        # x = self.hidden_layer(x)
        x = F.relu(x)  # Activation after the hidden layer output
        # Use global mean pooling
        x = global_mean_pool(x, batch_indice)
        # Finally apply the classification layer
        x = self.fc(x)

        return x  # Return raw logits for CrossEntropyLoss

    def training_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        preds = self(batch[0].x, batch[0].edge_index, batch[0].batch) # Pass through the model
        # labels = torch.tensor([self.label_str.index(label_str) for label_str in batch[1]], dtype=torch.long).to(preds.device)
        labels = torch.tensor([0 if label_str == "random" else 1 for label_str in batch[1]], dtype=torch.long).to(preds.device)

        # labels = torch.tensor([0 for label_str in batch[1] if label_str != "intranuclear"], dtype=torch.long).to(preds.device)
        # Define the multi-class classification loss
        loss = F.cross_entropy(preds, labels)  # For multi-class

        # Evaluate accuracy
        # acc = self.accuracy(preds, labels)
        acc = self.accuracy(preds.argmax(dim=1), labels)
        # Log loss and accuracy
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        self.log('train_acc', acc, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
       
        preds = self(batch[0].x, batch[0].edge_index, batch[0].batch)  # Pass through the model
        # labels = torch.tensor([self.label_str.index(label_str) for label_str in batch[1]], dtype=torch.long).to(preds.device)
        # Convert patterns from strings to binary labels
        labels = torch.tensor([0 if label_str == "random" else 1 for label_str in batch[1]], dtype=torch.long).to(preds.device)

        # labels = torch.tensor([0 if label_str != "intranuclear" for label_str in batch[1] else 1], dtype=torch.long).to(preds.device)
        # import pdb; pdb.set_trace()
        # Define the multi-class classification loss
        # import pdb; pdb.set_trace()
        # criterion = torch.nn.CrossEntropyLoss()
        # criterion(preds, labels)
        loss = F.cross_entropy(preds, labels)  # For multi-class

        # Evaluate accuracy
        acc = self.accuracy(preds.argmax(dim=1), labels)
        # acc = self.accuracy(preds, labels)

        # Log loss and accuracy
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log('val_acc', acc, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        return optimizer

    def evaluation(self, ckp_path, test_dataloader):
        ckp = torch.load(ckp_path)
        params = ckp["state_dict"]
        self.model.load_state_dict(params)
        self.model.eval()  # Set the model to evaluation mode
        all_preds = []
        all_labels = []

        with torch.no_grad():  # Disable gradient calculation for efficiency
            for batch in test_dataloader:
                preds = self(batch[0].x, batch[0].edge_index)  # Forward pass
                labels = torch.tensor([self.label_str.index(label_str) for label_str in batch[1]], dtype=torch.long)

                all_preds.append(preds.softmax(dim=1))  # Get predicted probabilities
                all_labels.append(labels)  # Collect true labels

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)  # Shape: [num_samples, num_classes]
        all_labels = torch.cat(all_labels)  # Shape: [num_samples]

        # Calculate metrics
        acc = self.accuracy(all_preds.argmax(dim=1), all_labels)  # Accuracy
        f1 = self.f1(all_preds, all_labels)  # F1 Score
        mcc = self.mcc(all_preds, all_labels)  # Matthews Correlation Coefficient

        # Calculate AUC (average over one-vs-rest for multi-class)
        AUCs = []
        for i in range(len(self.label_str)):
            AUCs.append(self.auc(all_preds[:, i], (all_labels == i).float()))

        average_auc = sum(AUCs) / len(AUCs)  # Average AUC over all classes

        # Log the metrics
        self.log('test_acc', acc, sync_dist=True)
        self.log('test_f1', f1, sync_dist=True)
        self.log('test_mcc', mcc, sync_dist=True)
        self.log('test_auc', average_auc, sync_dist=True)

        # Print final metrics
        print(f"Test Accuracy: {acc.item()}")
        print(f"Test F1 Score: {f1.item()}")
        print(f"Test MCC: {mcc.item()}")
        print(f"Average Test AUC: {average_auc.item()}")
        return acc.item(), f1.item(), mcc.item(), average_auc.item()






class MyTrainer:
    def __init__(self, config, input_dim, train_dataset):
        self.config = config
        self.plmodel = GraphSAGEModel(
            input_dim = input_dim,
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
            filename=f"{{step:07d}}-{{train_loss:.4f}}-{{val_loss:.4f}}-{{train_acc:.4f}}",
            every_n_train_steps=10000,
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
    def get_embedding(self, ckp_path, batch_size, token_path, output_dim):
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
            # genes = [index_to_gene[indice.item()] for indice in indices]
            genes = [vocab_list[indice.item()] for indice in indices]
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
    
class PatternTrainer:
    def __init__(self, model, lr, strategy, output_dim, output_dir, train_dataloader, val_dataloader, test_dataloader):
        self.plmodel = PatternGraphSAGEModel(
            model,
            output_dim=output_dim,
            nn_hidden_layer_dim = 8,
            lr = lr
            )
        self.output_dir = output_dir
        self.train_dataloader = train_dataloader
        self.gpus = torch.cuda.device_count()
        self.trainer = None
        self.strategy = strategy
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def make_callback(self):
        # Callbacks
        callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(self.output_dir, "GraphSAGE_model", "checkpoints"),
            filename=f"{{step:07d}}-{{train_loss:.4f}}-{{val_loss:.4f}}-{{train_acc:.4f}}",
            every_n_train_steps=10000,
            save_top_k=-1,
            # every_n_epochs=1,
            monitor='train_loss',
            save_on_train_epoch_end=False
        ), LearningRateMonitor(logging_interval="step"),
        # EarlyStopping(monitor = "val_loss", min_delta = 0.00, verbose = True, mode = "min")
        ]

        return callbacks
    def set_trainer(self):
        # self.logger = WandbLogger(project = "Spaformer", 
        #                           name = "GraphSAGE_pattern", 
        #                           log_model = "all", 
        #                           save_dir = self.output_dir)
        self.logger = CSVLogger("/scratch/project_465001027/Spatialformer/output/GraphSAGE_model", name="my_experiment")
        
        self.trainer = pl.Trainer(
            accelerator="auto",
            devices=self.gpus,
            max_steps=10000,
            val_check_interval = 0.1,
            default_root_dir=self.output_dir,
            callbacks=self.make_callback(),
            log_every_n_steps=50,
            logger=self.logger,
            precision='bf16',
            strategy = self.strategy,
            num_nodes = 1,
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
        self.trainer.fit(self.plmodel, self.train_dataloader, self.val_dataloader)
    def evaluation(self, ckp_path):
        acc, f1, mcc, average_auc = self.plmodel.evaluation(ckp_path, self.test_dataloader)
        return acc, f1, mcc, average_auc
        

    def get_embedding(self, ckp_path, batch_size, token_path, output_dim):
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
            # genes = [index_to_gene[indice.item()] for indice in indices]
            genes = [vocab_list[indice.item()] for indice in indices]
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
    parser.add_argument('--data_dir', type=str, default = "/tmp/erda/Spatialformer/downloaded_data/raw/", help='the parent path of the data')
    parser.add_argument('--mode', type=str, default = "train", help='the mode to run the code')

    args = parser.parse_args()

    
    data_dir = args.data_dir
    mode = args.mode

    mouse_names = ["Xenium_V1_FFPE_TgCRND8_17_9_months_outs",
                   "Xenium_V1_FFPE_TgCRND8_2_5_months_outs",
                   "Xenium_V1_FFPE_TgCRND8_5_7_months_outs",
                   "Xenium_V1_FFPE_wildtype_13_4_months_outs",
                   "Xenium_V1_FFPE_wildtype_2_5_months_outs",
                   "Xenium_V1_FFPE_wildtype_5_7_months_outs",
                   "Xenium_V1_mouse_pup_outs",
                   "Xenium_V1_mouse_Colon_FF_outs",
                   "Xenium_V1_FF_Mouse_Brain_Coronal_outs",
                   "Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_outs",
                   "Xenium_Prime_Mouse_Brain_Coronal_FF_outs",
                   "Xenium_V1_mFemur_formic_acid_24hrdecal_section_outs",
                   "Xenium_V1_mFemur_EDTA_3daydecal_section_outs",
                   "Xenium_V1_mFemur_EDTA_PFA_3daydecal_section_outs",
                   "Xenium_V1_FF_Mouse_Brain_MultiSection_1_outs",
                   "Xenium_V1_FF_Mouse_Brain_MultiSection_2_outs",
                   "Xenium_V1_FF_Mouse_Brain_MultiSection_3_outs"]
    large_human_file = [
            "Xenium_Prime_Human_Ovary_FF_outs",
            "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
            "Xenium_Prime_Cervical_Cancer_FFPE_outs",
            "Xenium_Prime_Human_Skin_FFPE_outs",
            "Xenium_Prime_Human_Prostate_FFPE_outs",
            "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
            "Xenium_V1_hBoneMarrow_nondiseased_section_outs",
            "Xenium_V1_hBone_nondiseased_section_outs"
    ]
    failed_human_file = [
            "Xenium_V1_hBoneMarrow_nondiseased_section_outs",
            "Xenium_V1_hBone_nondiseased_section_outs"
    ]
    
    #only select the human Xenium datasets
    # import pdb;pdb.set_trace()
    root_path = "/tmp/erda/Spatialformer/downloaded_data/raw"

    sample_files = [
    os.path.join(root_path, file, "transcripts.csv")
    for file in os.listdir("/tmp/erda/Spatialformer/downloaded_data/raw")
    if (
        ".zip" not in file and
        file not in mouse_names and
        file not in large_human_file and
        file not in failed_human_file and (
            os.path.exists(os.path.join(root_path, file, "transcripts.csv")) or
            os.path.exists(os.path.join(root_path, file, "transcripts.csv.gz"))
        )
    )
    ]
    global vocab_list
    vocab_list = index2gene("/home/sxr280/Spatialformer/tokenizer/tokenv3.json")

    if args.save_graph:
        #saving all the intermediate data
        # import pdb; pdb.set_trace()
        print("getting the subgraph")
        file = "/tmp/erda/Spatialformer/downloaded_data/raw/Xenium_V1_hHeart_nondiseased_section_FFPE_outs/transcripts.csv"
        build_graph_for_sample(load_and_preprocess_data(file), sample_id = file.split("/")[-2])
        # import pdb; pdb.set_trace()
        # build_graph_for_sample(load_and_preprocess_data(sample_files[0]), sample_id = sample_files[0].split("__")[-3])
        # all_samples = [build_graph_for_sample(load_and_preprocess_data(sample_file), sample_id = sample_file.split("__")[-3]) for sample_file in sample_files if f'subgraph_data_{sample_file.split("__")[-3]}.pt' not in os.listdir(data_dir)]
        all_samples = [build_graph_for_sample(load_and_preprocess_data(sample_file), sample_id = sample_file.split("/")[-2]) for sample_file in sample_files if f'subgraph_data_{sample_file.split("/")[-2]}.pt' not in os.listdir(data_dir)]

    else:
        print("WARNNING: please make sure you have already save the graph for each sample")
        print(f"loading the {str(len(sample_files))} saved subgraphs")
        subgraphs = [
            torch.load(os.path.join(data_dir, f"subgraph_data_{os.path.basename(os.path.dirname(sample_file))}.pt"))
            for sample_file in sample_files
                ]

    print("building the full graph")
    # Combine subgraphs into a joint large graph
    joint_x = torch.cat([subgraph.x for subgraph in subgraphs], dim=0)
    offset = 0
    edge_lists = []
    for subgraph in subgraphs:
        edge_lists.append(subgraph.edge_index + offset)
        offset += subgraph.x.size(0)
    joint_edge_index = torch.cat(edge_lists, dim=1)
    joint_graph = Data(x=joint_x, edge_index=joint_edge_index)
    print("building the dataloader")

    print("training the model")
    # Example execution
    with open(os.path.join("/home/sxr280/Spatialformer/config/_config_graphsave.json"), 'r') as json_file:
        config = json.load(json_file)
    trainer = MyTrainer(config, len(vocab_list),joint_graph)
    #training the model
    if mode == "trian":
        trainer.train()
    elif mode == "test":

    #getting the embeddings
        embeddings = trainer.get_embedding("/home/sxr280/Spatialformer/output/GraphSAGE_model/GraphSAGE_model/checkpoints/step=0010000-train_loss=0.3983-val_loss=0.0000-train_acc=0.8256.ckpt", 32,
                                        "/home/sxr280/Spatialformer/tokenizer/tokenv3.json", config["output_dim"])
        pickle.dump(embeddings, open("/home/sxr280/Spatialformer/data/gene_embeddings_GraphSAGE_pandavid.pkl", "wb"))
    


    



    
