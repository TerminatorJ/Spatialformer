"""
GraphSAGE Model for Spatial Transcriptomics Data
Refactored version with improved data loading from parquet files
Compatible with the find_gene_interaction.py data pipeline
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import gc
import sys
import json
import random
import logging
from collections import defaultdict
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, negative_sampling, to_networkx, from_networkx

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from scipy.spatial import KDTree
from sklearn.preprocessing import OneHotEncoder
import networkx as nx

from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# =====================================================================
# Data Loading and Preprocessing
# =====================================================================

class TranscriptDataLoader:
    """Handles loading and preprocessing of transcript data from parquet/csv files"""
    
    def __init__(self, 
                 gene_vocab_path: str,
                 min_transcripts_per_cell: int = 30,
                 max_cells_per_slide: Optional[int] = None):
        """
        Args:
            gene_vocab_path: Path to gene vocabulary JSON file
            min_transcripts_per_cell: Minimum transcripts to keep a cell
            max_cells_per_slide: Maximum number of cells to sample per slide (None = all)
        """
        self.gene_vocab_path = gene_vocab_path
        self.min_transcripts = min_transcripts_per_cell
        self.max_cells_per_slide = max_cells_per_slide

        # Load gene vocabulary
        with open(gene_vocab_path, 'r') as f:
            token_dict = json.load(f)
        # Extract gene list (skip special tokens)
        self.gene_vocab = [k for k in token_dict.keys() if not k.startswith('<')]
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_vocab)}

        logging.info(f"Loaded {len(self.gene_vocab)} genes from vocabulary")
        
    def load_and_save_data(self, transcript_file, save_path: Optional[str] = None) -> pd.DataFrame:
        """Load transcript data from file and save it to local if save_path is provided"""
        """
        transcript_file: paths to transcript files (parquet directory, csv, or csv.gz)
        """
        filename = os.path.basename(os.path.dirname(transcript_file))
        # to see whether this filename has be processed
        os.makedirs(save_path, exist_ok=True)
        file_generated = False
        if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
            
            file_generated = any(f.endswith('.parquet') and filename in f for f in os.listdir(save_path))  
        
        #main logic
        if file_generated:
            logging.info(f"Loading data from cached file {save_path} for filename {filename}")
            return pd.concat([pd.read_parquet(os.path.join(save_path, f)) for f in os.listdir(save_path) if f.endswith('.parquet') and filename in f])

        else:
        
            input_path = Path(transcript_file)
            
            logging.info(f"Loading data from {transcript_file}")
            # Determine file type and load

            if input_path.is_dir():
                # Parquet directory
                ddf = dd.read_parquet(transcript_file)
                # Fill NA cell_ids with UNASSIGNED
                ddf["cell_id"] = ddf["cell_id"].fillna("UNASSIGNED")

            # /scratch/project_465001820/Spatialformer/data/raw/Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon_outs/transcripts.parquet

            
            elif transcript_file.endswith("gz"):
                # transcript_file = os.path.join(self.transcript_file, "transcripts.gz")
                ddf = dd.read_csv(transcript_file, compression='gzip', blocksize='64MB')
            elif transcript_file.endswith("csv"):
                # transcript_file = os.path.join(self.transcript_file, "transcripts.csv")
                ddf = dd.read_csv(transcript_file, blocksize='64MB')
            elif transcript_file.endswith("parquet"):
                # transcript_file = os.path.join(self.transcript_file, "transcripts.parquet")
                ddf = dd.read_parquet(transcript_file)
            else:
                raise ValueError(f"Unsupported file format: {transcript_file}")
            # Standardize column names
            ddf = ddf.rename(columns={
                'x_location': 'x',
                'y_location': 'y', 
                'z_location': 'z',
                'feature_name': 'gene'
            })
            # Filter genes (remove negative controls, blanks, etc.)
            logging.info("Filtering genes...")
            ddf = ddf[~ddf['gene'].str.startswith((
                'Neg', 'BLANK', 'Unassigned', 'Deprecated', 'Intergenic', 'Total', 'Human'
            ))]
            
            # Select only needed columns
            required_cols = ['x', 'y', 'z', 'gene', 'cell_id']
            ddf = ddf[required_cols]
            
            # Filter cells by transcript count
            logging.info(f"Filtering cells with >= {self.min_transcripts} transcripts...")
            with ProgressBar():
                value_counts = ddf['cell_id'].value_counts().compute()
            
            # Remove UNASSIGNED if present
            try:
                clean_value_counts = value_counts.drop("UNASSIGNED")
                valid_cells = clean_value_counts.index[clean_value_counts >= self.min_transcripts]
            except KeyError:
                valid_cells = value_counts.index[value_counts >= self.min_transcripts]
            
            # Sample cells if max_cells is specified
            if self.max_cells_per_slide and len(valid_cells) > self.max_cells_per_slide:
                logging.info(f"Sampling {self.max_cells_per_slide} cells from {len(valid_cells)} total")
                valid_cells = pd.Series(valid_cells).sample(n=self.max_cells_per_slide, random_state=42).values
            
            valid_cells_set = set(valid_cells)
            ddf = ddf[ddf['cell_id'].isin(valid_cells_set)]
            
            # Convert to pandas for processing
            logging.info("Converting to pandas DataFrame...")
            with ProgressBar():
                df = ddf.compute()
                
            logging.info(f"Save to pandas DataFrame for slide {filename}")
            df.to_parquet(f"{os.path.abspath(save_path)}/{filename}.parquet")
            
            logging.info(f"Loaded {len(df)} transcripts from {len(valid_cells)} cells")
            logging.info(f"Unique genes: {df['gene'].nunique()}")
            
            return df


# =====================================================================
# Graph Construction
# =====================================================================

class SpatialGraphBuilder:
    """Builds spatial proximity graphs from transcript coordinates"""
    
    def __init__(self, 
                 radius: float = 3.0,
                 batch_size: int = 1000,
                 is_3d: bool = True):
        """
        Args:
            radius: Spatial radius threshold for edges
            batch_size: Batch size for edge construction
            is_3d: Whether to use 3D coordinates (x, y, z) or 2D (x, y)
        """
        self.radius = radius
        self.batch_size = batch_size
        self.is_3d = is_3d
        
    def build_graph(self, 
                   df: pd.DataFrame,
                   gene_vocab: List[str]) -> Data:
        """
        Build PyTorch Geometric graph from transcript dataframe
        
        Args:
            df: Transcript dataframe with columns [x, y, z, gene, cell_id]
            gene_vocab: List of gene names for encoding
            
        Returns:
            PyTorch Geometric Data object
        """
        # Get coordinates
        coord_cols = ['x', 'y', 'z'] if self.is_3d else ['x', 'y']
        coords = df[coord_cols].values
        
        # Encode genes as one-hot
        logging.info("Encoding genes as one-hot vectors...")
        encoder = OneHotEncoder(categories=[gene_vocab], sparse_output=False)
        gene_features = encoder.fit_transform(df['gene'].values.reshape(-1, 1))
        
        # Build spatial edges using KDTree
        logging.info("Building spatial graph...")
        kdtree = KDTree(coords)
        
        # Initialize NetworkX graph for construction
        G = nx.Graph()
        num_nodes = len(coords)
        
        # Add nodes with features
        for i in range(num_nodes):
            G.add_node(i, feature=gene_features[i])
        
        # Add edges in batches
        logging.info(f"Adding edges with radius={self.radius}...")
        for start_idx in tqdm(range(0, num_nodes, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, num_nodes)
            batch_coords = coords[start_idx:end_idx]
            
            edges_to_add = []
            for i, coord in enumerate(batch_coords, start=start_idx):
                neighbors = kdtree.query_ball_point(coord, self.radius)
                for j in neighbors:
                    if i < j:  # Avoid duplicates
                        edges_to_add.append((i, j))
            
            G.add_edges_from(edges_to_add)
        
        # Convert to PyTorch Geometric format
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        node_features = torch.tensor(gene_features, dtype=torch.float)
        
        logging.info(f"Graph built: {num_nodes} nodes, {edge_index.size(1)} edges")
        
        return Data(x=node_features, edge_index=edge_index)
    
    def create_subgraph(self,
                       data: Data,
                       num_root_nodes: int = 5000,
                       num_hops: int = 3) -> Data:
        """
        Create subgraph by k-hop sampling from random root nodes
        
        Args:
            data: Full graph data
            num_root_nodes: Number of random root nodes to sample from
            num_hops: Number of hops for neighborhood
            
        Returns:
            Subgraph Data object
        """
        num_nodes = data.x.size(0)
        root_nodes = torch.tensor(
            random.sample(range(num_nodes), min(num_root_nodes, num_nodes))
        )
        
        logging.info(f"Creating {num_hops}-hop subgraph from {len(root_nodes)} root nodes...")
        
        subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
            node_idx=root_nodes,
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True
        )
        
        x_subgraph = data.x[subgraph_nodes]
        
        logging.info(f"Subgraph: {x_subgraph.size(0)} nodes, {subgraph_edge_index.size(1)} edges")
        
        return Data(x=x_subgraph, edge_index=subgraph_edge_index)
    
    def filter_by_component_size(self, 
                                 data: Data,
                                 min_component_size: int = 10) -> Data:
        """
        Filter graph to keep only connected components above size threshold
        
        Args:
            data: Graph data
            min_component_size: Minimum component size to keep
            
        Returns:
            Filtered Data object
        """
        logging.info("Filtering by connected components...")
        G = to_networkx(data, to_undirected=True, node_attrs=['x'])
        
        # Keep large components
        components = [c for c in nx.connected_components(G) if len(c) >= min_component_size]
        G_filtered = G.subgraph(set.union(*map(set, components)))
        
        data_filtered = from_networkx(G_filtered, group_node_attrs=['x'])
        
        logging.info(f"Kept {data_filtered.x.size(0)} nodes after filtering")
        
        return data_filtered


# =====================================================================
# GraphSAGE Models
# =====================================================================

class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder"""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGELinkPrediction(pl.LightningModule):
    """GraphSAGE for link prediction (self-supervised pretraining)"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = GraphSAGEEncoder(input_dim, hidden_dim, output_dim)
        self.accuracy = BinaryAccuracy()
        self.lr = learning_rate
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)
    
    def training_step(self, batch, batch_idx):
        z = self(batch.x, batch.edge_index)
        
        # Positive edges
        pos_edge_index = batch.edge_index
        
        # Negative sampling
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=batch.x.size(0),
            num_neg_samples=pos_edge_index.size(1)
        )
        
        # Compute scores
        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        
        # Binary classification
        all_out = torch.cat([pos_out, neg_out])
        all_labels = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        
        loss = F.binary_cross_entropy_with_logits(all_out, all_labels)
        
        # Metrics
        preds = torch.sigmoid(all_out) > 0.5
        acc = self.accuracy(preds, all_labels.int())
        
        self.log('train_loss', loss, prog_bar=True, batch_size=batch.x.size(0))
        self.log('train_acc', acc, prog_bar=True, batch_size=batch.x.size(0))
        
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)


class GraphSAGEPatternClassifier(pl.LightningModule):
    """GraphSAGE for spatial pattern classification"""
    
    def __init__(self,
                 encoder: GraphSAGEEncoder,
                 output_dim: int,
                 num_classes: int = 2,
                 hidden_dim: int = 512,
                 learning_rate: float = 1e-3,
                 freeze_encoder: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.hidden_layer = nn.Linear(output_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.lr = learning_rate
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, edge_index)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        data, labels = batch
        preds = self(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(preds, labels)
        
        acc = self.accuracy(preds.argmax(dim=1), labels)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        preds = self(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(preds, labels)
        
        acc = self.accuracy(preds.argmax(dim=1), labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)


# =====================================================================
# Training Pipeline
# =====================================================================

class GraphSAGETrainer:
    """Handles training and evaluation of GraphSAGE models"""
    
    def __init__(self,
                 config: Dict,
                 output_dir: str = "./output/graphsage"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def create_dataloader(self, data: Data, batch_size: int, shuffle: bool = True) -> NeighborLoader:
        """Create NeighborLoader for mini-batch training"""
        loader = NeighborLoader(
            data=data,
            num_neighbors=[20, 10],  # 2-hop neighborhood
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4
        )
        return loader
    
    def train_link_prediction(self,
                              train_data: Data,
                              input_dim: int,
                              max_steps: int = 50000,
                              batch_size: int = 512,
                              use_wandb: bool = False) -> GraphSAGELinkPrediction:
        """
        Train GraphSAGE with link prediction (self-supervised)
        
        Args:
            train_data: Training graph data
            input_dim: Input feature dimension
            max_steps: Maximum training steps
            batch_size: Batch size
            use_wandb: Whether to use Weights & Biases logging
            
        Returns:
            Trained model
        """
        # Create model
        model = GraphSAGELinkPrediction(
            input_dim=input_dim,
            hidden_dim=self.config.get('hidden_dim', 256),
            output_dim=self.config.get('output_dim', 512),
            learning_rate=self.config.get('lr', 1e-3)
        )
        
        # Create dataloader
        train_loader = self.create_dataloader(train_data, batch_size, shuffle=True)
        
        # Setup logger
        if use_wandb:
            logger = WandbLogger(
                project="Spatialformer",
                name=f"GraphSAGE_LinkPred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                save_dir=str(self.output_dir)
            )
        else:
            logger = CSVLogger(str(self.output_dir), name="link_prediction")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=str(self.checkpoint_dir / "link_prediction"),
                filename="step={step:07d}-loss={train_loss:.4f}-acc={train_acc:.4f}",
                every_n_train_steps=5000,
                save_top_k=-1,
                monitor='train_loss'
            ),
            LearningRateMonitor(logging_interval="step")
        ]
        
        # Create trainer
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            max_steps=max_steps,
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=50,
            precision='bf16',
            default_root_dir=str(self.output_dir)
        )
        
        # Train
        logging.info("Starting link prediction training...")
        trainer.fit(model, train_loader)
        
        return model
    
    
    def extract_gene_embeddings(
                        self,
                        model,
                        data: Data,              
                        gene_vocab,
                        token_dict_path,
                        output_path,
                        batch_size=1024,
                    ):

        device = next(model.parameters()).device
        model.eval()

        with open(token_dict_path) as f:
            token_dict = json.load(f)

        out_dim = self.config.get("output_dim", 512)

        # ---- NeighborLoader ----
        loader = NeighborLoader(
            data,
            num_neighbors=[20, 10],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        embeddings = torch.zeros(data.num_nodes, out_dim)

        gene_embeds = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(loader, desc="NeighborLoader inference"):
                batch = batch.to(device)

                out = model(batch.x, batch.edge_index)

                # only keep seed nodes
                seed_out = out[:batch.batch_size]
                seed_nid = batch.n_id[:batch.batch_size]

                embeddings[seed_nid] = seed_out.cpu()

                # transcript → gene mapping
                gene_idx = torch.argmax(batch.x[:batch.batch_size], dim=1).cpu()
                for i, gidx in enumerate(gene_idx):
                    gene = gene_vocab[gidx.item()]
                    gene_embeds[gene].append(seed_out[i].cpu())

        # ---- average gene embeddings ----
        token_num = max(token_dict.values()) + 1
        pretrained_embeddings = torch.zeros(token_num, out_dim)

        for gene, embeds in gene_embeds.items():
            if gene in token_dict:
                pretrained_embeddings[token_dict[gene]] = torch.stack(embeds).mean(0)

        with open(output_path, "wb") as f:
            pickle.dump(pretrained_embeddings, f)

        return pretrained_embeddings


# =====================================================================
# Main Execution
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='GraphSAGE for Spatial Transcriptomics')
    
    # Data arguments
    parser.add_argument('--transcript_files', type=str, nargs = "+", required=True,
                       help='Path to transcript file (parquet dir, csv, or csv.gz)')
    parser.add_argument('--save_path', type=str, required=True,
                       help='Path to save processed data')
    parser.add_argument('--gene_vocab', type=str, required=True,
                       help='Path to gene vocabulary JSON file')
    parser.add_argument('--min_transcripts', type=int, default=30,
                       help='Minimum transcripts per cell')
    parser.add_argument('--max_cells', type=int, default=None,
                       help='Maximum cells to sample (None = all)')
    
    # Graph arguments
    parser.add_argument('--radius', type=float, default=3.0,
                       help='Spatial radius for edges')
    parser.add_argument('--num_root_nodes', type=int, default=5000,
                       help='Number of root nodes for subgraph sampling')
    parser.add_argument('--num_hops', type=int, default=3,
                       help='Number of hops for subgraph')
    
    # Training arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=512,
                       help='Output embedding dimension')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=50000,
                       help='Maximum training steps')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output/graphsage',
                       help='Output directory')
    parser.add_argument('--save_graph', action='store_true',
                       help='Save processed graph to disk')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # =========================================================
    # Step 1: Load Data
    # =========================================================
    subgraphs = []
    all_files = args.transcript_files
    data_loader = TranscriptDataLoader(
                gene_vocab_path=args.gene_vocab,
                min_transcripts_per_cell=args.min_transcripts,
                max_cells_per_slide=args.max_cells
            )
    # for file in all_files:
    #     filename = os.path.basename(os.path.dirname(file))
    #     graph_path = os.path.join(output_dir,f"{filename}_processed_graph.pt")
        
    #     if not os.path.exists(graph_path):


    #         df = data_loader.load_and_save_data(file, args.save_path)
    #         # =========================================================
    #         # Step 2: Build Graph
    #         # =========================================================
    #         graph_builder = SpatialGraphBuilder(
    #             radius=args.radius,
    #             batch_size=1000,
    #             is_3d=True
    #         )

    #         logging.info("Building spatial graph...")
    #         graph_data = graph_builder.build_graph(df, data_loader.gene_vocab)
    #         # Create subgraph
    #         subgraph_data = graph_builder.create_subgraph(
    #             graph_data,
    #             num_root_nodes=args.num_root_nodes,
    #             num_hops=args.num_hops
    #         )
            
    #         del graph_data
    #         gc.collect()

    #         # Save graph if requested
    #         if args.save_graph:

    #             logging.info(f"Saving graph to {graph_path}")
    #             torch.save(subgraph_data, graph_path)
    #         del subgraph_data
    #         gc.collect()
    #     else:
    #         subgraph = torch.load(graph_path)
    #         subgraphs.append(subgraph)
    subgraphs = []
    basedir = "/tmp/erda/Spatialformer/graphsage/"
    all_files = os.listdir(basedir)
    for file in all_files:

        subgraph = torch.load(os.path.join(basedir, file))
        subgraphs.append(subgraph)


    print("building the full graph")
    logging.info(f"loading the {len(subgraphs)} subgraphs")

    # Combine subgraphs into a joint large graph
    logging.info(f"Concat the {len(subgraphs)} subgraphs")
    joint_x = torch.cat([subgraph.x for subgraph in subgraphs], dim=0)

    offset = 0
    edge_lists = []
    for subgraph in subgraphs:
        edge_lists.append(subgraph.edge_index + offset)
        offset += subgraph.x.size(0)
    joint_edge_index = torch.cat(edge_lists, dim=1)
    joint_graph = Data(x=joint_x, edge_index=joint_edge_index)
        
    # =========================================================
    # Step 3: Train GraphSAGE
    # =========================================================
    config = {
        'hidden_dim': args.hidden_dim,
        'output_dim': args.output_dim,
        'lr': args.lr
    }
    
    trainer = GraphSAGETrainer(config, output_dir=args.output_dir)

    #mute it in testing
    # model = trainer.train_link_prediction(
    #     train_data=joint_graph,
    #     input_dim=len(data_loader.gene_vocab),
    #     max_steps=args.max_steps,
    #     batch_size=args.batch_size,
    #     use_wandb=args.use_wandb
    # )
    
    # =========================================================
    # Step 4: Extract Gene Embeddings
    # =========================================================
    #import the lightning model from a checkpoint
    model = GraphSAGELinkPrediction(
            input_dim=6061,
            hidden_dim=256,
            output_dim=512,
            learning_rate=1e-3
        )
    # Create model
    ckp = torch.load("/home/sxr280/Spatialformer/output/graphsage/checkpoints/link_prediction/step=step=0015000-loss=train_loss=0.3993-acc=train_acc=0.8034.ckpt")
    params = ckp["state_dict"]
    model.load_state_dict(params)
    model.eval()
    # import pdb; pdb.set_trace()
    embeddings_path = output_dir / 'gene_embeddings.pkl'
    
    trainer.extract_gene_embeddings(
        model=model,
        data=joint_graph,
        gene_vocab=data_loader.gene_vocab,
        token_dict_path=args.gene_vocab,
        output_path=str(embeddings_path)
    )
    
    logging.info("Training complete!")


if __name__ == "__main__":
    main()
