"""
Spatialformer Inference Module
==============================

This module provides functionality for:
1. Loading pretrained Spatialformer models
2. Tokenizing gene expression data (single cells or cell pairs)
3. Generating cell embeddings for downstream analysis
4. Predicting gene-gene co-occurrence relationships
5. Predicting cell pair relationships

Key Components:
- GeneTokenizer: Converts gene expression vectors to ranked token sequences
- GeneExpressionDataset: Dataset for single cell processing
- GeneExpressionPairDataset: Dataset for cell pair processing
- embed_data: Main function to generate embeddings from AnnData objects
"""

# =============================================================================
# Environment Setup
# =============================================================================

# Enable Flash Attention for AMD GPUs (must be set before importing torch)
import os
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

# =============================================================================
# Imports
# =============================================================================

import torch
import json
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime
import logging
from typing import List, Optional, Tuple, Dict, Any, Union
from torch.utils.data import Dataset, DataLoader
from spatialformer.model import Spaformer
import pickle
import anndata as ad
import scipy.sparse as sp
import pytorch_lightning as pl

# =============================================================================
# Global Configuration
# =============================================================================

# Device selection: use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Timestamp for logging and file naming
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Logger setup
logger = logging.getLogger("Spatialformer")

# =============================================================================
# Utility Functions
# =============================================================================

def get_file_path(path: str, filename: str) -> str:
    """
    Construct absolute file path relative to the project root directory.
    
    Args:
        path: Subdirectory within the project (e.g., "config", "tokenizer")
        filename: Name of the file to locate
        
    Returns:
        Absolute path to the specified file
        
    Example:
        >>> get_file_path("config", "config.json")
        '/path/to/project/config/config.json'
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, path, filename)


def manual_train_fm(config: Dict[str, Any]) -> Spaformer:
    """
    Initialize a Spaformer model with the specified configuration.
    
    This function creates a fresh model instance (not trained) that can
    later be loaded with pretrained weights.
    
    Args:
        config: Dictionary containing model hyperparameters:
            - dim_model: Transformer hidden dimension
            - nheads: Number of attention heads
            - nlayers: Number of transformer layers
            - dropout: Dropout probability
            - masking_p: Masking probability for MLM
            - n_tokens: Number of gene tokens in vocabulary
            - n_atokens: Number of auxiliary tokens (CLS, SEP, etc.)
            - warmup: Learning rate warmup steps
            - use_flash_attn: Whether to use Flash Attention
            - lr: Learning rate
            - max_epochs: Maximum training epochs
            - mask_way: Masking strategy
            
    Returns:
        Initialized Spaformer model (untrained)
    """
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    model = Spaformer(
        dim_model=config['dim_model'], 
        nheads=config['nheads'], 
        nlayers=config['nlayers'],
        dropout=config['dropout'],
        masking_p=config['masking_p'], 
        n_tokens=config['n_tokens'],
        n_atokens=config['n_atokens'],
        warmup=config['warmup'],
        use_flash_attn=config["use_flash_attn"],
        lr=config['lr'],
        max_epochs=config['max_epochs'],
        mask_way=config['mask_way'], 
        outer_config=config
    )
    
    return model


def valid_mean_embedding(
    attn_mask_array: np.ndarray, 
    embeddings_array: np.ndarray
) -> np.ndarray:
    """
    Compute mean embedding over valid (non-padded) gene positions only.
    
    This function excludes:
    - The first 5 positions (auxiliary tokens: CLS, condition, tissue, etc.)
    - All padding positions (where attention mask is 0)
    
    Args:
        attn_mask_array: Attention mask of shape (batch, seq_len)
                        1 indicates valid position, 0 indicates padding
        embeddings_array: Token embeddings of shape (batch, seq_len, embed_dim)
        
    Returns:
        Mean embeddings of shape (batch, embed_dim), averaged only over
        valid gene positions
        
    Example:
        For a cell with 100 expressed genes and 400 padding tokens:
        - Skip positions 0-4 (auxiliary tokens)
        - Average positions 5-104 (gene embeddings)
        - Ignore positions 105-499 (padding)
    """
    # Remove auxiliary token positions (first 5 tokens)
    embedding_raw = embeddings_array[:, 5:, :]      # (batch, new_seq_len, embed_dim)
    attn_mask_raw = attn_mask_array[:, 5:]          # (batch, new_seq_len)
    
    # Create boolean mask for valid positions, add dimension for broadcasting
    mask = (attn_mask_raw != 0)[..., np.newaxis]    # (batch, new_seq_len, 1)
    
    # Count valid positions per sequence for proper averaging
    length_num = (attn_mask_raw != 0).sum(axis=1, keepdims=True)  # (batch, 1)
    
    # Mask embeddings (zero out padding) and sum across sequence dimension
    embedding_val = (embedding_raw * mask).sum(axis=1)  # (batch, embed_dim)
    
    # Compute mean by dividing by number of valid positions
    cell_embedding = embedding_val / length_num
    
    return cell_embedding

def prepare_extended_checkpoint(model, ckpt_path, old_size=1950, new_size=6065):
        """
        Extend tensors and rebuild optimizer param_groups for model architecture changes.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        model.cpu()
        model_state = model.state_dict()
        
        # Count actual parameters (not buffers)
        total_params = sum(1 for _ in model.parameters())
        
        logging.info(f"[INFO] Model has {total_params} parameters, {len(model_state)} state_dict keys")
        
        # ============================================
        # 1. Fix state_dict
        # ============================================
        logging.info("[STATE_DICT] Fixing...")
        new_state_dict = {}
        
        for key, new_param in model_state.items():
            new_param = new_param.cpu().clone()
            
            if key in ckpt['state_dict']:
                old_param = ckpt['state_dict'][key].cpu().clone()
                
                if old_param.shape == new_param.shape:
                    new_state_dict[key] = old_param
                elif old_param.shape[0] == old_size and new_param.shape[0] == new_size:
                    logging.info(f"  [EXTEND] {key}: {old_param.shape} -> {new_param.shape}")
                    if old_param.dim() == 2:
                        new_param[:old_size, :] = old_param
                    elif old_param.dim() == 1:
                        new_param[:old_size] = old_param
                    new_state_dict[key] = new_param
                else:
                    logging.info(f"  [SHAPE MISMATCH] {key}: {old_param.shape} -> {new_param.shape}")
                    new_state_dict[key] = new_param
            else:
                logging.info(f"  [NEW] {key}")
                new_state_dict[key] = new_param
        
        # Log deleted keys
        for key in ckpt['state_dict']:
            if key not in model_state:
                logging.info(f"  [DELETED] {key}")
        
        ckpt['state_dict'] = new_state_dict
        
        # ============================================
        # 2. Rebuild optimizer_states completely
        # ============================================
        logging.info(f"\n[OPTIMIZER] Rebuilding for {total_params} parameters...")
        
        # Extract old settings if available
        old_settings = {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0,
            'amsgrad': False,
            'initial_lr': 0.001,
        }
        
        if 'optimizer_states' in ckpt and ckpt['optimizer_states']:
            opt_state = ckpt['optimizer_states'][0]
            if 'param_groups' in opt_state and opt_state['param_groups']:
                for k, v in opt_state['param_groups'][0].items():
                    if k != 'params':
                        old_settings[k] = v
        
        # Rebuild param_groups with correct count
        new_param_group = old_settings.copy()
        new_param_group['params'] = list(range(total_params))
        
        ckpt['optimizer_states'] = [
            {
                'state': {},
                'param_groups': [new_param_group]
            }
        ]
        logging.info(f"  [OK] Rebuilt with {total_params} params")
        
        # ============================================
        # 3. Reset lr_schedulers
        # ============================================
        if 'lr_schedulers' in ckpt:
            logging.info("[SCHEDULER] Resetting")
            ckpt['lr_schedulers'] = []
        
        # ============================================
        # 4. Save
        # ============================================
        logging.info(f"\n[INFO] Epoch: {ckpt.get('epoch', 0)}, Step: {ckpt.get('global_step', 0)}")
        
        new_ckpt_path = ckpt_path.replace('.ckpt', '_extended.ckpt')
        torch.save(ckpt, new_ckpt_path)
        logging.info(f"[SAVED] {new_ckpt_path}")
        
        return new_ckpt_path


# =============================================================================
# Main Embedding Function
# =============================================================================

def embed_data(
    adata,
    tissue: str, 
    condition: str,
    method: str,
    model_ckp_path: str, 
    batch_size: int,
    config_path: str = get_file_path("config", "_config_train_large_pair.json"),
    token_path: str = get_file_path("tokenizer", "tokenv5.json"),
    mode: str = "single",
    only_loader: bool = False,
    threshold: float = 0.8,
    left_cell: Optional[List[str]] = None,
    right_cell: Optional[List[str]] = None,
    pair_label: Optional[List[int]] = None,
    num_workers: int = 0,
    reveal_name: bool = False,
    gene_median_path: Optional[str] = None,
    max_len: int = None,
    resume_before_5k: bool = None,
    reverse_check: bool = None,
):
    """
    Generate Spatialformer embeddings for cells in an AnnData object.
    
    This is the main inference function that:
    1. Loads model configuration and pretrained weights
    2. Tokenizes gene expression data
    3. Generates embeddings through the transformer
    4. Optionally predicts gene co-occurrence or cell pair relationships
    
    Args:
        adata: AnnData object containing gene expression matrix
               - adata.X: Expression matrix (cells x genes)
               - adata.var["gene_name"]: Gene names matching tokenizer vocabulary
               
        tissue: Tissue type token (e.g., "Colon", "Brain", "Liver")
                Must exist in tokenizer vocabulary
                
        condition: Condition token (e.g., "Disease", "Normal")
                   Must exist in tokenizer vocabulary
                   
        method: Embedding extraction method
                - "cls": Use CLS token embedding as cell representation
                - "gene": Use mean of gene token embeddings
                
        model_ckp_path: Path to pretrained model checkpoint (.ckpt file)
        
        batch_size: Number of cells to process per batch
        
        config_path: Path to model configuration JSON file
        
        token_path: Path to tokenizer vocabulary JSON file
        
        mode: Processing mode
              - "single": Process individual cells
              - "pair": Process cell pairs for relationship prediction
              
        only_loader: If True, return DataLoader without running inference
                     (useful for debugging or custom processing)
                     
        threshold: Probability threshold for gene co-occurrence prediction
                   Pairs with probability > threshold are reported
                   
        left_cell: List of left cell IDs for pair mode
        
        right_cell: List of right cell IDs for pair mode
        
        pair_label: Ground truth labels for cell pairs (0 or 1)
        
        num_workers: Number of DataLoader worker processes
        
        reveal_name: If True, decode gene IDs back to gene names for
                     co-occurrence predictions
                     
        gene_median_path: Path to gene median expression file for normalization
        max_len: The max length of the input sequence
        resume_before_5k: Bool, load the ckp before 5k
        reverse_check: Bool, whether to check reverse pairs for positive prediction. The 5k panel model doesn't need this because it was pretrained via anchor based solution.
    Returns:
        For single mode:
            AnnData object with embeddings stored in adata.obsm["X_SpaF"]
            If reveal_name=True, also adds adata.obs["Gene_Pairs"]
            
        For pair mode:
            Tuple of (embeddings, probabilities) where:
            - embeddings: numpy array of CLS embeddings
            - probabilities: list of pair prediction probabilities
    
    Example:
        >>> # Single cell embedding
        >>> adata = embed_data(
        ...     adata=my_adata,
        ...     tissue="Colon",
        ...     condition="Disease", 
        ...     method="cls",
        ...     model_ckp_path="model.ckpt",
        ...     batch_size=32,
        ...     mode="single"
        ... )
        >>> cell_embeddings = adata.obsm["X_SpaF"]  # (n_cells, dim_model)
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Load Configuration
    # -------------------------------------------------------------------------
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize and Load Model
    # -------------------------------------------------------------------------
    logger.info("Loading the SpatialFormer model...")
    
    # Create model architecture
    model = manual_train_fm(config=config)
    
    # Load pretrained weights

    if resume_before_5k:
        # Prepare checkpoint with extended embeddings
        logger.info(f"Loading the ckp before the 5k panel")
        model_ckp_path = prepare_extended_checkpoint(model, model_ckp_path)


    checkpoint = torch.load(model_ckp_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    
    # Set to evaluation mode (disables dropout, etc.)
    logger.info("Setting the model to evaluation mode...")
    model.eval()
    
    # Move model to appropriate device
    logger.info(f"Model mapped to device: {device}")
    model.to(device)

    # -------------------------------------------------------------------------
    # Step 3: Process Based on Mode
    # -------------------------------------------------------------------------
    logger.info("Encoding data into batches...")
    
    if mode == "single":
        model.input_type = "single"
        return _process_single_mode(
            adata=adata,
            model=model,
            token_path=token_path,
            tissue=tissue,
            condition=condition,
            method=method,
            batch_size=batch_size,
            num_workers=num_workers,
            threshold=threshold,
            reveal_name_flag=reveal_name,
            gene_median_path=gene_median_path,
            max_len=max_len
        )
        
    elif mode == "pair":
        model.input_type = "pair"
        return _process_pair_mode(
            adata=adata,
            model=model,
            token_path=token_path,
            tissue=tissue,
            condition=condition,
            batch_size=batch_size,
            num_workers=num_workers,
            left_cell=left_cell,
            right_cell=right_cell,
            pair_label=pair_label,
            only_loader=only_loader,
            gene_median_path=gene_median_path,
            reverse_check=reverse_check,
            max_len=max_len
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'single' or 'pair'.")


def _process_single_mode(
    adata,
    model: Spaformer,
    token_path: str,
    tissue: str,
    condition: str,
    method: str,
    batch_size: int,
    num_workers: int,
    threshold: float,
    reveal_name_flag: bool,
    gene_median_path: str,
    max_len: int
):
    """
    Process single cells to generate embeddings.
    
    Internal function called by embed_data when mode="single".
    """
    # Initialize tokenizer and dataset
    tokenizer = GeneTokenizer(
        token_path, 
        mode="single", 
        tissue=tissue, 
        condition=condition, 
        gene_median_path=gene_median_path
    )
    dataset = GeneExpressionDataset(adata, tokenizer, max_len)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        prefetch_factor=3,
        pin_memory=True,
        persistent_workers=num_workers
    )
    
    # Storage for results
    all_embeddings = []
    all_pairs = []
    
    # Inference loop (no gradient computation needed)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(
            dataloader, 
            total=len(dataloader), 
            desc="Generating embeddings"
        )):
            # Get model outputs
            # Returns: (hidden_states, co_occurrence_predictions)
            last_hidden_repr, co_adj_prob = model.get_embeddings(
                batch, 
                layers=[-1],           # Get last layer only
                pair_prediction=False,  # Not doing pair prediction
                co_prediction=True      # Get gene co-occurrence predictions
            )
            
            # Extract embeddings based on method
            if method == "cls":
                # Use CLS token (position 0) as cell representation
                embeddings = last_hidden_repr[0][:, 0].detach().cpu()
                
            elif method == "gene":
                # Use mean of gene embeddings (excluding special tokens)
                attn_mask = batch["attention_mask"]
                attn_mask_array = attn_mask.detach().cpu().numpy()
                embeddings_array = last_hidden_repr[0].detach().cpu().numpy()
                embeddings = valid_mean_embedding(attn_mask_array, embeddings_array)
                
            else:
                raise ValueError(
                    f"Unsupported method: '{method}'. Use 'cls' or 'gene'."
                )
            
            # Optionally decode gene pairs above threshold
            if reveal_name_flag:
                batch_pairs = reveal_gene_pairs(
                    tokenizer=tokenizer,
                    co_adj_prob=co_adj_prob,
                    threshold=threshold,
                    batch=batch
                )
                all_pairs.extend(batch_pairs)
            
            all_embeddings.append(embeddings)
            
            # Log memory usage periodically
            if batch_idx % 100 == 0:
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                reserved_gb = torch.cuda.memory_reserved() / 1e9
                logger.debug(
                    f"Batch {batch_idx}: Allocated={allocated_gb:.2f}GB, "
                    f"Reserved={reserved_gb:.2f}GB"
                )
    
    # Combine all batch embeddings
    if isinstance(all_embeddings[0], np.ndarray):
        combined_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        combined_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    
    # Store results in AnnData object
    adata.obsm["X_SpaF"] = combined_embeddings
    
    if reveal_name_flag:
        adata.obs["Gene_Pairs"] = all_pairs
    
    return adata


def _process_pair_mode(
    adata,
    model: Spaformer,
    token_path: str,
    tissue: str,
    condition: str,
    batch_size: int,
    num_workers: int,
    left_cell: List[str],
    right_cell: List[str],
    pair_label: List[int],
    only_loader: bool,
    gene_median_path: str,
    reverse_check: bool,
    max_len: int
):
    """
    Process cell pairs to predict relationships.
    
    Internal function called by embed_data when mode="pair".
    
    For each cell pair, the model predicts a probability that the
    cells are spatially related (e.g., neighbors in tissue).
    
    To increase prediction robustness, we also compute the reverse
    prediction (swap left and right cells) and combine results.
    """
    # Initialize tokenizer and dataset
    tokenizer = GeneTokenizer(
        token_path, 
        mode="pair", 
        tissue=tissue, 
        condition=condition, 
        gene_median_path=gene_median_path
    )

    dataset = GeneExpressionPairDataset(
        adata, left_cell, right_cell, pair_label, tokenizer, max_len
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=True, 
        prefetch_factor=3, 
        persistent_workers=True
    )
    # dataloader = DataLoader(
    #     dataset, 
    #     batch_size=batch_size, 
    #     collate_fn=collate_fn, 
    #     num_workers=0, 
    #     pin_memory=True
    # )
    
    # Option to return just the dataloader
    if only_loader:
        return dataloader
    
    # Storage for results
    all_embeddings = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing pairs")):
            # Forward pass: get embeddings and pair predictions
            last_hidden_repr1, probabilities = model.get_embeddings(
                batch, 
                layers=[-1], 
                pair_prediction=True, 
                co_prediction=False
            )
            # Extract CLS embeddings
            cls_embeddings = last_hidden_repr1[0][:, 0]
            all_embeddings.append(cls_embeddings.detach().cpu())
            # import pdb; pdb.set_trace()
            # Compute reverse predictions (swap cell order)
            # This helps verify prediction consistency
            if reverse_check:
                reversed_batch = rearrange_sentences(batch)
                last_hidden_repr2, reverse_probabilities = model.get_embeddings(
                    reversed_batch, 
                    layers=[-1], 
                    pair_prediction=True, 
                    co_prediction=False
                )
                # import pdb; pdb.set_trace()
                # Combine forward and reverse predictions
                confirmed_probs = [
                    process_bidirectional_predictions(fwd, rev) 
                    for fwd, rev in zip(probabilities, reverse_probabilities)
                ]
                confirmed_probs = torch.stack(confirmed_probs).detach().cpu().numpy()
                all_probabilities.append(confirmed_probs)
            else:
                all_probabilities.append(probabilities.detach().cpu())
    # import pdb; pdb.set_trace()
    # Combine results
    combined_embeddings = torch.cat(all_embeddings, dim=0).detach().cpu().numpy()

    return combined_embeddings, all_probabilities


# =============================================================================
# Pair Prediction Utilities
# =============================================================================

def process_bidirectional_predictions(
    forward_pred: torch.Tensor, 
    reverse_pred: torch.Tensor
) -> torch.Tensor:
    """
    Combine forward and reverse pair predictions for robustness.
    
    Given predictions for (cell_A, cell_B) and (cell_B, cell_A),
    select the more confident prediction.
    
    A prediction is "aligned" if P(positive) > P(negative), i.e., tensor[1] > tensor[0].
    
    Logic:
    - If both predictions agree (both aligned or both not), randomly choose one
    - If they disagree, choose the one with higher confidence for its prediction
    
    Args:
        forward_pred: Probability tensor [P(neg), P(pos)] for (A, B)
        reverse_pred: Probability tensor [P(neg), P(pos)] for (B, A)
        
    Returns:
        Selected prediction tensor
        
    Raises:
        ValueError: If inputs are not 2-element tensors
    """
    # Input validation
    if not isinstance(forward_pred, torch.Tensor) or not isinstance(reverse_pred, torch.Tensor):
        raise ValueError("Both inputs must be PyTorch tensors.")
    
    if forward_pred.numel() != 2 or reverse_pred.numel() != 2:
        raise ValueError("Both tensors must contain exactly two elements [P(neg), P(pos)].")
    
    # Check alignment: is P(positive) > P(negative)?
    forward_aligned = forward_pred[1] > forward_pred[0]
    reverse_aligned = reverse_pred[1] > reverse_pred[0]
    
    # Selection logic
    if forward_aligned and reverse_aligned:
        # Both predict positive - randomly choose one
        chosen = random.choice([forward_pred, reverse_pred])
    elif not forward_aligned and not reverse_aligned:
        # Both predict negative - randomly choose one
        chosen = random.choice([forward_pred, reverse_pred])
    elif forward_pred[0] > forward_pred[1]:
        # Forward predicts negative with higher confidence
        chosen = forward_pred
    else:
        # Reverse predicts negative with higher confidence
        chosen = reverse_pred
    
    return chosen


def rearrange_sentences(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Swap the order of cell pairs in a batch for bidirectional prediction.
    
    Given input sequence: [CLS] [prefix1] [genes1] [SEP] [prefix2] [genes2] [SEP] [PAD...]
    Output sequence:      [CLS] [prefix2] [genes2] [SEP] [prefix1] [genes1] [SEP] [PAD...]
    
    This allows the model to see (cell_B, cell_A) after seeing (cell_A, cell_B),
    which helps validate prediction consistency.
    
    Args:
        batch: Dictionary containing:
            - indices: Token IDs (batch, seq_len)
            - attention_mask: Attention mask (batch, seq_len)
            - token_type_ids: Token type IDs (batch, seq_len)
            
    Returns:
        New batch with swapped cell order
    """
    batch_size = len(batch['indices'])
    max_length = batch['indices'].shape[1]  # Fixed maximum sequence length
    sep_token = 1949  # Token ID for separator
    # Initialize output tensors
    new_indices = torch.full((batch_size, max_length), 0, dtype=torch.int)
    new_attention_mask = torch.full((batch_size, max_length), 0, dtype=torch.int)
    new_token_types = torch.full((batch_size, max_length), 0, dtype=torch.int)
    new_sequence_length = torch.full((batch_size, 2), 0, dtype=torch.int)
    
    indices = batch['indices']
    attention_masks = batch['attention_mask']
    token_types = batch['token_type_ids']
    sequence_lengths = batch['sequence_length']
    
    for i, (indice, attn_mask, token_type, sequence_length) in enumerate(zip(indices, attention_masks, token_types, sequence_lengths)):
        # Find separator positions
        sep_positions = (indice == sep_token).nonzero(as_tuple=True)[0]
        
        if len(sep_positions) < 2:
            logger.warning(f"Warning: Expected 2 SEP tokens, found {len(sep_positions)} in sample {i}")
            continue
        
        mid_idx = sep_positions[0]  # First SEP (end of cell 1)
        end_idx = sep_positions[1]  # Second SEP (end of cell 2)
        # Split sequence into components
        cls_token = indice[:1]                    # [CLS]
        sentence1 = indice[1:mid_idx + 1]         # [prefix1] [genes1] [SEP]
        sentence2 = indice[mid_idx + 1:end_idx + 1]  # [prefix2] [genes2] [SEP]

        # Combine in reversed order: [CLS] + sentence2 + sentence1
        combined = torch.cat((cls_token, sentence2, sentence1))

        length1 = sequence_length[0]
        length2 = sequence_length[1]
        new_length =  torch.tensor([[length2 + 1, length1 - 1]]) # + cls; -cls
        
        # Pad to max_length
        pad_length = max_length - combined.size(0)
        new_sequence = torch.cat((combined, torch.zeros(pad_length, dtype=torch.int)))
        new_indices[i, :] = new_sequence[:max_length]
        # Attention mask stays the same (same valid length)
        new_attention_mask[i, :] = attn_mask
        
        # Update token type IDs (swap types 1 and 2)
        left_types = torch.full((len(cls_token) + len(sentence2),), 1)  # Now sentence2 is first
        right_types = torch.full((len(sentence1),), 2)                  # Sentence1 is second
        pad_types = torch.full((pad_length,), 0)
        new_token_type = torch.cat((left_types, right_types, pad_types))
        new_token_types[i, :] = new_token_type[:max_length]

        new_sequence_length[i, :] = new_length
        # import pdb; pdb.set_trace()
    return {
        "indices": new_indices,
        "attention_mask": new_attention_mask.bool(),
        "token_type_ids": new_token_types,
        "sequence_length": new_sequence_length
    }


# =============================================================================
# Gene Co-occurrence Utilities
# =============================================================================

def reveal_gene_pairs(
    tokenizer: 'GeneTokenizer',
    co_adj_prob: List[List[torch.Tensor]],
    threshold: float,
    batch: Dict[str, torch.Tensor]
) -> List[List[List[str]]]:
    """
    Decode gene pair predictions to gene names.
    
    For each cell, find gene pairs with co-occurrence probability above threshold
    and convert token IDs back to gene names.
    
    Args:
        tokenizer: GeneTokenizer instance with token_to_id mapping
        co_adj_prob: Co-occurrence predictions from model
                    List of [block1, block2] for each sample
        threshold: Probability threshold for reporting pairs
        batch: Batch containing token indices
        
    Returns:
        List of gene pair lists for each cell in batch
        Each gene pair is [gene_name_A, gene_name_B]
    """
    # Create reverse lookup: ID -> gene name
    id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
    
    all_pairs = []
    
    for sample_idx, indice in enumerate(batch["indices"]):
        length = len(indice)
        
        # Get co-occurrence matrix for this sample
        # co_adj_prob[sample_idx] contains [block1, block2]
        # For single mode, we mainly care about block1
        if isinstance(co_adj_prob, list) and len(co_adj_prob) > sample_idx:
            prob_matrix = co_adj_prob[sample_idx][0]  # First block
            if isinstance(prob_matrix, list) and len(prob_matrix) == 0:
                all_pairs.append([])
                continue
        else:
            all_pairs.append([])
            continue
        
        # Apply sigmoid to convert logits to probabilities
        prob_matrix = torch.sigmoid(prob_matrix)
        
        # Get upper triangle indices (avoid duplicates and self-pairs)
        upper_indices = torch.triu_indices(
            prob_matrix.size(0), 
            prob_matrix.size(1), 
            offset=1, 
            device=prob_matrix.device
        )
        
        # Filter pairs above threshold
        probs = prob_matrix[upper_indices[0], upper_indices[1]]
        above_threshold = probs > threshold
        filtered_i = upper_indices[0][above_threshold]
        filtered_j = upper_indices[1][above_threshold]
        
        # Convert to gene names
        sample_pairs = []
        # Offset by 5 to account for auxiliary tokens at start
        aux_offset = 5
        
        for idx_i, idx_j in zip(filtered_i.tolist(), filtered_j.tolist()):
            # Map back to original token positions
            token_pos_i = idx_i + aux_offset
            token_pos_j = idx_j + aux_offset
            
            # Skip if positions are out of range
            if token_pos_i >= len(indice) or token_pos_j >= len(indice):
                continue
            
            token_id_i = indice[token_pos_i].item()
            token_id_j = indice[token_pos_j].item()
            
            # Skip special tokens (IDs < 10 are typically special tokens)
            if token_id_i < 10 or token_id_j < 10:
                continue
            
            # Get gene names
            gene_a = id_to_token.get(token_id_i, f"UNK_{token_id_i}")
            gene_b = id_to_token.get(token_id_j, f"UNK_{token_id_j}")
            
            sample_pairs.append([gene_a, gene_b])
        
        all_pairs.append(sample_pairs)
    
    return all_pairs


# =============================================================================
# Tokenizer
# =============================================================================

class GeneTokenizer:
    """
    Tokenizer for converting gene expression vectors to ranked token sequences.
    
    The tokenizer:
    1. Normalizes expression by gene-specific median values
    2. Ranks genes by normalized expression (descending)
    3. Converts gene names to token IDs
    
    Token Vocabulary Structure:
    - 0: PAD token
    - 1: CLS token  
    - 2: MASK token
    - 3-9: Other special tokens
    - 10+: Gene tokens
    
    Attributes:
        token_vocab: Full vocabulary mapping
        token_to_id: Gene name to token ID mapping
        mode: "single" or "pair" processing mode
        tissue_id: Token ID for tissue type
        condition_id: Token ID for condition
        genes: Gene names for current dataset (set dynamically)
        gene_median_dict: Gene-specific median expression values
    """
    
    def __init__(
        self, 
        token_file: str, 
        mode: str = "single", 
        tissue: str = None, 
        condition: str = None, 
        gene_median_path: str = None
    ):
        """
        Initialize the tokenizer.
        
        Args:
            token_file: Path to JSON file containing token vocabulary
            mode: Processing mode ("single" or "pair")
            tissue: Tissue type (must exist in vocabulary)
            condition: Condition type (must exist in vocabulary)
            gene_median_path: Path to pickle file with gene median expression values
        """
        # Load vocabulary
        with open(token_file, 'r') as f:
            self.token_vocab = json.load(f)
        
        self.token_to_id = {gene: idx for gene, idx in self.token_vocab.items()}
        self.mode = mode
        
        # Get tissue and condition token IDs
        if tissue not in self.token_to_id:
            raise ValueError(f"Tissue '{tissue}' not found in vocabulary")
        if condition not in self.token_to_id:
            raise ValueError(f"Condition '{condition}' not found in vocabulary")
            
        self.tissue_id = self.token_to_id[tissue]
        self.condition_id = self.token_to_id[condition]
        
        # Placeholder for gene names (set per dataset)
        self.genes = None
        
        # Load gene median expression for normalization
        if gene_median_path:
            with open(gene_median_path, "rb") as f:
                self.gene_median_dict = pickle.load(f)
        else:
            self.gene_median_dict = {}
    
    def single_cell(self, expression_vector: np.ndarray) -> List[int]:
        """
        Convert a single cell's expression vector to ranked token sequence.
        
        Processing steps:
        1. Normalize by gene-specific median expression
        2. Identify non-zero expression genes
        3. Sort genes by normalized expression (descending)
        4. Convert gene names to token IDs
        
        Args:
            expression_vector: Raw expression values for all genes (1D array)
            
        Returns:
            List of token IDs for expressed genes, ordered by expression level
        """
        # Step 1: Normalize by gene median expression
        # This reduces technical bias from highly abundant genes
        gene_medians = np.array([
            self.gene_median_dict.get(gene, 1.0) 
            for gene in self.genes
        ])
        normalized_expression = expression_vector / gene_medians
        
        # Step 2: Identify zero-expression genes
        zero_indices = np.where(expression_vector == 0)[0]
        
        # Step 3: Sort genes by normalized expression (descending)
        sorted_indices = np.argsort(-normalized_expression)
        # Remove zero-expression genes
        sorted_nonzero_indices = sorted_indices[~np.isin(sorted_indices, zero_indices)]
        
        # Step 4: Convert to tokens
        sorted_genes = self.genes[sorted_nonzero_indices]
        gene_tokens = []
        
        for gene_name in sorted_genes:
            if gene_name in self.token_to_id:
                gene_tokens.append(self.token_to_id[gene_name])
            # Genes not in vocabulary are silently skipped
        
        return gene_tokens
    
    def encode(
        self, 
        expression_vector1: np.ndarray, 
        expression_vector2: np.ndarray = None
    ) -> Tuple[Union[List[int], Tuple[List[int], List[int]]], List[int], List[int]]:
        """
        Encode expression vector(s) into token sequences.
        
        Args:
            expression_vector1: Expression vector for first cell
            expression_vector2: Expression vector for second cell (pair mode only)
            
        Returns:
            Tuple of (gene_tokens, prefix_tokens, end_tokens):
            - gene_tokens: Token IDs for genes (single list or tuple of two lists)
            - prefix_tokens: Auxiliary tokens at sequence start [CLS, condition, tissue, ...]
            - end_tokens: Separator token(s) [SEP]
        """
        # Build prefix and suffix tokens
        cls_token = self.token_to_id["<CLS>"]
        sep_token = self.token_to_id["<SEP>"]
        
        # Prefix: [CLS, condition, tissue, special1, special2]
        # Tokens 3 and 6 appear to be additional special tokens
        prefix = [cls_token, self.condition_id, self.tissue_id, 3, 6]
        end = [sep_token]
        
        if self.mode == "single":
            gene_tokens = self.single_cell(expression_vector1)
            return gene_tokens, prefix, end
            
        elif self.mode == "pair":
            gene_tokens1 = self.single_cell(expression_vector1)
            gene_tokens2 = self.single_cell(expression_vector2)
            return (gene_tokens1, gene_tokens2), prefix, end
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# =============================================================================
# Datasets
# =============================================================================

class GeneExpressionDataset(Dataset):
    """
    PyTorch Dataset for single cell gene expression data.
    
    Wraps an AnnData object and provides tokenized gene expression
    sequences for each cell.
    
    Attributes:
        tokenizer: GeneTokenizer instance
        mode: Processing mode (from tokenizer)
        expression_data: Dense expression matrix (cells x genes)
        genes: Gene names array
    """
    
    def __init__(self, adata, tokenizer: GeneTokenizer, max_len: None):
        """
        Initialize dataset from AnnData object.
        
        Args:
            adata: AnnData object with:
                - X: Expression matrix (cells x genes)
                - var["gene_name"]: Gene names
            tokenizer: Configured GeneTokenizer instance
            max_len: The max length of the sequence
        """
        self.tokenizer = tokenizer
        self.mode = self.tokenizer.mode
        self.max_len = max_len
        
        # Convert sparse matrix to dense if needed
        if sp.issparse(adata.X):
            self.expression_data = adata.X.toarray()
        else:
            self.expression_data = np.asarray(adata.X)
        
        self.genes = adata.var["gene_name"].to_numpy()
    
    def __len__(self) -> int:
        """Return number of cells in dataset."""
        return self.expression_data.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Get tokenized sequence for a single cell.
        
        Args:
            idx: Cell index
            
        Returns:
            Tuple of (gene_tokens, prefix_tokens, end_tokens)
        """
        expression_vector = self.expression_data[idx]
        
        # Set genes for tokenizer (needed for encoding)
        self.tokenizer.genes = self.genes
        
        tokens, prefix, end = self.tokenizer.encode(expression_vector)
        if self.max_len is not None:
            tokens = tokens[:self.max_len]
        return tokens, prefix, end


class GeneExpressionPairDataset(Dataset):
    """
    PyTorch Dataset for cell pair relationship prediction.
    
    Given pairs of cells, provides tokenized sequences for both cells
    along with their relationship labels.
    
    Attributes:
        tokenizer: GeneTokenizer instance
        mode: Processing mode
        pair_label: Binary labels for cell pairs
        left_cells: List of left cell identifiers
        right_cells: List of right cell identifiers
        left_expression_data: Expression matrix for left cells
        right_expression_data: Expression matrix for right cells
        genes: Gene names array
    """
    
    def __init__(
        self, 
        adata, 
        left_cells: List[str], 
        right_cells: List[str], 
        pair_label: List[int], 
        tokenizer: GeneTokenizer,
        max_len: int
    ):
        """
        Initialize dataset for cell pair processing.
        
        Args:
            adata: AnnData object with expression data
            left_cells: List of cell IDs for left side of pairs
            right_cells: List of cell IDs for right side of pairs  
            pair_label: Binary labels (1=related, 0=not related)
            tokenizer: Configured GeneTokenizer instance
            max_len: the max length for each sequence
        """
        self.tokenizer = tokenizer
        self.mode = self.tokenizer.mode
        self.pair_label = pair_label
        self.left_cells = left_cells
        self.right_cells = right_cells
        self.max_len = max_len
        # Map cell names to indices
        all_cell_names = list(adata.obs.index)
        self.left_indices = [all_cell_names.index(cell) for cell in left_cells]
        self.right_indices = [all_cell_names.index(cell) for cell in right_cells]
        
        # Extract expression data for specified cells
        try:
            # Sparse matrix case
            self.left_expression_data = np.array(
                adata.X[self.left_indices, :].todense()
            )
            self.right_expression_data = np.array(
                adata.X[self.right_indices, :].todense()
            )
            # import pdb; pdb.set_trace()
        except AttributeError:
            # Dense matrix case
            self.left_expression_data = np.array(adata.X[self.left_indices, :])
            self.right_expression_data = np.array(adata.X[self.right_indices, :])
        
        self.genes = adata.var["gene_name"].to_numpy()
    
    def __len__(self) -> int:
        """Return number of cell pairs."""
        return self.left_expression_data.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get tokenized sequences and metadata for a cell pair.
        
        Args:
            idx: Pair index
            
        Returns:
            Tuple of:
            - (gene_tokens1, gene_tokens2): Token sequences for both cells
            - prefix: Auxiliary prefix tokens
            - end: Separator tokens
            - left_index: Index of left cell in original data
            - right_index: Index of right cell in original data
            - pair_label: Binary relationship label
            - left_cell_id: Left cell identifier
            - right_cell_id: Right cell identifier
        """
        left_expression = self.left_expression_data[idx]
        right_expression = self.right_expression_data[idx]
        
        self.tokenizer.genes = self.genes
        tokens, prefix, end = self.tokenizer.encode(left_expression, right_expression)
        if self.max_len:
            tokens = (tokens[0][:self.max_len], tokens[1][:self.max_len])

        if self.pair_label is None:
            return (
                tokens,                      # (gene_tokens1, gene_tokens2)
                prefix,                      # [CLS, condition, tissue, ...]
                end,                         # [SEP]
                self.left_indices[idx],      # Original data index
                self.right_indices[idx],     # Original data index
                None,
                self.left_cells[idx],        # Cell ID string
                self.right_cells[idx]        # Cell ID string
                )
        else:
            return (
                tokens,                      # (gene_tokens1, gene_tokens2)
                prefix,                      # [CLS, condition, tissue, ...]
                end,                         # [SEP]
                self.left_indices[idx],      # Original data index
                self.right_indices[idx],     # Original data index
                self.pair_label[idx],        # 0 or 1
                self.left_cells[idx],        # Cell ID string
                self.right_cells[idx]        # Cell ID string
            )


# =============================================================================
# Collate Function
# =============================================================================
def collate_fn(batch: List[Tuple[Any, ...]]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function for batching sequences with padding and attention masks.
    
    Handles two modes:
        1. Single mode: Each sample contains a single sequence
        2. Pair mode: Each sample contains a pair of sequences for comparison
    
    Args:
        batch: A list of tuples containing tokenized sequences and metadata.
            - Single mode format: (tokens, prefix_tokens, end_tokens)
            - Pair mode format: ((tokens1, tokens2), prefix_tokens, end_tokens,
                                 left_index, right_index, pair_label, left_cell, right_cell)
    
    Returns:
        Dictionary containing:
            - indices: Padded token indices [batch_size, seq_length]
            - attention_mask: Boolean attention mask [batch_size, seq_length]
            - token_type_ids: Segment IDs [batch_size, seq_length]
            - sequence_length: (single mode only) Sequence lengths [batch_size, 2]
            - left_index, right_index, pair_label, left_cell_ids, right_cell_ids:
              (pair mode only) Additional metadata for pair processing
    """
    # Extract common elements from batch
    indices: List[Any] = [item[0] for item in batch]
    prefix_tokens: List[int] = batch[0][1]
    end_tokens: List[int] = batch[0][2]
    auxiliary_length: int = len(prefix_tokens) + len(end_tokens)
    
    
    
    # Initialize containers
    padded_indices: List[List[int]] = []
    attention_masks: List[List[int]] = []
    token_type_ids: List[List[int]] = []
    sequence_lengths: List[List[int]] = []
    
    # Determine mode based on indices structure
    is_pair_mode: bool = isinstance(indices[0], (list, tuple)) and len(indices[0]) == 2
    
    if not is_pair_mode:
        # Calculate max sequence length
        max_seq_length: int = max(
            len(item[0]) + len(item[1]) + len(item[2]) for item in batch
        )
        # ==================== Single Sequence Mode ====================
        for i, tokens in enumerate(indices):
            tokens_len: int = len(tokens)
            prefix_len: int = len(batch[i][1])
            end_len: int = len(batch[i][2])
            seq_length: int = prefix_len + tokens_len + end_len
            
            # Construct sequence: [prefix] + tokens + [end] + [padding]
            if seq_length >= max_seq_length:
                padded_sequence = prefix_tokens + tokens + end_tokens
                attention_mask = [1] * len(padded_sequence)
            else:
                pad_size = max_seq_length - seq_length
                padded_sequence = prefix_tokens + tokens + end_tokens + [0] * pad_size
                attention_mask = [1] * seq_length + [0] * pad_size
            
            token_type_id = [1] * len(padded_sequence)
            
            padded_indices.append(padded_sequence)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            sequence_lengths.append([np.sum(attention_mask), 0])
        # import pdb; pdb.set_trace()
        return {
            "indices": torch.tensor(padded_indices),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
            "token_type_ids": torch.tensor(token_type_ids),
            "sequence_length": torch.tensor(sequence_lengths)
        }
    
    else:
        # ==================== Pair Sequence Mode ====================
        
        # Extract pair-specific metadata
        left_indices: List[int] = [item[3] for item in batch]
        right_indices: List[int] = [item[4] for item in batch]
        pair_labels: List[int] = [item[5] for item in batch]
        left_cells: List[Any] = [item[6] for item in batch]
        right_cells: List[Any] = [item[7] for item in batch]

        max_seq_length: int = max(
                len(item[0][0]) + len(item[0][1]) + len(item[1])*2 + len(item[2])*2 - 1 for item in batch
                ) #cls + 4 + token1 + sep + 4 + token2 + sep
        max_single_length: int = int((max_seq_length / 2) - auxiliary_length)
        exclude_cls: List[int] = prefix_tokens[1:]  # Prefix without CLS token

        for i, (token1, token2) in enumerate(indices):
            token1_len: int = len(token1)
            token2_len: int = len(token2)
            
            # Build complete sequence without truncation
            first_segment = prefix_tokens + token1 + end_tokens
            second_segment = exclude_cls + token2 + end_tokens
            padded_sequence = first_segment + second_segment
            # Calculate padding
            pad_size = max_seq_length - len(padded_sequence)
            attention_mask = [1] * len(padded_sequence) + [0] * pad_size
            padded_sequence += [0] * pad_size
            # Build token type IDs: 1 for first segment, 2 for second, 0 for padding
            token_type_id = (
                [1] * len(first_segment) +
                [2] * len(second_segment) +
                [0] * pad_size
            )
            sequence_lengths.append([len(first_segment), len(second_segment)])
            padded_indices.append(padded_sequence)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)

        if pair_labels[0] is None:
            return {
                "indices": torch.tensor(padded_indices),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
                "token_type_ids": torch.tensor(token_type_ids),
                "left_index": torch.tensor(left_indices),
                "right_index": torch.tensor(right_indices),
                "pair_label": None,
                "left_cell_ids": left_cells,
                "right_cell_ids": right_cells,
                "sequence_length": torch.tensor(sequence_lengths)
            }
        else:
            return {
                "indices": torch.tensor(padded_indices),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
                "token_type_ids": torch.tensor(token_type_ids),
                "left_index": torch.tensor(left_indices),
                "right_index": torch.tensor(right_indices),
                "pair_label": torch.tensor(pair_labels),
                "left_cell_ids": left_cells,
                "right_cell_ids": right_cells,
                "sequence_length": torch.tensor(sequence_lengths)
            }


if __name__ == "__main__":
    import scanpy as sc
    import time
    import numpy as np
    batch_size = 16
    ct_train_adata = ad.read_h5ad("/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/VUILD110_ct_train_adata.h5ad")
    ct_train_adata.var["gene_name"] = ct_train_adata.var.index
    #configuring the model
    model_ckp_path = "/scratch/project_465001820/Spatialformer/output/checkpoints/stepstep=0176000-traintrain_total_loss=-2.7789-valval_total_loss=0.0000.ckpt"
    # model_ckp_path = "/scratch/project_465001820/Spatialformer/output/checkpoints/stepstep=0192000-traintrain_total_loss=-4.1449-valval_total_loss=0.0000.ckpt"
    tissue = "Lung"
    condition = "Disease"
    method = "cls" #getting the cls token embeddings
    # import pdb; pdb.set_trace()
    ct_train_embed_adata = embed_data(adata = ct_train_adata, 
                                tissue = tissue,
                                condition = condition,
                                method = method,
                                model_ckp_path = model_ckp_path, 
                                batch_size = batch_size,
                                mode = "pair",
                                threshold = 0.7,
                                num_workers = 8,
                                gene_median_path = "/scratch/project_465001820/Spatialformer/data/gene_median.pkl",
                                left_cell = ct_train_adata.obs.index[:1000],
                                right_cell = ct_train_adata.obs.index[1000:2000],
                                reverse_check=False,
                                max_len=500
                                )
        
    #save the 
    # np.save("/scratch/project_465001820/Spatialformer/downstream/cell_types_nich_annotation/data/CRC_VisiumHD_adata_train_spatialformer.npy",embed_adata_train.obsm["X_SpaF"])
    
