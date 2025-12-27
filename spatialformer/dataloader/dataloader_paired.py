import os
import json
import pickle
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from dataclasses import dataclass, field
import gc
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from scipy.sparse import coo_matrix
from datasets import Dataset as HFDataset, load_from_disk, DatasetDict, disable_caching
from tqdm import tqdm
import faiss
import multiprocessing as mp
from multiprocessing import get_context
from functools import partial
import random
import sys
sys.path.append("/scratch/project_465001820/Spatialformer/utils")
from utils import *
sys.path.append("/scratch/project_465001820/Spatialformer/train")
from data_loader import create_dataloader
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable caching to prevent Arrow conflicts
disable_caching()


# Load tokenizer
with open('/scratch/project_465001820/Spatialformer/tokenizer/tokenv5.json', 'r') as f:
    token_dict = json.load(f)


###############################################################################
# Data Structures
###############################################################################

@dataclass
class AnchorPairIndices:
    """Stores precomputed pair indices for a single anchor cell."""
    anchor_idx: int
    slide_name: str
    positive_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    hard_negative_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    easy_negative_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

class RemappingUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps old module paths to current class."""
    
    def find_class(self, module, name):
        # Remap any module's AnchorPairIndices to our local one
        if name == 'AnchorPairIndices':
            return AnchorPairIndices
        return super().find_class(module, name)


def load_pairs_pickle(path: str):
    """Load pickle file with class remapping."""
    with open(path, 'rb') as f:
        return RemappingUnpickler(f).load()
###############################################################################
# FAISS Index Builder
###############################################################################

class FAISSIndexManager:
    """Manages FAISS indices for all slides."""
    
    def __init__(self, use_gpu: bool = False, gpu_id: int = 0):
        self.indices: Dict[str, Dict] = {}
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu_id = gpu_id
        self.gpu_res = None
        
    def build_indices(
        self, 
        dataset: HFDataset,
        num_workers: int = 64,
        chunk_size: int = 50000
    ) -> Dict[str, Dict]:
        """Build FAISS L2 indices for all slides using parallel processing."""
        logging.info("Building FAISS indices for all slides...")
        # Step 1: Group indices by slide name (parallel)
        slide_names = dataset.select_columns(['Sample_Names'])
        slide_to_indices = self._group_by_slide_parallel(slide_names, num_workers)
        logging.info(f"Found {len(slide_to_indices)} unique slides")
        
        # Step 2: Build FAISS index for each slide
        for slide_name, global_indices in tqdm(slide_to_indices.items(), desc="Building FAISS"):
            slide_data = dataset.select(global_indices)
            
            coords = np.column_stack([
                np.array(slide_data['centroid_x']),
                np.array(slide_data['centroid_y'])
            ]).astype('float32')
            
            # Create CPU index
            index = faiss.IndexFlatL2(2)
            index.add(coords)
            
            self.indices[slide_name] = {
                'index': index,
                'coords': coords,
                'global_indices': np.array(global_indices, dtype=np.int64),
                'n_cells': len(global_indices)
            }

        if self.use_gpu:
            self._convert_to_gpu()
        
        logging.info(f"Built FAISS indices for {len(self.indices)} slides")
        return self.indices
    
    def _group_by_slide_parallel(
        self, 
        dataset: HFDataset, 
        num_workers: int
    ) -> Dict[str, List[int]]:
        """
        Group cell indices by slide name using batched reads.
        
        For 10M+ rows, we can't load all into memory at once.
        Instead, we read in batches and process sequentially.
        Batched reads are fast; parallelism on HF datasets causes issues.
        """
        n = len(dataset)
        batch_size = 50000  # Read 50k rows at a time
        
        logging.info(f"Grouping {n:,} cells by slide name (batch_size={batch_size:,})...")
        
        slide_to_indices = defaultdict(list)
        
        # Sequential batched reads (HF datasets are not thread-safe for reads)
        num_batches = (n + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Grouping by slide"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            
            # Bulk read: single I/O operation for entire batch
            batch_names = dataset.select(range(start, end))['Sample_Names']
            
            # Process batch in memory
            for local_idx, slide_name in enumerate(batch_names):
                global_idx = start + local_idx
                slide_to_indices[str(slide_name)].append(global_idx)
        
        logging.info(f"Found {len(slide_to_indices):,} unique slides")
        return dict(slide_to_indices)
    
    def _convert_to_gpu(self):
        """Convert CPU indices to GPU."""
        self.gpu_res = faiss.StandardGpuResources()
        logging.info(f"Converting FAISS indices to GPU {self.gpu_id}...")
        
        for slide_name, data in self.indices.items():
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, self.gpu_id, data['index'])
            data['index'] = gpu_index
    
    def save(self, path: str):
        """Save indices to disk (CPU version)."""
        save_data = {}
        for slide_name, data in self.indices.items():
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(data['index'])
            else:
                cpu_index = data['index']
            
            save_data[slide_name] = {
                'index': faiss.serialize_index(cpu_index),
                'coords': data['coords'],
                'global_indices': data['global_indices'],
                'n_cells': data['n_cells']
            }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved FAISS indices to {path}")
    
    def load(self, path: str):
        """Load indices from disk."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        for slide_name, data in save_data.items():
            index = faiss.deserialize_index(data['index'])
            self.indices[slide_name] = {
                'index': index,
                'coords': data['coords'],
                'global_indices': data['global_indices'],
                'n_cells': data['n_cells']
            }
        
        if self.use_gpu:
            self._convert_to_gpu()
        
        logging.info(f"Loaded FAISS indices for {len(self.indices)} slides")


###############################################################################
# Pair Index Precomputer
###############################################################################

# ============================================================
# WORKER FUNCTION - Module level for pickling
# ============================================================

def _process_cell_chunk_worker(args: Tuple) -> List[Dict]:
    """
    Process a chunk of cells.
    
    Each chunk contains cells that may span multiple slides.
    For each unique slide in the chunk, we build a FAISS index
    and query only the cells from this chunk.
    
    Args:
        args: (
            chunk_cells: List of (global_idx, slide_name, local_idx_in_slide),
            slide_data: Dict of {slide_name: {'coords': np.array, 'global_indices': np.array}},
            pos_thresh_sq: float,
            hard_min_sq: float,
            hard_max_sq: float,
        )
    
    Returns:
        List of dicts with pair info
    """
    (
        chunk_cells,
        slide_data,
        pos_thresh_sq,
        hard_min_sq,
        hard_max_sq,
    ) = args
    
    # Group chunk cells by slide
    cells_by_slide = defaultdict(list)
    for global_idx, slide_name, local_idx in chunk_cells:
        cells_by_slide[slide_name].append((global_idx, local_idx))
    
    results = []
    
    for slide_name, cells in cells_by_slide.items():
        sdata = slide_data[slide_name]
        all_coords = np.ascontiguousarray(sdata['coords'], dtype='float32')
        all_global_indices = sdata['global_indices']
        n_total = len(all_coords)
        
        # Build FAISS index for the FULL slide
        index = faiss.IndexFlatL2(2)
        index.add(all_coords)
        
        # Extract local indices for cells in this chunk
        local_indices = np.array([c[1] for c in cells], dtype=np.int64)
        global_indices_chunk = np.array([c[0] for c in cells], dtype=np.int64)
        
        # Query coordinates for this chunk's cells
        query_coords = np.ascontiguousarray(all_coords[local_indices], dtype='float32')
        
        # Range search for positives
        lims_pos, dists_pos, idx_pos = index.range_search(query_coords, pos_thresh_sq)
        
        # Range search for hard negatives
        lims_hard, dists_hard, idx_hard = index.range_search(query_coords, hard_max_sq)
        
        all_local_indices = np.arange(n_total)
        
        for i in range(len(cells)):
            anchor_global = int(global_indices_chunk[i])
            
            # === POSITIVE PAIRS ===
            p_start, p_end = lims_pos[i], lims_pos[i + 1]
            pos_dists = dists_pos[p_start:p_end]
            pos_local = idx_pos[p_start:p_end]
            pos_mask = pos_dists > 1e-6  # Exclude self
            positive_local = pos_local[pos_mask]
            
            if len(positive_local) == 0:
                continue  # Skip anchors without positives
            
            positive_global = all_global_indices[positive_local]
            
            # === HARD NEGATIVE PAIRS ===
            h_start, h_end = lims_hard[i], lims_hard[i + 1]
            hard_dists = dists_hard[h_start:h_end]
            hard_local = idx_hard[h_start:h_end]
            hard_mask = (hard_dists >= hard_min_sq) & (hard_dists > 1e-6)
            hard_neg_local = hard_local[hard_mask]
            hard_neg_global = all_global_indices[hard_neg_local]
            if len(hard_neg_global) > 10:
                hard_neg_global = np.random.choice(hard_neg_global, 10, replace=False)#only select 10 to save memory
            
            
            # === EASY NEGATIVE PAIRS ===
            within_hard_set = set(idx_hard[h_start:h_end].tolist())
            easy_mask = ~np.isin(all_local_indices, list(within_hard_set))
            easy_neg_local = all_local_indices[easy_mask]
            easy_neg_global = all_global_indices[easy_neg_local]
            easy_neg_global = np.random.choice(easy_neg_global, 10, replace=False)#only select 10 to save memory
            
            results.append({
                'anchor_idx': anchor_global,
                'slide_name': slide_name,
                'pos': positive_global.astype(np.int32),
                'hard': hard_neg_global.astype(np.int32),
                'easy': easy_neg_global.astype(np.int32),
            })
        
        # Free FAISS index memory
        del index
    
    return results


# ============================================================
# Memory-efficient Cell-based Precomputer
# ============================================================

class PairIndexPrecomputer:
    """
    Memory-efficient cell-based pair precomputation.
    
    Strategy:
    1. Create flat list of all cells (global_idx, slide_name, local_idx)
    2. Sort by slide_name to minimize unique slides per chunk
    3. Chunk into batches of N cells
    4. For each chunk, only load slide data for slides in that chunk
    5. Process with multiprocessing
    """
    
    def __init__(
        self,
        faiss_manager: 'FAISSIndexManager',
        positive_threshold: float = 30.0,
        hard_negative_min: float = 30.0,
        hard_negative_max: float = 50.0,
        num_workers: int = 32,
        chunk_size: int = 10000,  # Cells per chunk
        batch_chunks: int = 64,   # Chunks per batch (controls memory)
    ):
        self.faiss_manager = faiss_manager
        self.positive_threshold = positive_threshold
        self.hard_negative_min = hard_negative_min
        self.hard_negative_max = hard_negative_max
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.batch_chunks = batch_chunks
        
        self.pos_thresh_sq = positive_threshold ** 2
        self.hard_min_sq = hard_negative_min ** 2
        self.hard_max_sq = hard_negative_max ** 2
    
    def _build_cell_list(self) -> List[Tuple[int, str, int]]:
        """
        Build flat list of all cells: (global_idx, slide_name, local_idx).
        Sorted by slide_name to maximize cells from same slide in each chunk.
        """
        logging.info("Building cell list...")
        
        all_cells = []
        for slide_name, data in tqdm(
            self.faiss_manager.indices.items(), 
            desc="Collecting cells"
        ):
            global_indices = data['global_indices']
            n_cells = data['n_cells']
            
            for local_idx in range(n_cells):
                global_idx = int(global_indices[local_idx])
                all_cells.append((global_idx, slide_name, local_idx))
        
        # Sort by slide_name to cluster cells from same slide together
        logging.info("Sorting cells by slide...")
        all_cells.sort(key=lambda x: x[1])
        
        logging.info(f"Total cells: {len(all_cells):,}")
        return all_cells
    
    def _create_chunks(
        self, 
        all_cells: List[Tuple[int, str, int]]
    ) -> List[List[Tuple[int, str, int]]]:
        """Split cells into chunks."""
        chunks = []
        for i in range(0, len(all_cells), self.chunk_size):
            chunk = all_cells[i:i + self.chunk_size]
            chunks.append(chunk)
        
        logging.info(f"Created {len(chunks):,} chunks of ~{self.chunk_size:,} cells each")
        return chunks
    
    def _get_slide_data_for_chunks(
        self, 
        chunks: List[List[Tuple[int, str, int]]]
    ) -> Dict[str, Dict]:
        """
        Get slide data only for slides that appear in these chunks.
        Returns minimal data needed for processing.
        """
        # Find unique slides across all chunks in this batch
        unique_slides = set()
        for chunk in chunks:
            for _, slide_name, _ in chunk:
                unique_slides.add(slide_name)
        
        # Extract only needed slide data
        slide_data = {}
        for slide_name in unique_slides:
            data = self.faiss_manager.indices[slide_name]
            slide_data[slide_name] = {
                'coords': data['coords'],  # Reference, not copy
                'global_indices': data['global_indices'],
            }
        
        return slide_data
    
    def compute_all_pairs(self, partitions: str = None, partition: str = None) -> List[AnchorPairIndices]:
        """
        Compute all valid pairs using cell-based chunked multiprocessing.
        """
        logging.info(
            f"Computing pairs (cell-based chunking):\n"
            f"  Workers: {self.num_workers}\n"
            f"  Cells per chunk: {self.chunk_size:,}\n"
            f"  Chunks per batch: {self.batch_chunks}\n"
            f"  Positive threshold: {self.positive_threshold}\n"
            f"  Hard negative range: [{self.hard_negative_min}, {self.hard_negative_max}]"
        )
        
        # Step 1: Build sorted cell list
        all_cells = self._build_cell_list()
        
        # Step 2: Create chunks
        chunks = self._create_chunks(all_cells)
        
        # Free all_cells memory
        del all_cells
        gc.collect()
        
        # Step 3: Process chunks in batches
        all_pairs = []
        n_batches = (len(chunks) + self.batch_chunks - 1) // self.batch_chunks
        
        ctx = get_context('spawn')

        print("partition:", partition)
        print("partitions:", partitions)
         
        
        for batch_idx in [partition-1]:
            batch_start = batch_idx * self.batch_chunks
            batch_end = min(batch_start + self.batch_chunks, len(chunks))
            batch_chunks_list = chunks[batch_start:batch_end]
            
            # Count cells and slides in this batch
            batch_cells = sum(len(c) for c in batch_chunks_list)
            batch_slides = len(set(
                slide for chunk in batch_chunks_list 
                for _, slide, _ in chunk
            ))
            
            logging.info(
                f"Batch {batch_idx + 1}/{n_batches}: "
                f"{len(batch_chunks_list)} chunks, "
                f"{batch_cells:,} cells, "
                f"{batch_slides} unique slides"
            )
            
            # Get slide data for this batch only
            slide_data = self._get_slide_data_for_chunks(batch_chunks_list)
            
            # Create tasks for this batch
            tasks = [
                (
                    chunk,
                    slide_data,
                    self.pos_thresh_sq,
                    self.hard_min_sq,
                    self.hard_max_sq,
                )
                for chunk in batch_chunks_list
            ]
            
            # Process batch with multiprocessing
            actual_workers = min(self.num_workers, len(tasks))
            
            with ctx.Pool(
                processes=actual_workers,
                maxtasksperchild=4,  # Restart workers periodically
            ) as pool:
                
                with tqdm(
                    total=len(tasks),
                    desc=f"Batch {batch_idx + 1}/{n_batches}",
                    leave=False,
                ) as pbar:
                    for chunk_results in pool.imap_unordered(
                        _process_cell_chunk_worker,
                        tasks,
                        chunksize=1,
                    ):
                        # Convert dicts to dataclass
                        for r in chunk_results:
                            all_pairs.append(AnchorPairIndices(
                                anchor_idx=r['anchor_idx'],
                                slide_name=r['slide_name'],
                                positive_indices=r['pos'].astype(np.int64),
                                hard_negative_indices=r['hard'].astype(np.int64),
                                easy_negative_indices=r['easy'].astype(np.int64),
                            ))
                        pbar.update(1)
                        pbar.set_postfix({'pairs': f"{len(all_pairs):,}"})
            
            # Cleanup batch memory
            del slide_data
            del tasks
            gc.collect()
            
            logging.info(
                f"Batch {batch_idx + 1} complete. "
                f"Total pairs: {len(all_pairs):,}"
            )
        
        logging.info(f"Found {len(all_pairs):,} valid anchors with positive pairs")
        return all_pairs


###############################################################################
# Datasets
###############################################################################
class HFPairDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that wraps HuggingFace pairs dataset.
    Loads pair indices on-the-fly from memory-mapped Arrow files.
    """
    
    def __init__(
        self,
        pairs_dataset: HFDataset,
        indices: Optional[np.ndarray] = None,
        num_positives: int = 1,
        num_hard_negatives: int = 2,
        num_easy_negatives: int = 1,
    ):
        """
        Args:
            pairs_dataset: HuggingFace dataset with pair indices
            indices: Optional subset of indices to use (for train/val split)
            num_positives: Number of positives to sample per anchor
            num_hard_negatives: Number of hard negatives to sample
            num_easy_negatives: Number of easy negatives to sample
        """
        self.pairs = pairs_dataset
        self.indices = indices if indices is not None else np.arange(len(pairs_dataset))
        self.num_positives = num_positives
        self.num_hard_negatives = num_hard_negatives
        self.num_easy_negatives = num_easy_negatives

        self.pairs = pairs_dataset
    
    def __len__(self) -> int:
        return len(self.indices)



    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load pair data on-the-fly."""
        actual_idx = int(self.indices[idx])
        
        # HF dataset handles memory mapping automatically
        item = self.pairs[actual_idx]
        
        # Sample from available pairs
        positives = self._sample(item['positive_indices'], self.num_positives)
        hard_negs = self._sample(item['hard_negative_indices'], self.num_hard_negatives)
        easy_negs = self._sample(item['easy_negative_indices'], self.num_easy_negatives)
        
        return {
            'anchor_idx': item['anchor_idx'],
            'slide_name': item['slide_name'],
            'positive_indices': positives,
            'hard_negative_indices': hard_negs,
            'easy_negative_indices': easy_negs,
        }
    
    def _sample(self, items: List[int], n: int) -> List[int]:
        """Sample n items from list."""
        if len(items) == 0:
            return []
        if len(items) <= n:
            return list(items)
        return random.sample(items, n)
class EdgeBasedHFPairDataset(torch.utils.data.Dataset):
    """
    Expands anchor-based pairs to edge-based pairs.
    Each (anchor, positive) pair becomes a separate sample.
    
    This ensures ALL positive pairs are used during training,
    matching Method A's behavior.
    
    Example:
        Anchor 0 has 5 positives → 5 separate samples
        Anchor 1 has 3 positives → 3 separate samples
        Total samples = sum of all positives across all anchors
    """
    
    def __init__(
        self,
        pairs_dataset: HFDataset,
        indices: Optional[np.ndarray] = None,
        num_hard_negatives: int = 2,
        num_easy_negatives: int = 1,
        cache_dir: Optional[str] = None,
        split_name: str = "train",
        slide_name: str = None,
        shuffle_edges: bool = True

    ):
        """
        Args:
            pairs_dataset: HuggingFace dataset with precomputed pairs
            indices: Subset of anchor indices to use (for train/val split)
            num_hard_negatives: Number of hard negatives to sample per positive
            num_easy_negatives: Number of easy negatives to sample per positive
            cache_dir: Directory to cache the edge index
            split_name: 'train' or 'val' for cache naming
            slide_name: the slide name for building the specific edge index
            shuffle_indices: whether to shuffle the generated anchor positive indieces
        """
        self.pairs = pairs_dataset
        self.anchor_indices = indices if indices is not None else np.arange(len(pairs_dataset))
        self.num_hard_negatives = num_hard_negatives
        self.num_easy_negatives = num_easy_negatives
        self.shuffle_edges = shuffle_edges
        
        # Build or load edge index
        cache_path = None


        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            if slide_name is None:
                cache_path = os.path.join(cache_dir, f"edge_index_{split_name}.npy")
            else:
                cache_path = os.path.join(cache_dir, f"edge_index_{split_name}_{slide_name}.npy")
        
        
        
        self.edge_index, self.total_edges = self._get_edge_index(cache_path)
        if self.shuffle_edges:
            self._shuffle_edges()
        logging.info(
            f"EdgeBasedDataset initialized:\n"
            f"  Anchors: {len(self.anchor_indices):,}\n"
            f"  Total edges (positive pairs): {self.total_edges:,}\n"
            f"  Expansion factor: {self.total_edges / len(self.anchor_indices):.1f}x"
        )
    
    def _get_edge_index(self, cache_path: Optional[str]) -> Tuple[np.ndarray, int]:
        """Get edge index, loading from cache if available."""
        if cache_path and os.path.exists(cache_path):
            logging.info(f"Loading cached edge index from: {cache_path}")
            edge_index = np.load(cache_path)
            return edge_index, len(edge_index)
        
        edge_index = self._build_edge_index()
        
        if cache_path:
            logging.info(f"Saving edge index to: {cache_path}")
            np.save(cache_path, edge_index)

        return edge_index, len(edge_index)
    
    def _build_edge_index(self) -> np.ndarray:
        """
        Build mapping from edge index to (anchor_dataset_idx, positive_position).
        
        Returns:
            np.ndarray of shape (N_edges, 2) where each row is [dataset_idx, pos_position]
        """
        logging.info("Building edge index (this runs once and gets cached)...")
        
        edges = []
        total_positives = 0

        for i, dataset_idx in enumerate(tqdm(self.anchor_indices, desc="Building edge index")):
            item = self.pairs[int(dataset_idx)]
            positive_indices = item['positive_indices']
            n_positives = len(positive_indices)
            total_positives += n_positives
            
            # Each positive becomes a separate edge
            for pos_position in range(n_positives):
                edges.append([int(dataset_idx), pos_position])
        edge_array = np.array(edges, dtype=np.int64)
        logging.info(
            f"Edge index built:\n"
            f"  Total edges: {len(edge_array):,}\n"
            f"  Avg positives per anchor: {total_positives / len(self.anchor_indices):.1f}"
        )
        
        return edge_array
    def _shuffle_edges(self):
        """Shuffle the edge access order."""
        rng = np.random.default_rng(seed=42)
        self.shuffled_indices = rng.permutation(self.total_edges)
        logging.info(f"Shuffled edge indices (seed 42)")
    def __len__(self) -> int:
        return self.total_edges
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single (anchor, positive) pair with sampled negatives.
        
        Returns:
            Dict with:
                - anchor_idx: int
                - slide_name: str
                - positive_indices: List[int] with single positive
                - hard_negative_indices: List[int] 
                - easy_negative_indices: List[int]
        """
        if self.shuffle_edges:
            # Map through shuffled indices for random access
            shuffled_idx = self.shuffled_indices[idx]
            # Get the anchor and position for this edge
            dataset_idx, pos_position = self.edge_index[shuffled_idx]
        else:
            # Get the anchor and position for this edge
            dataset_idx, pos_position = self.edge_index[idx]
        # Load the anchor's data
        item = self.pairs[int(dataset_idx)]
        
        # Get the SPECIFIC positive for this edge
        positive_idx = item['positive_indices'][int(pos_position)]
        # Sample negatives from the stored pool (these are shared across all positives of this anchor)
        hard_negs = self._sample(item['hard_negative_indices'], self.num_hard_negatives)
        easy_negs = self._sample(item['easy_negative_indices'], self.num_easy_negatives)
        return {
            'anchor_idx': item['anchor_idx'],
            'slide_name': item['slide_name'],
            'positive_indices': [positive_idx],  # Single positive for this edge
            'hard_negative_indices': hard_negs,
            'easy_negative_indices': easy_negs,
        }
    
    def _sample(self, items, n: int) -> List[int]:
        """Sample n items from list without replacement."""
        items = list(items)
        if len(items) == 0:
            return []
        if len(items) <= n:
            return items
        return random.sample(items, n)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'n_anchors': len(self.anchor_indices),
            'n_edges': self.total_edges,
            'expansion_factor': self.total_edges / len(self.anchor_indices),
            'avg_positives_per_anchor': self.total_edges / len(self.anchor_indices),
        }
class SingleCellDataset(TorchDataset):
    """
    Dataset for SINGLE mode (no pairing).
    Simply returns individual cell data without any spatial validation.
    """
    
    def __init__(
        self, 
        dataset: HFDataset, 
        indices: Optional[np.ndarray] = None,
        filter_empty: bool = True,
    ):
        self.data = dataset
        
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(dataset))
        
        # Optionally filter out cells with empty tokens/adjacency
        if filter_empty:
            self.indices = self._filter_valid_cells()
    
    def _filter_valid_cells(self) -> np.ndarray:
        """Filter out cells with empty tokens or adjacency matrix."""
        valid = []
        for idx in tqdm(self.indices, desc="Filtering valid cells"):
            item = self.data[int(idx)]
            if len(item['Full_Tokens']) > 0 and len(item['Rows']) > 0:
                valid.append(idx)
        
        logging.info(f"Filtered to {len(valid):,}/{len(self.indices):,} valid cells")
        return np.array(valid, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return single cell data."""
        actual_idx = int(self.indices[idx])
        item = self.data[actual_idx]
        
        return {
            'Full_Tokens': item['Full_Tokens'],
            'Rows': item['Rows'],
            'Cols': item['Cols'],
            'Shape': item['Shape'],
            'global_idx': actual_idx,
        }


###############################################################################
# Collators
###############################################################################

class SpatialPairCollator:
    """
    Collator for PAIR mode.
    Loads actual cell data for the given indices and creates paired tensors.
    """
    
    def __init__(
        self,
        dataset: HFDataset,
        device: str = 'cpu',
        use_cuda_in_collator: bool = False,
        no_sparse: bool = True
    ):
        self.data = dataset
        
        if use_cuda_in_collator and torch.cuda.is_available():
            if dist.is_initialized():
                self.device = torch.device(f'cuda:{dist.get_rank()}')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device(device)
        
        self.sep_id = token_dict["<SEP>"]
        self.cls_id = token_dict["<CLS>"]
        self.pad_id = token_dict["<PAD>"]
        self.no_sparse = no_sparse
    
    def __call__(self, batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process batch of pair indices into tensors."""
        all_tokens = []
        all_labels = []
        all_adjmtx = []
        all_type_ids = []
        all_seq_lens = []
        
        for item in batch_list:
            anchor_idx = item['anchor_idx']
            anchor_data = self._load_cell(anchor_idx)
            
            if anchor_data is None:
                continue
            
            # Process POSITIVE pairs (label=1)
            for pair_idx in item['positive_indices']:
                pair_result = self._create_pair(anchor_data, int(pair_idx), label=1)
                if pair_result is not None:
                    all_tokens.append(pair_result['tokens'])
                    all_labels.append(1)
                    all_adjmtx.append(pair_result['adjmtx'])
                    all_type_ids.append(pair_result['type_ids'])
                    all_seq_lens.append(pair_result['seq_len'])
            
            # Process HARD NEGATIVE pairs (label=0)
            for pair_idx in item['hard_negative_indices']:
                pair_result = self._create_pair(anchor_data, int(pair_idx), label=0)
                if pair_result is not None:
                    all_tokens.append(pair_result['tokens'])
                    all_labels.append(0)
                    all_adjmtx.append(pair_result['adjmtx'])
                    all_type_ids.append(pair_result['type_ids'])
                    all_seq_lens.append(pair_result['seq_len'])
            
            # Process EASY NEGATIVE pairs (label=0)
            for pair_idx in item['easy_negative_indices']:
                pair_result = self._create_pair(anchor_data, int(pair_idx), label=0)
                if pair_result is not None:
                    all_tokens.append(pair_result['tokens'])
                    all_labels.append(0)
                    all_adjmtx.append(pair_result['adjmtx'])
                    all_type_ids.append(pair_result['type_ids'])
                    all_seq_lens.append(pair_result['seq_len'])
        
        if len(all_tokens) == 0:
            return self._empty_batch()
        
        # Pad and create tensors
        tokens = self._pad_tokens(all_tokens)
        type_ids = self._pad_tokens(all_type_ids)
        adjmtx = self._pad_adjmtx_sparse(all_adjmtx)
        labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)
        seq_lens = torch.tensor(all_seq_lens, dtype=torch.long, device=self.device)
        attn_mask = (tokens != self.pad_id).bool()
        if self.no_sparse:
            adjmtx = adjmtx.to_dense()

        # import pdb; pdb.set_trace()
        return {
            'indices': tokens,
            'pair_label': labels,
            'adjmtx': adjmtx,
            'token_type_ids': type_ids,
            'attention_mask': attn_mask,
            'sequence_length': seq_lens,
        }
    
    def _load_cell(self, idx: int) -> Optional[Dict]:
        """Load cell data from dataset."""
        try:
            item = self.data[idx]
            if len(item['Rows']) == 0 or len(item['Full_Tokens']) == 0:
                return None
            return {
                'tokens': torch.tensor(item['Full_Tokens'], dtype=torch.long, device=self.device),
                'adjmtx': self._build_adj(item['Rows'], item['Cols'], item['Shape']),
            }
        except Exception as e:
            logging.debug(f"Failed to load cell {idx}: {e}")
            return None
    
    def _create_pair(
        self, 
        anchor_data: Dict, 
        pair_idx: int, 
        label: int
    ) -> Optional[Dict]:
        """Create a single anchor-pair combination."""
        pair_data = self._load_cell(pair_idx)
        if pair_data is None:
            return None
        
        anchor_tokens = anchor_data['tokens']
        pair_tokens = pair_data['tokens']
        
        # Combine tokens: [CLS] anchor [SEP] pair [SEP]
        combined = torch.cat([
            torch.tensor([self.cls_id], dtype=torch.long, device=self.device),
            anchor_tokens,
            torch.tensor([self.sep_id], dtype=torch.long, device=self.device),
            pair_tokens,
            torch.tensor([self.sep_id], dtype=torch.long, device=self.device),
        ])
        
        n_anchor = len(anchor_tokens)
        n_pair = len(pair_tokens)
        
        # Token type IDs: 1 for anchor segment, 2 for pair segment
        type_ids = torch.cat([
            torch.zeros(1 + n_anchor + 1, dtype=torch.long, device=self.device)+1,
            torch.ones(n_pair + 1, dtype=torch.long, device=self.device)+1,
        ])
        
        # Combine adjacency matrices
        combined_adj = self._combine_adj(anchor_data['adjmtx'], pair_data['adjmtx'])
        
        return {
            'tokens': combined,
            'adjmtx': combined_adj,
            'type_ids': type_ids,
            'seq_len': [n_anchor + 2, n_pair + 1],
        }
    
    def _build_adj(self, rows, cols, shape) -> coo_matrix:
        """Rebuild adjacency matrix from sparse representation."""
        data = [1] * len(rows)
        return coo_matrix((data, (rows, cols)), shape=tuple(shape))
    
    def _combine_adj(self, adj1: coo_matrix, adj2: coo_matrix) -> coo_matrix:
        """Combine two adjacency matrices with proper offsets."""
        n1, n2 = adj1.shape[0], adj2.shape[0]
        n_total = 1 + 4 + n1 + 1 + 4 + n2 + 1
        
        adj1, adj2 = adj1.tocoo(), adj2.tocoo()
        
        rows1 = adj1.row + 1 + 4
        cols1 = adj1.col + 1 + 4
        
        offset2 = 1 + 4 + n1 + 1 + 4
        rows2 = adj2.row + offset2
        cols2 = adj2.col + offset2
        
        return coo_matrix(
            (np.concatenate([adj1.data, adj2.data]),
             (np.concatenate([rows1, rows2]), np.concatenate([cols1, cols2]))),
            shape=(n_total, n_total)
        )
    
    def _pad_tokens(self, token_list: List[torch.Tensor]) -> torch.Tensor:
        """Pad token sequences to max length."""
        if len(token_list) == 0:
            return torch.empty(0, 0, dtype=torch.long, device=self.device)
        
        max_len = max(len(t) for t in token_list)
        padded = []
        for t in token_list:
            if len(t) < max_len:
                padding = torch.full(
                    (max_len - len(t),), 
                    self.pad_id, 
                    dtype=t.dtype, 
                    device=t.device
                )
                t = torch.cat([t, padding])
            padded.append(t)
        
        return torch.stack(padded)
    
    def _pad_adjmtx_sparse(self, adj_list: List[coo_matrix]) -> torch.Tensor:
        """Pad adjacency matrices and return sparse tensor."""
        if len(adj_list) == 0:
            return torch.sparse_coo_tensor(
                torch.empty(3, 0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.float32, device=self.device),
                size=(0, 0, 0)
            )
        
        batch_size = len(adj_list)
        max_nodes = max(a.shape[0] for a in adj_list)
        
        batch_idx, row_idx, col_idx, values = [], [], [], []
        for b, adj in enumerate(adj_list):
            adj = adj.tocoo()
            n_edges = len(adj.data)
            batch_idx.extend([b] * n_edges)
            row_idx.extend(adj.row.tolist())
            col_idx.extend(adj.col.tolist())
            values.extend(adj.data.tolist())
        
        indices = torch.tensor(
            [batch_idx, row_idx, col_idx], 
            dtype=torch.long, 
            device=self.device
        )
        vals = torch.tensor(values, dtype=torch.float32, device=self.device)
        
        return torch.sparse_coo_tensor(
            indices, vals, (batch_size, max_nodes, max_nodes)
        ).coalesce()
    
    def _empty_batch(self) -> Dict[str, torch.Tensor]:
        """Return empty batch structure."""
        return {
            'indices': torch.empty(0, 0, dtype=torch.long, device=self.device),
            'pair_label': torch.empty(0, dtype=torch.long, device=self.device),
            'adjmtx': torch.sparse_coo_tensor(
                torch.empty(3, 0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.float32, device=self.device),
                size=(0, 0, 0)
            ),
            'token_type_ids': torch.empty(0, 0, dtype=torch.long, device=self.device),
            'attention_mask': torch.empty(0, 0, dtype=torch.bool, device=self.device),
            'sequence_length': torch.empty(0, 2, dtype=torch.long, device=self.device),
        }


class SingleCellCollator:
    """
    Collator for SINGLE mode.
    Processes individual cells without pairing.
    """
    
    def __init__(
        self, 
        device: str = 'cpu',
        use_cuda_in_collator: bool = False,
    ):
        if use_cuda_in_collator and torch.cuda.is_available():
            if dist.is_initialized():
                self.device = torch.device(f'cuda:{dist.get_rank()}')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device(device)
        
        self.cls_id = token_dict["<CLS>"]
        self.pad_id = token_dict["<PAD>"]
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process batch of single cells into tensors."""
        all_tokens = []
        all_adj = []
        all_type_ids = []
        all_seq_lens = []
        
        for item in batch:
            if len(item['Rows']) == 0 or len(item['Full_Tokens']) == 0:
                continue
            
            tokens = torch.tensor(item['Full_Tokens'], dtype=torch.long, device=self.device)
            
            # Format: [CLS] tokens
            combined = torch.cat([
                torch.tensor([self.cls_id], dtype=torch.long, device=self.device),
                tokens
            ])
            
            # Build adjacency with offset for CLS and special tokens
            adj = self._build_adj(item['Rows'], item['Cols'], item['Shape'])
            n_genes = adj.shape[0]
            n_total = 1 + 4 + n_genes  # CLS + 4 special + genes
            
            adj = adj.tocoo()
            offset_rows = adj.row + 1 + 4
            offset_cols = adj.col + 1 + 4
            combined_adj = coo_matrix(
                (adj.data, (offset_rows, offset_cols)),
                shape=(n_total, n_total)
            )
            
            type_ids = torch.zeros(len(combined), dtype=torch.long, device=self.device)
            
            all_tokens.append(combined)
            all_adj.append(combined_adj)
            all_type_ids.append(type_ids)
            all_seq_lens.append([len(tokens) + 1, 0])  # [cell_len, 0] for single mode
        
        if len(all_tokens) == 0:
            return self._empty_batch()
        
        tokens = self._pad_tokens(all_tokens)
        type_ids = self._pad_tokens(all_type_ids)
        adj = self._pad_adj(all_adj)
        seq_lens = torch.tensor(all_seq_lens, dtype=torch.long, device=self.device)
        attn_mask = (tokens != self.pad_id).bool()
        
        return {
            'indices': tokens,
            'pair_label': None,  # No labels in single mode
            'adjmtx': adj,
            'token_type_ids': type_ids,
            'attention_mask': attn_mask,
            'sequence_length': seq_lens,
        }
    
    def _build_adj(self, rows, cols, shape) -> coo_matrix:
        return coo_matrix(([1]*len(rows), (rows, cols)), shape=tuple(shape))
    
    def _pad_tokens(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(len(t) for t in tokens)
        padded = []
        for t in tokens:
            if len(t) < max_len:
                padding = torch.full((max_len - len(t),), self.pad_id, dtype=t.dtype, device=t.device)
                t = torch.cat([t, padding])
            padded.append(t)
        return torch.stack(padded)
    
    def _pad_adj(self, adj_list: List[coo_matrix]) -> torch.Tensor:
        batch_size = len(adj_list)
        max_nodes = max(a.shape[0] for a in adj_list)
        
        batch_idx, row_idx, col_idx, values = [], [], [], []
        for b, adj in enumerate(adj_list):
            adj = adj.tocoo()
            batch_idx.extend([b] * len(adj.data))
            row_idx.extend(adj.row.tolist())
            col_idx.extend(adj.col.tolist())
            values.extend(adj.data.tolist())
        
        indices = torch.tensor([batch_idx, row_idx, col_idx], dtype=torch.long, device=self.device)
        vals = torch.tensor(values, dtype=torch.float32, device=self.device)
        
        return torch.sparse_coo_tensor(
            indices, vals, (batch_size, max_nodes, max_nodes)
        ).coalesce()
    
    def _empty_batch(self) -> Dict[str, torch.Tensor]:
        return {
            'indices': torch.empty(0, 0, dtype=torch.long, device=self.device),
            'pair_label': None,
            'adjmtx': torch.sparse_coo_tensor(
                torch.empty(3, 0, dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device), (0, 0, 0)
            ),
            'token_type_ids': torch.empty(0, 0, dtype=torch.long, device=self.device),
            'attention_mask': torch.empty(0, 0, dtype=torch.bool, device=self.device),
            'sequence_length': torch.empty(0, 2, dtype=torch.long, device=self.device),
        }


###############################################################################
# DataModule
###############################################################################

class PairwiseSpatialDataModule(pl.LightningDataModule):
    """
    Lightning DataModule supporting both SINGLE and PAIR modes.
    
    Modes:
    - "single": Simple train/test split, no spatial validation, no pairing
    - "pair": Precomputes all valid pairs using FAISS, splits on pair indices
    
    For PAIR mode:
    - prepare_data(): Build FAISS indices, compute pairs, save cache
    - setup(): Load cached pairs, split into train/val
    
    For SINGLE mode:
    - prepare_data(): Nothing to precompute
    - setup(): Simple train/test split on raw data
    """
    
    def __init__(
        self,
        path: str,
        suffix: str = "arrow",
        input_type: str = "pair",  # "pair" or "single"
        train_frac: float = 0.8,
        batch_size: int = 32,
        num_workers: int = 4,
        # Pair-mode specific parameters
        positive_threshold: float = 30.0,
        hard_negative_min: float = 30.0,
        hard_negative_max: float = 50.0,
        num_positives_per_query: int = 1,
        num_hard_negatives_per_query: int = 2,
        num_easy_negatives_per_query: int = 1,
        num_precompute_workers: int = 128,
        chunk_size: int = 5000,
        use_gpu_faiss: bool = False,
        force_rebuild: bool = False,
        partitions: int = 60,
        partition: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        use_cuda_in_collator: bool = False,
        drop_last: bool = True,
        seed: int = 42,
        # Single-mode specific
        filter_empty_cells: bool = False,
        slide_name: str = None,
        no_sparse: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Validate input_type
        if input_type not in ("pair", "single"):
            raise ValueError(f"input_type must be 'pair' or 'single', got '{input_type}'")
        
        # Paths
        self.path = path
        self.suffix = suffix
        self.base_dir = os.path.dirname(path)
        self.input_type = input_type
        
        # Training params
        self.train_frac = train_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.use_cuda_in_collator = use_cuda_in_collator
        self.drop_last = drop_last
        self.seed = seed
        
        # Pair-mode params
        self.positive_threshold = positive_threshold
        self.hard_negative_min = hard_negative_min
        self.hard_negative_max = hard_negative_max
        self.num_positives = num_positives_per_query
        self.num_hard_negs = num_hard_negatives_per_query
        self.num_easy_negs = num_easy_negatives_per_query
        self.num_precompute_workers = num_precompute_workers
        self.chunk_size = chunk_size
        self.use_gpu_faiss = use_gpu_faiss
        self.force_rebuild = force_rebuild
        self.partitions = partitions
        self.partition = partition
        
        # Single-mode params
        self.filter_empty_cells = filter_empty_cells
        self.slide_name = slide_name
        self.mode = "nodebug"
        self.no_sparse = no_sparse
        
        # Will be set during setup
        self.dataset = None
        self.train_pairs = None
        self.val_pairs = None
        self.train_indices = None
        self.val_indices = None
        
        logging.info(
            f"Initializing DataModule:\n"
            f"  Mode: {input_type.upper()}\n"
            f"  Path: {path}\n"
            f"  Train fraction: {train_frac}\n"
            f"  Batch size: {batch_size}\n"
            f"  Num workers: {num_workers}"
        )
        if self.mode == "debug":
            self.build_debug_dataloader()
    
    def _get_cache_path(self, name: str) -> str:
        """Get cache file path."""
        return os.path.join(self.base_dir, f"cache_{name}.pkl")
    
    def _compute_fingerprint(self) -> str:
        """Compute cache fingerprint based on parameters."""
        data = {
            'path': self.path,
            'positive_threshold': self.positive_threshold,
            'hard_negative_min': self.hard_negative_min,
            'hard_negative_max': self.hard_negative_max,
        }
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    
    def _load_dataset(self) -> HFDataset:
        """Load the HuggingFace dataset."""
        if self.suffix == "arrow":
            ds = load_from_disk(self.path)
            # Cleanup cache on main process only
            if not dist.is_initialized() or dist.get_rank() == 0:
                try:
                    ds.cleanup_cache_files()
                except Exception:
                    pass
            if dist.is_initialized():
                dist.barrier()
        elif self.suffix == "parquet":
            cache_dir = os.path.join(self.base_dir, "hf_cache")
            os.makedirs(cache_dir, exist_ok=True)
            ds = HFDataset.from_parquet(f"{self.path}/*.parquet", cache_dir=cache_dir)
        else:
            raise ValueError(f"Unsupported suffix: {self.suffix}")
        
        logging.info(f"Loaded dataset with {len(ds):,} samples")
        return ds
    
    def prepare_data(self):
        """
        Precompute data (runs on rank 0 only in distributed).
        
        - PAIR mode: Build FAISS indices, compute all pairs, save cache
        - SINGLE mode: Nothing to precompute
        """
        if self.input_type == "single":
            logging.info("SINGLE mode: No precomputation needed")
            return
        
        # PAIR MODE
        pairs_path = self._get_cache_path('pairs')
        faiss_path = self._get_cache_path('faiss')
        fingerprint = self._compute_fingerprint()
        
        # Check if valid cache exists
        if self.force_rebuild and not os.path.exists(pairs_path) and not os.path.exists(faiss_path):
            

            logging.info("="*60)
            logging.info("PRECOMPUTING PAIR INDICES (PAIR MODE)")
            logging.info("="*60)
            
            # Load dataset
            dataset = self._load_dataset()
            
            # Build FAISS indices
            faiss_manager = FAISSIndexManager(use_gpu=self.use_gpu_faiss)
            faiss_manager.build_indices(
                dataset,
                num_workers=self.num_precompute_workers,
                chunk_size=self.chunk_size * 10
            )
            faiss_manager.save(faiss_path) #the faiss index should be saved locally
        
            # Compute all pairs
            precomputer = PairIndexPrecomputer(
                faiss_manager=faiss_manager,
                positive_threshold=self.positive_threshold,
                hard_negative_min=self.hard_negative_min,
                hard_negative_max=self.hard_negative_max,
                num_workers=self.num_precompute_workers,
                chunk_size=self.chunk_size,
            )
        
            all_pairs = precomputer.compute_all_pairs(self.partitions, self.partition)
        
            # Calculate stats
            total_cells = sum(d['n_cells'] for d in faiss_manager.indices.values())
            
            # Save pairs with metadata
            cache_data = {
                'pairs': all_pairs,
                'fingerprint': fingerprint,
                'positive_threshold': self.positive_threshold,
                'hard_negative_min': self.hard_negative_min,
                'hard_negative_max': self.hard_negative_max,
                'total_cells': total_cells,
                'valid_anchors': len(all_pairs),
            }
            
            with open(pairs_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
            logging.info(f"Saved {len(all_pairs):,} valid anchor pairs to {pairs_path}")
            logging.info(f"Valid anchor ratio: {100*len(all_pairs)/total_cells:.1f}%")
        else:
            logging.info(f"Found cached pair path: {pairs_path} and faiss index path {faiss_path}, all pairs are loaded from the cache data")
            logging.info(f"Update! the pickled file has been converted to huggingface dataset format")
            
    def _load_cache_data(self):
        """if you precompute the pairs, use it directly"""
        fingerprint = self._compute_fingerprint()
        pairs_path = self._get_cache_path('pairs')
        try:

            cache_data = load_pairs_pickle(pairs_path)
            

            return cache_data
            if cache_data.get('fingerprint') == fingerprint:
                logging.info(f"Found valid cached pairs at {pairs_path}")
                return
            else:
                logging.info("Cache fingerprint mismatch, you should rebuild it...")
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}, you should rebuild it...")
    def setup(self, stage: Optional[str] = None):
        """
        Setup train/val datasets.
        
        - PAIR mode: Load cached pairs, split on pair indices
        - SINGLE mode: Simple train/test split on raw data indices
        """
        # Load dataset
        self.dataset = self._load_dataset()
        
        if self.input_type == "single":
            self._setup_single_mode()
        else:
            self._setup_pair_mode()
    
    def _setup_single_mode(self):
        """Setup for SINGLE mode - simple train/test split."""
        logging.info("Setting up SINGLE mode...")
        
        n_total = len(self.dataset)
        n_train = int(n_total * self.train_frac)
        
        # Deterministic shuffle
        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(n_total)
        
        self.train_indices = shuffled_indices[:n_train]
        self.val_indices = shuffled_indices[n_train:]
        
        logging.info(
            f"SINGLE mode split complete:\n"
            f"  Train: {len(self.train_indices):,} cells\n"
            f"  Val: {len(self.val_indices):,} cells"
        )
    
    def _get_pairds_dir(self) -> str:
        return os.path.join(self.base_dir, "pairs_dataset")

    def _filter_ds(self):
        if self.slide_name is not None:
            self.pairs_dataset = self.pairs_dataset.filter(
                lambda x: x["slide_name"] == self.slide_name,
                load_from_cache_file=True,
            )
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(self.pairs_dataset.num_rows, size=2000, replace=False)
            self.pairs_dataset = self.pairs_dataset.select(indices)
        else:
            pass
    
    
    def _setup_pair_mode(self):
        """Setup for PAIR mode with pair dataset loading."""
        logging.info("Setting up PAIR mode with datasets index loading...")
        
        pairs_dataset_dir = self._get_pairds_dir()
        
        # Check mmap exists, if not convert from pickle
        if not os.path.exists(f"{self.base_dir}/metadata.json"):
            logging.info("Dataset not found, please run 'prepare_data' and 'convert_pickle_to_hf_dataset' to convert to the huggingface datasets first...")
            logging.info("Finally, you should get the metadata.json and pairs_dataset in the base_dir")
            # self.convert_pickle_to_hf_dataset()
        
        # Lazy load (does NOT load data into RAM)
        self.pairs_dataset = load_from_disk(pairs_dataset_dir)
        if self.mode != "debug":
            self._filter_ds()
        logging.info(f"Loaded pairs dataset with {len(self.pairs_dataset):,} pairs, specify slide: {self.slide_name}")
        
        
        # Split indices
        n_total = len(self.pairs_dataset)
        n_train = int(n_total * self.train_frac)
        
        rng = np.random.default_rng(self.seed)
        shuffled = rng.permutation(n_total)
        
        self.train_pair_indices = shuffled[:n_train]
        self.val_pair_indices = shuffled[n_train:]
        
        logging.info(
            f"PAIR mode ready:\n"
            f"  Train: {len(self.train_pair_indices):,}\n"
            f"  Val: {len(self.val_pair_indices):,}"
        )
    
    def _create_pair_dataloader(
        self, 
        anchor_indices: np.ndarray, 
        shuffle: bool,
        split_name: str = "train"
    ) -> DataLoader:
        """Create DataLoader for given indices."""
        
        #all the positive neighbors from the anchor should be visible
        logging.info("Using the EdgeBasedHFPairDataset to sample all the positive pairs")
        dataset = EdgeBasedHFPairDataset(
            pairs_dataset=self.pairs_dataset,
            indices=anchor_indices,
            num_hard_negatives=self.num_hard_negs,
            num_easy_negatives=self.num_easy_negs,
            cache_dir=os.path.join(self.base_dir, "edge_cache"),
            split_name=split_name,
            slide_name=self.slide_name
        )
        # Log the expansion
        stats = dataset.get_stats()
        logging.info(
            f"{split_name.upper()} dataset:\n"
            f"  Anchors: {stats['n_anchors']:,}\n"
            f"  Edges (samples): {stats['n_edges']:,}\n"
            f"  Expansion: {stats['expansion_factor']:.1f}x"
        )
        
        # Create collator with cell data path for lazy loading
        collator = SpatialPairCollator(
            dataset=self.dataset,
            use_cuda_in_collator=False, 
            no_sparse=self.no_sparse
        )
        
        sampler = None
        do_shuffle = shuffle
        
        if dist.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
            )
            do_shuffle = False
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=do_shuffle,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=self.pin_memory,
            # persistent_workers=self.persistent_workers,
            sampler=sampler,
            drop_last=self.drop_last,
        )
    def _create_single_dataloader(
        self, 
        indices: np.ndarray, 
        shuffle: bool
    ) -> DataLoader:
        """Create dataloader for SINGLE mode."""
        dataset = SingleCellDataset(
            dataset=self.dataset,
            indices=indices,
            filter_empty=self.filter_empty_cells,
        )
        
        collator = SingleCellCollator(
            use_cuda_in_collator=self.use_cuda_in_collator,
        )
        
        sampler = None
        do_shuffle = shuffle
        
        if dist.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
            )
            do_shuffle = False
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=do_shuffle,
            num_workers=self.num_workers,#test
            collate_fn=collator,
            pin_memory=self.pin_memory,
            # persistent_workers=self.persistent_workers,
            sampler=sampler,
            drop_last=self.drop_last,
            # prefetch_factor=2 if self.num_workers > 0 else None,
        )
    
    def train_dataloader(self) -> DataLoader:
        if self.input_type == "single":
            return self._create_single_dataloader(self.train_indices, shuffle=True)
        else:
            if self.mode != "debug":
                return self._create_pair_dataloader(self.train_pair_indices, shuffle=True, split_name = "train")
            else:
                logging.info("wait! let's debug the sampler")
                return self.train_loader
    
    def val_dataloader(self) -> DataLoader:
        if self.input_type == "single":
            return self._create_single_dataloader(self.val_indices, shuffle=False)
        else:

            if self.mode != "debug":
                return self._create_pair_dataloader(self.val_pair_indices, shuffle=False, split_name = "validation")
            else:
                
                logging.info("wait! let's debug the sampler")

                return self.val_loader
    def build_debug_dataloader(self):
        datapath = "/scratch/project_465001820/Spatialformer/cache"
        self.train_loader, self.val_loader = create_dataloader(datapath, 
                                                    num_workers = 4, 
                                                    batch_size = 8,
                                                    directionality = False,
                                                    context_length = 500, 
                                                    padding_idx = 0, 
                                                    special_token_num = 4, 
                                                    n_bins = 51, 
                                                    sep_token = 1949, 
                                                    cls_token = 1)


###############################################################################
# Convert the large pickled file into the numpy array and meta data
###############################################################################
def convert_pickle_to_hf_dataset(
    pickle_path: str, 
    output_dir: str,
    num_shards: int = 16,  # Number of shards for large datasets
) -> str:
    """
    Convert pickle cache to HuggingFace Arrow dataset.
    
    Benefits over numpy mmap:
    - Native HF integration
    - Built-in memory mapping
    - Better handling of variable-length sequences
    - Easy distributed loading
    
    Args:
        pickle_path: Path to pickle file with pairs
        output_dir: Output directory for HF dataset
        num_shards: Number of shards (for parallel writing)
    
    Returns:
        Path to saved dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    from datasets import Dataset as HFDataset, Features, Sequence, Value, load_from_disk
    logging.info(f"Loading pickle from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        cache_data = RemappingUnpickler(f).load()
    
    all_pairs = cache_data['pairs']
    n_pairs = len(all_pairs)
    logging.info(f"Converting {n_pairs:,} pairs to HuggingFace dataset...")
    
    # Process in batches for memory efficiency
    batch_size = 100000
    all_records = []
    
    for i in tqdm(range(0, n_pairs, batch_size), desc="Processing pairs"):
        batch = all_pairs[i:i + batch_size]
        
        for p in batch:
            record = {
                'anchor_idx': int(p.anchor_idx),
                'slide_name': str(p.slide_name),
                'positive_indices': p.positive_indices.astype(np.int32).tolist(),
                'hard_negative_indices': p.hard_negative_indices.astype(np.int32).tolist(),
                'easy_negative_indices': p.easy_negative_indices.astype(np.int32).tolist(),
            }
            all_records.append(record)
    
    # Free memory
    del all_pairs
    del cache_data
    
    logging.info("Creating HuggingFace dataset...")
    
    # Define features explicitly for better type handling
    features = Features({
        'anchor_idx': Value('int32'),
        'slide_name': Value('string'),
        'positive_indices': Sequence(Value('int32')),
        'hard_negative_indices': Sequence(Value('int32')),
        'easy_negative_indices': Sequence(Value('int32')),
    })
    
    # Create dataset
    dataset = HFDataset.from_list(all_records, features=features)
    
    # Save to disk with sharding for better I/O
    dataset_path = os.path.join(output_dir, 'pairs_dataset')
    logging.info(f"Saving dataset to {dataset_path}...")
    dataset.save_to_disk(dataset_path, num_shards=num_shards)
    
    # Save metadata separately
    metadata = {
        'n_pairs': n_pairs,
        'positive_threshold': cache_data.get('positive_threshold') if 'cache_data' in dir() else None,
        'hard_negative_min': cache_data.get('hard_negative_min') if 'cache_data' in dir() else None,
        'hard_negative_max': cache_data.get('hard_negative_max') if 'cache_data' in dir() else None,
    }
    with open(os.path.join(dataset_path, 'pair_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Report size
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(dataset_path)
        for f in filenames
    )
    logging.info(f"Done! Total size: {total_bytes / 1e9:.2f} GB")
    logging.info(f"Dataset saved to: {dataset_path}")
    
    return dataset_path

###############################################################################
# Usage Examples
###############################################################################

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='getting the pos and neg pairs')
    # parser.add_argument('--partitions', type = int, default=1, help='The partitions number of the data')
    # parser.add_argument('--partition', type = int, default = None, help='The partition of the data')
    # args = parser.parse_args()

    # partition = args.partition
    # partitions = args.partitions


    # Example 1: SINGLE mode (no pairing, no spatial validation)
    # dm_single = PairwiseSpatialDataModule(
    #     path="/path/to/data",
    #     suffix="arrow",
    #     input_type="single",  # <-- Simple mode
    #     train_frac=0.8,
    #     batch_size=64,
    #     num_workers=8,
    #     filter_empty_cells=True,  # Filter cells with no tokens/adj
    # )
    
    # dm_single.prepare_data()  # Does nothing in single mode
    # dm_single.setup()
    
    # train_loader = dm_single.train_dataloader()
    # val_loader = dm_single.val_dataloader()
    
    # Batch structure in SINGLE mode:
    # {
    #     'indices': [B, max_seq_len],
    #     'pair_label': None,  # No labels
    #     'adjmtx': sparse [B, max_nodes, max_nodes],
    #     'token_type_ids': [B, max_seq_len],
    #     'attention_mask': [B, max_seq_len],
    #     'sequence_length': [B, 2],  # [cell_len, 0]
    # }
    
    # Example 2: PAIR mode (full spatial pairing with FAISS)

    #####################################################################
    #ATTENTION: this should be run in multiple nodes (<2h) otherwise ~1d#
    #####################################################################
    # dm_pair = PairwiseSpatialDataModule(
    #     path="/scratch/project_465001820/Spatialformer/cache/xenium_5k_pandavid_dataset_v2",
    #     suffix="arrow",
    #     input_type="pair",  # <-- Pair mode
    #     train_frac=0.99,
    #     batch_size=32,
    #     num_workers=8,
    #     positive_threshold=30.0,
    #     hard_negative_min=50.0,
    #     hard_negative_max=150.0,
    #     num_positives_per_query=1,
    #     num_hard_negatives_per_query=2,
    #     num_easy_negatives_per_query=1,
    #     num_precompute_workers=128,  # Use all 128 workers
    #     chunk_size=5000,
    #     force_rebuild=False,  # Use cache if available
    #     partitions=partitions,
    #     partition=partition
    # )
    
    # dm_pair.prepare_data()  # Builds FAISS, computes pairs, saves cache
    # dm_pair.setup()
    
    # train_loader = dm_pair.train_dataloader()
    # val_loader = dm_pair.val_dataloader()
    
    # Batch structure in PAIR mode:
    # {
    #     'indices': [B * (1+2+1), max_seq_len],  # anchor-pair combinations
    #     'pair_label': [B * 4],  # 1 for positive, 0 for negative
    #     'adjmtx': sparse [B * 4, max_nodes, max_nodes],
    #     'token_type_ids': [B * 4, max_seq_len],
    #     'attention_mask': [B * 4, max_seq_len],
    #     'sequence_length': [B * 4, 2],  # [anchor_len, pair_len]
    # }
    convert_pickle_to_hf_dataset(pickle_path = "/scratch/project_465001820/Spatialformer/cache/cache_pairs.pkl", 
                            output_dir = "/scratch/project_465001820/Spatialformer/cache")
    
