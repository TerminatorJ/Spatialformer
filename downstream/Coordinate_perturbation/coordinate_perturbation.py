#!/usr/bin/env python
# coding: utf-8

# # The coordinates perturbation for investgating the gene-gene co-occurrent robustness
# 
# - ***Motivation***:  
#   Spatial technologies inherently own the incorrectness of gene spatial coordinations and cell segmentation. The definition of our gene-gene co-occurrence should be largely based on the accurate positional information.
#   Therefore, we assay the gene pairs robustness based on different level of perturbations in this notebook
# 
# - ***test files***
#   - Brain: from Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs as [***brain_transcripts.csv.gz***](https://figshare.com/account/projects/238169/articles/30627677)
#   - Lung: from GSM7990543_output-XETG00048__0003817__VUILD102LA__20230308__003731 as [**lung_transcripts.csv.gz***](https://figshare.com/account/projects/238169/articles/30627677)
#   


import sys
sys.path.append("/scratch/project_465001820/Spatialformer/data_preprocessing")
from process import KNN_Radius_Graph
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from dask.distributed import Client
import itertools
from tqdm.auto import tqdm
from dask.distributed import as_completed
# Usage
from typing import Optional, List, Tuple
import pickle
import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue
from functools import partial
import gc


def get_gco(gene_binary_matrix: np.ndarray, all_genes: List[str]) -> List:
    """
    Get the non-redundance gene pairs from the co-occurrence matrix

    Args:
    gene_binary_matrix: np.ndarray, the binary gene-gene co-occurrence matrix
    all_genes: List. A list with all the gene names in the dataset
    ==========
    Return:
    gene_pair: a single list that contains the paired gene pairs 
    
    """
    all_genes = list(sorted(all_genes))
    
    # Create upper triangle mask (excluding diagonal)
    upper_triangle = np.triu(gene_binary_matrix, k=1)
    
    # Find non-zero positions in upper triangle
    nonzero_rows, nonzero_cols = np.where(upper_triangle != 0)
    nonzero_values = gene_binary_matrix[nonzero_rows, nonzero_cols]
    
    gene_pair = []
    
    for row, col in zip(nonzero_rows, nonzero_cols):
        gene_pair.append([all_genes[row], all_genes[col]])
    return gene_pair
def gaussian_noise(df: pd.DataFrame = None, std: int = 0, iteration: int = 0, sigma: float=0.05) -> pd.DataFrame:
    """
    Add gaussian noise to the transcript spatial coordinates

    Args:
    df: dataframe of the transcripts.csv file
    std: the standard deviation of the perturbation, which indicates the strength of the noise.
    sigma: the 
    ==============================
    Return:
    the perturbed dataframe

    """
    perturb_df = df.copy()
    #0.2 is normal noise, but we want very small as 0.1 in the first 50 steps
    # Sigmas array: first 30 points from 0.01 to 0.1, then 500 points all 0.2
    sigmas = np.concatenate([
        np.linspace(0.01, 0.2, 50),  
        [std] * 2000
    ])
    
    noise_x = np.random.normal(0, sigmas[iteration], perturb_df.shape[0])
    noise_y = np.random.normal(0, sigmas[iteration], perturb_df.shape[0])
    noise_z = np.random.normal(0, sigmas[iteration], perturb_df.shape[0])
    

    # accumulate the noise
    if iteration == 0:
        perturb_df["x_0"] = perturb_df["x"] + noise_x
        perturb_df["y_0"] = perturb_df["y"] + noise_y
        perturb_df["z_0"] = perturb_df["z"] + noise_z
    else:
        perturb_df[f"x_{iteration}"] = perturb_df[f"x_{iteration-1}"] + noise_x
        perturb_df[f"y_{iteration}"] = perturb_df[f"y_{iteration-1}"] + noise_y
        perturb_df[f"z_{iteration}"] = perturb_df[f"z_{iteration-1}"] + noise_z
        #delete the previous columns
        cols_to_delete = [f"x_{iteration-1}", f"y_{iteration-1}", f"z_{iteration-1}"]
        perturb_df = perturb_df.drop(columns=cols_to_delete, errors="ignore")


    #assign the noise back to the original coordinates
    perturb_df["x"] = perturb_df[f"x_{iteration}"]
    perturb_df["y"] = perturb_df[f"y_{iteration}"]
    perturb_df["z"] = perturb_df[f"z_{iteration}"]
    return perturb_df

def metric(gene_pair1: List, gene_pair2: List)-> float:
    """
    evaluation of the robustness

    Args:
    gene_pair1: List, the reference gene pairs that were originally detected by the actual coordinate system
    gene_pair2: List, the gene pairs that were found when the perturbed coordinates were introduced
    ============================
    Return:
    
    """
    
    assert len(gene_pair1) >=1, "The reference gene pairs not found"
    ovlp_pairs = [pair for pair in gene_pair2 if pair in gene_pair1]
    keep_pct = len(ovlp_pairs)/len(gene_pair1)

    return keep_pct
 
def boundary_selection(df: pd.DataFrame, quantile: float = 0.95, frac: float = 0.9):
    """
    Randomly select the transcripts in the boundary

    Args:
    df: pd.DataFrame. The dataframe waits for filtering
    quantile: float. The quantile of the distance to the nucleus. default 0.8.
    frac: float. The fraction of the transcripts that used to be selected in the boundary region
    ====================
    Return:
    the filtered dataframe
    
    
    """
        
    # keep all that are close to the nucleus
    df_nucleus = df[df["overlaps_nucleus"] == 1]

    df_cyto = df[df["overlaps_nucleus"] == 0]

    # Compute xth percentile value
    val_pct = df_cyto["nucleus_distance"].quantile(quantile)

    
    # Bottom x%: randomly select 20%
    df_cytp_kp = df_cyto[(df_cyto["nucleus_distance"] < val_pct)]
    df_bdr = df_cyto[(df_cyto["nucleus_distance"] >= val_pct)]
    # print("df_nucleus:", df_nucleus.shape)
    # print("df_cytp_kp:", df_cytp_kp.shape)
    df_sample = df_bdr.sample(frac=frac)
    # print("df_sample:", df_sample.shape)
    
    # Combine
    df_selected = pd.concat([df_nucleus, df_cytp_kp, df_sample]).reset_index(drop=True)
    # all_genes = df_selected["gene"]
    return df_selected
# ----------------------------
# Pre-calculate noise transformations
# ----------------------------
def generate_all_perturbations(
    raw_df: pd.DataFrame,
    cell_ids: List[str],
    iter_num: int,
    std: int,
    boundary_noise: bool = False,
    quantile: float = 0.8,
    frac: float = 0.8
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Pre-generate ALL perturbed dataframes for all cells and all iterations
    
    Returns:
        Dict[(cell_id, iteration)] = perturbed_df
    """
    print(f"Pre-generating {len(cell_ids) * iter_num} perturbed dataframes...")
    
    # Group by cell
    cell_groups = {cid: group.reset_index(drop=True) 
                   for cid, group in raw_df.groupby("cell_id") 
                   if cid in cell_ids}
    
    perturbed_data = {}
    
    for cid in tqdm(cell_ids, desc="Generating perturbations"):
        if cid not in cell_groups:
            continue
            
        cell_df = cell_groups[cid]
        current_df = cell_df.copy()
        
        # Generate perturbations for all iterations
        for iteration in range(iter_num):
            # Apply cumulative noise
            df_noise = gaussian_noise(df=current_df, std=std, iteration=iteration)
            
            if boundary_noise:
                df_noise = boundary_selection(df_noise, quantile, frac)
            
            # Store this perturbation
            perturbed_data[(cid, iteration)] = df_noise.copy()
            
            # Update current state for next iteration (cumulative)
            current_df = df_noise
    
    print(f"Generated {len(perturbed_data)} perturbed dataframes")
    return perturbed_data
# ----------------------------
# Process single task
# ----------------------------

def run_for_sc_precomputed(
    cell_id: str,
    iteration: int,
    raw_df: pd.DataFrame,
    perturbed_df: pd.DataFrame,
    all_genes: List[str],
    clustering: bool,
    radius: int
) -> Tuple[float, int]:
    """
    Process a single cell at a single iteration with pre-computed perturbation
    
    Args:
        raw_df: Original reference dataframe for this cell
        perturbed_df: Pre-computed perturbed dataframe
    """
    # Get reference gene pairs (from original)
    data_graph = KNN_Radius_Graph(
        radius=radius, 
        dataset=raw_df, 
        is_3D=True, 
        cell_ID=cell_id, 
        ref_gene=all_genes, 
        clustering=clustering
    )
    gene_binary_matrix, _, _ = data_graph.get_gene_matrix(
        pair_threshold=3, 
        self_threshold=3, 
        plot=False
    )
    ref_pair_num = int(np.sum(np.triu(gene_binary_matrix, k=1)))
    ref_gene_pairs = get_gco(gene_binary_matrix, all_genes)
    
    # Get perturbed gene pairs
    data_graph_ptb = KNN_Radius_Graph(
        radius=radius, 
        dataset=perturbed_df, 
        is_3D=True, 
        cell_ID=cell_id, 
        ref_gene=all_genes,
        clustering=clustering
    )
    gene_binary_matrix_n, _, _ = data_graph_ptb.get_gene_matrix(
        pair_threshold=3, 
        self_threshold=3, 
        plot=False
    )
    ptb_pair_num = int(np.sum(np.triu(gene_binary_matrix_n, k=1)))
    ptb_gene_pairs = get_gco(gene_binary_matrix_n, all_genes)
    
    # Calculate metric

    pct = metric(ref_gene_pairs, ptb_gene_pairs)
    
    return pct, ref_pair_num


def run_for_sc_precomputed_wrapper(args):
    """Wrapper for multiprocessing"""
    try:
        cell_id, iteration, raw_df, perturbed_df, all_genes, clustering, radius = args
        
        pct, ref_pair_num = run_for_sc_precomputed(
            cell_id=cell_id,
            iteration=iteration,
            raw_df=raw_df,
            perturbed_df=perturbed_df,
            all_genes=all_genes,
            clustering=clustering,
            radius=radius
        )
        
        # Clean up
        del raw_df, perturbed_df
        gc.collect()
        
        return (cell_id, iteration, pct, ref_pair_num)
        
    except Exception as e:
        pass
        # print(f"Error processing cell {args[0]} iteration {args[1]}: {e}")
        # import traceback
        # traceback.print_exc()
        return None


# ----------------------------
# Optimized Perturb class
# ----------------------------

class Perturb:
    def __init__(self, std: int = 20):
        self.std = std
        
    def qc(self, df) -> pd.DataFrame:
        """Run QC and return filtered dataframe."""
        rename_map = {}
        if "x_location" in df.columns:
            rename_map["x_location"] = "x"
        if "y_location" in df.columns:
            rename_map["y_location"] = "y"
        if "z_location" in df.columns:
            rename_map["z_location"] = "z"
        if "feature_name" in df.columns:
            rename_map["feature_name"] = "gene"
        if rename_map:
            df = df.rename(columns=rename_map)
        
        df_flt1 = df[
            ~(df["gene"].str.startswith("Neg") |
              df["gene"].str.startswith("BLANK") |
              df["gene"].str.startswith("Unassigned"))
        ].copy()

        vc = df_flt1["cell_id"].value_counts()
        if "UNASSIGNED" in vc.index:
            vc = vc.drop("UNASSIGNED")
        good_cells = vc[vc >= 30].index
        df_flt2 = df_flt1[df_flt1["cell_id"].isin(good_cells)].reset_index(drop=True)
        return df_flt2

    def run_all_precomputed(
        self,
        raw_df: pd.DataFrame,
        perturbed_data: Dict[Tuple[str, int], pd.DataFrame],
        all_genes: List[str],
        cell_ids: List[str],
        iter_num: int,
        n_processes: int = None,
        chunk_size: int = 10,
        clustering: bool = True,
        radius: int = 5
    ) -> Tuple[Dict, Dict]:
        """
        Process all pre-computed perturbations in parallel
        """
        if n_processes is None:
            n_processes = max(1, os.cpu_count() - 2)
        
        # Get raw data for each cell
        raw_cell_groups = {cid: group.reset_index(drop=True) 
                          for cid, group in raw_df.groupby("cell_id") 
                          if cid in cell_ids}
        
        # ✅ Generate ALL tasks upfront
        all_tasks = []
        for cid in cell_ids:
            if cid not in raw_cell_groups:
                continue
                
            raw_cell_df = raw_cell_groups[cid]
            
            for iteration in range(iter_num):
                key = (cid, iteration)
                if key not in perturbed_data:
                    continue
                
                perturbed_df = perturbed_data[key]
                
                all_tasks.append((
                    cid,
                    iteration,
                    raw_cell_df,
                    perturbed_df,
                    all_genes,
                    clustering,
                    radius
                ))
        
        total_tasks = len(all_tasks)
        print(f"Processing {total_tasks} tasks with {n_processes} workers...")
        
        # ✅ Process ALL tasks in parallel
        all_results = []
        failed_num = 0
        with Pool(processes=n_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(run_for_sc_precomputed_wrapper, all_tasks, chunksize=chunk_size),
                total=total_tasks,
                desc="Processing all perturbations"
            ):
                if result is not None:
                    all_results.append(result)
                
                    
        
        # ✅ Organize results by iteration
        results = {i: [] for i in range(iter_num)}
        all_nums = {i: [] for i in range(iter_num)}
        
        for cell_id, iteration, pct, ref_pair_num in all_results:
            results[iteration].append(pct)
            all_nums[iteration].append(ref_pair_num)
        
        print(f"Successfully processed {len(all_results)}/{total_tasks} tasks")
        
        return results, all_nums


# ----------------------------
# Optimized pipeline
# ----------------------------

def run_optimized_pipeline_precomputed(
    raw_df,
    iter_num: int = 100,
    limit_cells: int = 500,
    std: int = 1,
    boundary_noise: bool = False,
    quantile: float = 0.8,
    frac: float = 0.8,
    clustering: bool = True,
    chunk_size: int = 10,
    n_processes: int = None,
    radius: int = 5
):
    """
    Optimized pipeline:
    1. Pre-generate ALL perturbations (sequential, fast)
    2. Process ALL tasks in parallel (parallel, slow part)
    """
    if n_processes is None:
        n_processes = max(1, os.cpu_count() - 2)
    
    print(f"Using {n_processes} worker processes with chunk size {chunk_size}")
    
    p = Perturb(std=std)
    raw_df = p.qc(raw_df)
    
    # Get cell IDs
    all_cell_ids = list(raw_df["cell_id"].unique())
    if limit_cells:
        all_cell_ids = all_cell_ids[:limit_cells]
    
    all_genes = raw_df["gene"].unique().tolist()
    
    print(f"Processing {len(all_cell_ids)} cells for {iter_num} iterations")
    print(f"Total tasks: {len(all_cell_ids) * iter_num}")
    
    # ✅ STEP 1: Pre-generate all perturbations (fast, sequential)
    print("\n" + "="*60)
    print("STEP 1: Generating all perturbations")
    print("="*60)
    
    perturbed_data = generate_all_perturbations(
        raw_df=raw_df,
        cell_ids=all_cell_ids,
        iter_num=iter_num,
        std=std,
        boundary_noise=boundary_noise,
        quantile=quantile,
        frac=frac
    )
    
    # ✅ STEP 2: Process all in parallel (slow, but parallelized)
    print("\n" + "="*60)
    print("STEP 2: Processing all gene pairs in parallel")
    print("="*60)
    
    results, all_nums = p.run_all_precomputed(
        raw_df=raw_df,
        perturbed_data=perturbed_data,
        all_genes=all_genes,
        cell_ids=all_cell_ids,
        iter_num=iter_num,
        n_processes=n_processes,
        chunk_size=chunk_size,
        clustering=clustering,
        radius=radius
    )
    
    # Get final perturbed dataframe
    final_dfs = []
    for cid in all_cell_ids:
        key = (cid, iter_num - 1)
        if key in perturbed_data:
            final_dfs.append(perturbed_data[key])
    
    final_df = pd.concat(final_dfs, axis=0, ignore_index=True) if final_dfs else pd.DataFrame()
    
    # Clean up
    del perturbed_data
    gc.collect()
    
    return results, all_nums, final_df


# ----------------------------
# Batched version for memory efficiency
# ----------------------------

def run_optimized_pipeline_precomputed_batched(
    raw_df,
    iter_num: int = 100,
    limit_cells: int = 500,
    std: int = 1,
    boundary_noise: bool = False,
    quantile: float = 0.8,
    frac: float = 0.8,
    clustering: bool = True,
    chunk_size: int = 10,
    n_processes: int = None,
    radius: int = 5,
    batch_size: int = 100  # Process 100 cells at a time
):
    """
    Memory-efficient version: Process cells in batches
    """
    if n_processes is None:
        n_processes = max(1, os.cpu_count() - 2)
    
    p = Perturb(std=std)
    raw_df = p.qc(raw_df)
    
    all_cell_ids = list(raw_df["cell_id"].unique())
    if limit_cells:
        all_cell_ids = all_cell_ids[:limit_cells]
    
    all_genes = raw_df["gene"].unique().tolist()
    
    print(f"Processing {len(all_cell_ids)} cells in batches of {batch_size}")
    
    # Store all results
    all_results = {i: [] for i in range(iter_num)}
    all_nums_combined = {i: [] for i in range(iter_num)}
    
    # Process in batches
    for batch_start in range(0, len(all_cell_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(all_cell_ids))
        batch_cells = all_cell_ids[batch_start:batch_end]
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_start//batch_size + 1}: Cells {batch_start}-{batch_end-1}")
        print(f"{'='*60}")
        
        # Filter raw_df for this batch
        batch_raw_df = raw_df[raw_df['cell_id'].isin(batch_cells)].reset_index(drop=True)
        
        # Generate perturbations for this batch
        perturbed_data = generate_all_perturbations(
            raw_df=batch_raw_df,
            cell_ids=batch_cells,
            iter_num=iter_num,
            std=std,
            boundary_noise=boundary_noise,
            quantile=quantile,
            frac=frac
        )
        
        # Process this batch
        results, all_nums = p.run_all_precomputed(
            raw_df=batch_raw_df,
            perturbed_data=perturbed_data,
            all_genes=all_genes,
            cell_ids=batch_cells,
            iter_num=iter_num,
            n_processes=n_processes,
            chunk_size=chunk_size,
            clustering=clustering,
            radius=radius
        )
        
        # Combine results
        for i in range(iter_num):
            all_results[i].extend(results[i])
            all_nums_combined[i].extend(all_nums[i])
        
        # Clean up
        del perturbed_data, results, all_nums, batch_raw_df
        gc.collect()
    
    return all_results, all_nums_combined




if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Run coordinate perturbation analysis.")
    parser.add_argument("--data_name", type=str, default="lung_transcripts.csv.gz")
    parser.add_argument("--output_name", type=str, default="data/lung_results_bd_noise.pkl")
    parser.add_argument("--iter_num", type=int, default=500)
    parser.add_argument("--limit_cells", type=int, default=50)
    parser.add_argument("--std", type=float, default=0.2)
    parser.add_argument("--boundary_noise", action='store_true', help="Enable boundary noise")
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--frac", type=float, default=0.90)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument(
    "--clustering",
    action="store_true", 
    help="Enable clustering"
)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    current_path = "/scratch/project_465001820/Spatialformer/downstream/Coordinate_perturbation"
    magic_path = lambda *x: os.path.join(os.path.abspath(current_path), *x)


    print("output_name:", args.output_name)
    print("data_name:", args.data_name)

    # loading the .csv path
    df = pd.read_csv(magic_path("data", args.data_name), compression='gzip') 
    # Run the optimized pipeline
    all_results, all_nums_combined = run_optimized_pipeline_precomputed_batched(
    raw_df = df,
    iter_num = args.iter_num,
    limit_cells = args.limit_cells,
    std = args.std,
    boundary_noise = True,
    quantile = args.quantile,
    frac = args.frac,
    clustering = args.clustering,
    chunk_size = 10,
    n_processes = None,
    radius = args.radius,
    batch_size = 100  # Process 100 cells at a time
    )


    pickle.dump(all_results, open(magic_path("data", args.output_name),"wb"))

