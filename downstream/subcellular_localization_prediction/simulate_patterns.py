# import simfish
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import *
from multiprocessing import Pool
from datasets import load_dataset
import scanpy as sc
from multiprocessing import Pool, Manager
# Load the AnnData object
# Load the AnnData object
# Load the AnnData object
# Load the AnnData object
adata_sub = sc.read_h5ad("/scratch/project_465001027/Spatialformer/downstream/subcellular_localization_prediction/data/fourcelltypes.h5ad")

def process_single_cell(args):
    """Process a single cell and return the result."""
    cell_exp, ct, ct_id, gene_list, cell_id = args
    try:
        gene_df = sim_single_cell(cell_exp, ct, ct_id, gene_list, cell_id)
        return gene_df
    except Exception as e:
        print(f"Error processing cell {cell_id}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def collect_results(result, results_list):
    """Collect results from the processed cell."""
    if not result.empty:  # Only append if result is not empty
        results_list.append(result)

def batch_tasks(n, batch_size):
    """Yield batches of tasks to limit the number of processes running simultaneously."""
    for i in range(0, len(n), batch_size):
        yield n[i:i + batch_size]

# Prepare tasks
tasks = [
    (
        adata_sub.layers["counts"][list(adata_sub.obs_names).index(cell_id), :],
        adata_sub.obs["Annotations"][cell_id],
        list(np.unique(adata_sub.obs["Annotations"])).index(adata_sub.obs["Annotations"][cell_id]),
        adata_sub.var_names,
        cell_id
    )
    for cell_id in adata_sub.obs_names
]

if __name__ == "__main__":
    manager = Manager()
    cell_coord = manager.list()  # Use a Manager list to share results across processes

    num_processes = 48  # Set this to the number of logical CPUs or more as required
    batch_size = 32   # You can adjust this based on your specific task's size and the load that can be handled efficiently

    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks)) as pbar:
            for batch in batch_tasks(tasks, batch_size):
                for task in batch:
                    # Use apply_async to distribute tasks effectively
                    pool.apply_async(process_single_cell, args=(task,),
                                     callback=lambda result: (collect_results(result, cell_coord), pbar.update(1)))

            pool.close()
            pool.join()

    # Check if any results were collected
    if len(cell_coord) > 0:
        # Concatenate the results into a DataFrame
        cell_df = pd.concat(cell_coord, ignore_index=True)
        cell_df.to_csv("/scratch/project_465001027/Spatialformer/downstream/subcellular_localization_prediction/data/transcripts.csv")

        print("Processing completed successfully. Data shape:", cell_df.shape)
    else:
        print("No results were collected. Please check for errors.")
    