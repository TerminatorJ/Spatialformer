#This preprocess make the large panel possible to be include in our experiment


import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import dask.dataframe as dd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import json

def filledict(args):
    parquet_file, chunk_index = args
    reader = pq.ParquetFile(parquet_file)
    chunk_df = reader.read_row_group(chunk_index).to_pandas()
    
    out_dict = {}
    for cell_id, group_df in chunk_df.groupby('cell_id'):
        group_array = group_df.to_numpy() 
        # group_array = group_array.tolist()
        if cell_id not in out_dict:
            out_dict[cell_id] = []
        out_dict[cell_id].append(group_array)

    return out_dict

def process_batch(cell_id_dfs):
    """Process a batch of (cell_id, group_array) tuples."""
    result = {}
    for item in cell_id_dfs:
        cell_id, group_array = item[0], item[1]
        if group_array:  # Check if there's data to process
            concatenated_array = np.concatenate(group_array, axis=0)
            result[cell_id] = concatenated_array
    return result

def save_to_csv(save_dict, chunk_num, output_directory, column_names):
    """Save a dictionary of DataFrames to a CSV file."""
    if save_dict:
        # import pdb; pdb.set_trace()
        concatenated_array = np.concatenate(list(save_dict.values()), axis=0)
        save_df = pd.DataFrame(concatenated_array, columns=column_names)
        file_path = os.path.join(output_directory, f'transcripts_{chunk_num}.parquet')
        # save_df.to_csv(file_path, index=False)
        save_df.to_parquet(file_path, index=False)
        save_df.to_feather(file_path)
def process_with_executor(function, args):
    cell_id_dfs = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit tasks to the executor pool
        futures = {executor.submit(function, arg): arg for arg in args}

        # Use tqdm with as_completed to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Chunks"):
            result = future.result()
            for cell_id, group_array in result.items():
                if cell_id not in cell_id_dfs:
                    cell_id_dfs[cell_id] = []

            
                cell_id_dfs[cell_id].extend(group_array)

    return cell_id_dfs
def process_and_save(batch, chunk_num, output_directory, column_names):
    result_dict = process_batch(batch)
    save_to_csv(result_dict, chunk_num, output_directory, column_names)
def splitparquet(file_name):
    parquet_file = f"/tmp/erda/Spatialformer/downloaded_data/raw/{file_name}/transcripts.parquet"
    output_directory = f"/tmp/erda/Spatialformer/downloaded_data/raw/{file_name}"
    os.makedirs(output_directory, exist_ok=True)

    reader = pq.ParquetFile(parquet_file)
    exp_df = reader.read_row_group(0).to_pandas()
    column_names = exp_df.columns
    total_chunks = reader.num_row_groups
    # valid_indices = range(total_chunks)
    args = [(parquet_file, i) for i in range(total_chunks)]

    cell_id_dfs = process_with_executor(filledict, args)
    # import pdb; pdb.set_trace()
    print(f"Total cell_ids processed: {len(cell_id_dfs)}")
    # import pdb; pdb.set_trace()
    chunk_size = 1000
    chunk_num = 0
    total_cells = len(cell_id_dfs)
    cell_items = list(cell_id_dfs.items())
    # for batch in tqdm(range(0,total_cells, chunk_size), desc="Processing storage"):
    #     import pdb; pdb.set_trace()
    #     chunk_num += 1
    #     result_dict = process_batch(cell_items[batch*chunk_size: min((batch+1)*chunk_size, total_cells)])
    #     save_to_csv(result_dict, chunk_num, output_directory, column_names)
    num_batches = (total_cells + chunk_size - 1) // chunk_size
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * chunk_size
            end_idx = min((batch_idx + 1) * chunk_size, total_cells)
            batch_data = cell_items[start_idx:end_idx]
            chunk_num += 1
            futures.append(executor.submit(process_and_save, batch_data, chunk_num, output_directory, column_names))
        
        for future in tqdm(futures, desc="Processing storage", total=num_batches):
            future.result()  # Wait for all submitted tasks to complete
       
    
   
    
file_names = [
    # "Xenium_Prime_Human_Ovary_FF_outs",
    "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
    "Xenium_Prime_Cervical_Cancer_FFPE_outs",
    "Xenium_Prime_Human_Skin_FFPE_outs",
    "Xenium_Prime_Human_Prostate_FFPE_outs",
    "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs"
]

for file_name in tqdm(file_names):
    print(f"running - {file_name}")
    splitparquet(file_name)