#!/usr/bin/env python
# coding: utf-8
import zarr
import numpy as np
from shapely.geometry import Point, Polygon
from shapely import points
import dask.array as da
from typing import List, Tuple
from shapely.strtree import STRtree
from dask.distributed import Client
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
import os


class process_5k:
    def __init__(self, transcript_file: str, cell_file: str):
        self.transcript_file = transcript_file
        self.cell_file = cell_file
        self.cell_ids = None
        self.dataset_suffix = None
        self.cell_string = None
        self.ref_gene_names = None
        self.vertices = None
        self.transcripts_zarr = None
        self.cell_zarr = None
        self.tree = None
        self.polygon_indices = None  # Track polygon indices

    def load_location(self):
        """Load the transcripts.zarr.zip file as object"""
        transcripts_store = zarr.ZipStore(self.transcript_file, mode="r")
        self.transcripts_zarr = zarr.open(transcripts_store, mode='r')
        self.ref_gene_names = self.transcripts_zarr.attrs["gene_names"]

    def load_cell(self):
        """Load the cells.zarr.zip file as an object"""
        cell_store = zarr.ZipStore(self.cell_file, mode="r")
        self.cell_zarr = zarr.open(cell_store, mode='r')
        self.cell_ids = [x[0] for x in self.cell_zarr["cell_id"]]
        self.dataset_suffix = [x[1] for x in self.cell_zarr["cell_id"]]
        # print(self.dataset_suffix[:10])
        self.hex2string()
    def get_polygon(self):
        """Getting the polygon for mapping the transcripts"""
        self.vertices = self.cell_zarr["polygon_sets"][1]["vertices"]
        # Pre-build spatial index for all workers to use
        all_polys = []
        self.polygon_indices = []  # Track which polygon corresponds to which cell index
        
        for i, vertices_i in enumerate(self.vertices):
            poly = Polygon(vertices_i.reshape(-1, 2))
            all_polys.append(poly)
            self.polygon_indices.append(i)  # Store the original index
        
        self.tree = STRtree(all_polys)

    def get_location_and_gene_identity(self, chunk):
        """Load the transcript localizations in chunks"""
        # print("The chunk name and types:", chunk, type(chunk))
        # Ensure we're getting numpy arrays with proper indexing
        locations = np.array(self.transcripts_zarr["grids"]["0"][chunk]["location"])
        # print("locations:", locations.shape)
        identities = np.array(self.transcripts_zarr["grids"]["0"][chunk]["gene_identity"])[:,0]
        # print("identities:", identities.shape)
        return locations, identities

    def hex2string(self):
        """
        convert the hex to the cell_id string
        """
        hex_to_shifted = {
                "0": "a",
                "1": "b",
                "2": "c",
                "3": "d",
                "4": "e",
                "5": "f",
                "6": "g",
                "7": "h",
                "8": "i",
                "9": "j",
                "a": "k",
                "b": "l",
                "c": "m",
                "d": "n",
                "e": "o",
                "f": "p"
            }
        # print(self.cell_ids[:10])    
        pad_func = lambda x: x.rjust(8, "0")
        cell_id_prefix = [
                pad_func("".join(hex_to_shifted[str(i)] for i in str(x)))
                for x in self.cell_ids
            ]
            
        self.cell_string = [
            "".join([i, "-", str(j)])
            for i, j in zip(cell_id_prefix, self.dataset_suffix)
        ]
        

    def process_transcript_batch(self, batch_data):
        """
        Process a batch of transcripts (locations and identities)
        This will be sent to workers via client.map
        """
        locations_batch, identities_batch = batch_data
        
        # Ensure we have proper numpy arrays
        locations_batch = np.array(locations_batch)
        identities_batch = np.array(identities_batch)
        
        # VECTORIZED: Create all Point objects at once (C-level operation)
        all_points = points(locations_batch[:, 0], locations_batch[:, 1])
        
        # VECTORIZED: Bulk query returns (point_idx, polygon_idx) pairs
        # This is 10-100x faster than looping!
        query_result = self.tree.query(all_points, predicate='intersects')
        
        # query_result is a tuple: (point_indices, polygon_indices)
        point_indices, polygon_indices = query_result
        
        # Create mapping from point index to cell_id
        # Initialize all as None
        cell_ids_mapped = np.array([None] * len(locations_batch), dtype=object)
        
        # For points that intersect polygons, assign cell IDs
        # If multiple polygons intersect, we take the first one
        unique_point_indices, first_occurrence = np.unique(point_indices, return_index=True)
        
        for i, point_idx in enumerate(unique_point_indices):
            poly_idx = polygon_indices[first_occurrence[i]]
            cell_ids_mapped[point_idx] = self.cell_string[poly_idx]
        
        # VECTORIZED: Map all gene names at once using numpy indexing
        gene_names = [self.ref_gene_names[idx] for idx in identities_batch]
        
        # Create DataFrame
        batch_df = pd.DataFrame({
            "x_location": locations_batch[:, 0], 
            "y_location": locations_batch[:, 1], 
            "z_location": locations_batch[:, 2], 
            "gene": gene_names, 
            "cell_id": cell_ids_mapped
        })
        
        return batch_df

    def main_process(self, save_path: str, 
                     batch_size: int = 5000, 
                     n_workers: int = 4, 
                     max_concurrent_batches: int = 20, 
                     save_batch_size: int = 1000000,
                     partitions: int = 1,
                     partition: int = 0):
        """
        Process transcripts with precise control over saved transcript count
        """
        
        os.makedirs(os.path.join(save_path, "transcript_processed"), exist_ok=True)
        client = Client(n_workers=n_workers, threads_per_worker=1, memory_limit='20GB', processes=True)
        print(f"Dask Dashboard: {client.dashboard_link}")
        
        try:
            self.load_location()
            self.load_cell()
            self.get_polygon()




            if partitions == 1:
                all_chunks = list(self.transcripts_zarr["grids"]["0"].keys())
                print(f"Found {len(all_chunks)} chunks to process")
            
            if partitions >= 1:
                
                
                all_chunks = list(self.transcripts_zarr["grids"]["0"].keys())
                #split all chunks into partitions
                
                all_chunks = list(np.array_split(all_chunks, partitions)[partition])
                print(f"Found {len(all_chunks)} chunks to process")




            
            chunk_dfs = []
            save_chunk_index = 0
            current_save_count = 0
            
            for chunk_idx, chunk in enumerate(tqdm(all_chunks, desc="Processing chunks")):
                print(f"Processing chunk {chunk_idx}: {chunk}")
                
                try:
                    locations, identities = self.get_location_and_gene_identity(chunk)
                    total_transcripts = len(locations)
                    print(f"Chunk {chunk} has {total_transcripts} transcripts")
                    
                    # Split into batches
                    batches = []
                    for i in range(0, total_transcripts, batch_size):
                        end_idx = min(i + batch_size, total_transcripts)
                        batch_locations = locations[i:end_idx].copy()
                        batch_identities = identities[i:end_idx].copy()
                        batches.append((batch_locations, batch_identities))
                    
                    print(f"Split into {len(batches)} batches")


                    # With this:
                    # max_concurrent_batches = 20  # Process 20 batches at a time
                    
                    for batch_idx in tqdm(range(0, len(batches), max_concurrent_batches), desc=f"Run max {max_concurrent_batches} each time", total = len(batches)):
                        batch_subset = batches[batch_idx:batch_idx + max_concurrent_batches]
                        # scattered_data = client.scatter(batch_subset, broadcast=False)
                        futures = client.map(self.process_transcript_batch, batch_subset)
                        
                        for future in futures:
                            batch_df = future.result()
                            batch_size_current = len(batch_df)

                    
                    # Process batches
                    # futures = client.map(self.process_transcript_batch, batches)
                    
                    # for future in futures:
                    #     batch_df = future.result()
                    #     batch_size_current = len(batch_df)
                        
                            # If adding this batch would exceed our save threshold
                            if current_save_count + batch_size_current > save_batch_size and current_save_count > 0:
                                # Save current accumulated data
                                combined_df = pd.concat(chunk_dfs, ignore_index=True)
                                combined_df.to_parquet(
                                    f"{save_path}/transcript_processed/chunk_{partition}_{save_chunk_index:05d}.parquet", 
                                    index=False
                                )
                                print(f"Saved file {save_chunk_index} with {len(combined_df)} transcripts")
                                
                                # Start new accumulation with current batch
                                save_chunk_index += 1
                                chunk_dfs = [batch_df]
                                current_save_count = batch_size_current
                            else:
                                # Add to current accumulation
                                chunk_dfs.append(batch_df)
                                current_save_count += batch_size_current
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk}: {e}")
                    continue
            
            # Save any remaining data
            if chunk_dfs:
                combined_df = pd.concat(chunk_dfs, ignore_index=True)
                combined_df.to_parquet(
                    f"{save_path}/transcript_processed/chunk_{partition}_{save_chunk_index:05d}.parquet", 
                    index=False
                )
                print(f"Saved final file {save_chunk_index} with {len(combined_df)} transcripts")
                
        finally:
            client.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to the uncompressed xenium output file")
    parser.add_argument("--workers", type=int, default=5, help="Number of Dask workers to use")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for processing")
    parser.add_argument("--save_batch_size", type=int, default=5000, help="Batch size for saving")
    parser.add_argument("--partitions", type=int, default=1, help="The number of partitions to split the large transcript files")
    parser.add_argument("--partition", type=int, default=1, help="The order of the partition to run")
    args = parser.parse_args()
    
    
    input_file = args.input_file
    # Silence stdout/stderr of Dask workers completely
    os.environ["DASK_DISTRIBUTED__LOGGING__DISTRIBUTED"] = "error"
    os.environ["DASK_DISTRIBUTED__LOGGING__BOKEH"] = "error"
    os.environ["DASK_DISTRIBUTED__LOGGING__TQDM"] = "error"
    
    Process = process_5k(transcript_file = os.path.join(os.path.abspath(input_file), "transcripts.zarr.zip"),
          cell_file = os.path.join(os.path.abspath(input_file), "cells.zarr.zip"))

    Process.main_process(save_path = os.path.abspath(input_file), 
                     batch_size = args.batch_size,
                     max_concurrent_batches = 10, 
                     n_workers = args.workers,
                     save_batch_size = args.save_batch_size,
                     partitions = args.partitions,
                     partition = args.partition)