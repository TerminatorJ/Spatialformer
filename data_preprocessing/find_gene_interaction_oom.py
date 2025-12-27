###this is design for the dataset with large cell number, which will lead to oom issue during the processing

import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys 
import psutil
import h5py
import numpy as np
import argparse
from pathlib import Path
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]
sys.path.append(str(p_path))
sys.path.append(os.path.join(str(p_path), "utils"))
from process import KNN_Radius_Graph
import argparse
from utils import *
import logging
from multiprocessing import Pool
from datetime import datetime
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import dask
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_all_parquet(directory):
    pattern = os.path.join(directory, "*.parquet")
    return dd.read_parquet(pattern)



class GeneInteractionProcessor:
    def __init__(self, threshold, gene_threshold, gene_repeat, radius, pair_threshold, transcript_file):
        self.threshold = threshold
        self.radius = radius
        self.pair_threshold = pair_threshold
        self.ddf = None
        self.genes = None
        self.transcript_file = transcript_file
        self.gene_threshold = gene_threshold
        self.gene_repeat = gene_repeat

    def load_and_preprocess_data(self):
        input_path = Path(self.transcript_file)
        if self.transcript_file[-2:] == "gz":
            ddf = dd.read_csv(self.transcript_file, compression='gzip', blocksize='64MB')
        elif self.transcript_file[-3:] == "zarr":
            ddf = dd.read_parquet(self.transcript_file)
        elif input_path.is_dir():
            ddf = load_all_parquet(self.transcript_file)
            #fill the na with UNASSIGNED
            ddf["cell_id"] = ddf["cell_id"].fillna("UNASSIGNED")
        else:
            ddf = dd.read_csv(self.transcript_file, blocksize='64MB')
        
        #save the grids as parquet for chunks of cells
        all_cell_ids = ddf['cell_id'].unique().compute().tolist()
        cell_ids = all_cell_ids[args.chunks * (args.partition - 1): args.chunks * args.partition]
        parquet_file_path = os.path.join("/scratch/project_465001820/Spatialformer/data/raw", args.dataname, "cell_parquet_files")
        os.makedirs(parquet_file_path, exist_ok=True)


        logging.info(f"Saving {len(cell_ids)} cells to parquet partition {args.partition}...")
        
        # 1. Filter lazily
        cell_ddf = ddf[ddf['cell_id'].isin(cell_ids)]
        
        # 2. Compute to bring into local memory (Pandas DataFrame)
        #    and save as a SINGLE parquet file.
        cell_output_path = os.path.join(parquet_file_path, f"cell_chunk_{args.partition}.parquet")
        
        # Check if empty before computing to save time/errors
        # (Optional, but good practice if some chunks might have no data)
        # if len(cell_ddf.head(1)) == 0: continue 

        cell_ddf.compute().to_parquet(cell_output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate the gene-gene interaction')
    parser.add_argument('--threshold', type=int, default=30, help='the threshold of transcripts for filtering the cells')
    parser.add_argument('--gene_threshold', type=int, default=10, help='the minimum number of gene for each cells')
    parser.add_argument('--gene_repeat', type=int, default=2, help='the number of transcript for each gene')
    parser.add_argument('--radius', type=int, default=5, help='the radius to separate compartments')
    parser.add_argument('--pair_threshold', type=int, default=3, help='the pair threshold for the same transcripts and different transcripts')
    parser.add_argument('--transcript_file', type=str, default="/scratch/project_465001027/nicheformer/src/nicheformer/data/raw/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/transcripts.csv", help='the file path of the transcript')
    parser.add_argument('--partition', type=int, default=1, help='The partition of cell_id that are used to run separately')
    parser.add_argument('--chunks', type=int, default=20000, help='The number of chunks for dividing the cell_ids')
    parser.add_argument('--dataname', type=str, default=None, help='The overall name of the dataset')
    args = parser.parse_args()

    #adding the partitions information
    # h5_file_path = h5_file_path.split(".")[0] + "_" + str(args.partition) + "." + h5_file_path.split(".")[2]
    processor = GeneInteractionProcessor(args.threshold, args.gene_threshold, args.gene_repeat, args.radius, args.pair_threshold, args.transcript_file)
    processor.load_and_preprocess_data()
  