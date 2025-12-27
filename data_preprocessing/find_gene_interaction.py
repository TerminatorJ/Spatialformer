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


def load_all_parquet(directory, partition=None):
    if partition:
        pattern = os.path.join(directory, f"cell_chunk_{partition}.parquet")
    else:
        pattern = os.path.join(directory, "*.parquet")
    return dd.read_parquet(pattern)



class GeneInteractionProcessor:
    def __init__(self, threshold, gene_threshold, gene_repeat, radius, pair_threshold, transcript_file, h5_file_path):
        self.threshold = threshold
        self.radius = radius
        self.pair_threshold = pair_threshold
        self.ddf = None
        self.genes = None
        self.h5_file_path = h5_file_path
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
            if args.dataname != "Xenium_Prime_Human_Ovary_FF_xe_outs":
                ddf = load_all_parquet(self.transcript_file)
            else:
                ddf = load_all_parquet(self.transcript_file, partition=args.parquet_partition)
            #fill the na with UNASSIGNED
            ddf["cell_id"] = ddf["cell_id"].fillna("UNASSIGNED")
        else:
            ddf = dd.read_csv(self.transcript_file, blocksize='64MB')
        # Rename columns
        ddf = ddf.rename(columns={
            'x_location': 'x', 
            'y_location': 'y', 
            'z_location': 'z', 
            'feature_name': 'gene'
        })

        # Filter by gene level
        logging.info("Filtering genes...")
        ddf = ddf[~ddf['gene'].str.startswith(('Neg', 'BLANK', 'Unassigned', 'Deprecated', 'Intergenic'))]
 
        # OPTIMIZATION 2: Only select needed columns early
        ddf = ddf[['x', 'y', 'z', 'gene', 'cell_id']]
        # Compute value counts for filtering (small object)
        logging.info("Computing cell transcript counts...")
        with ProgressBar():
            logging.info("Calculating the number of transcripts for each cell...")
            value_counts = ddf['cell_id'].value_counts().compute()

        # Filter cells by threshold
        try:
            clean_value_counts = value_counts.drop("UNASSIGNED")
            valid_cells = clean_value_counts.index[clean_value_counts >= self.threshold]
        except:
            valid_cells = value_counts.index[value_counts >= self.threshold]

        logging.info(f"Filtering cells with threshold >= {self.threshold}...")
        # Convert to set for O(1) lookup
        valid_cells_set = set(valid_cells)
        ddf = ddf[ddf['cell_id'].isin(valid_cells_set)]

        # Store filtered Dask DataFrame (no .compute())
        self.ddf = ddf
        
        # Compute statistics efficiently
        with ProgressBar():
            logging.info("Calculating gene and transcript stats...")
            # Use parallel computation
            kept_cells_num, genes = dask.compute(
                ddf['cell_id'].nunique(),
                ddf["gene"].unique()
            )
            self.genes = genes.tolist()
            final_value_counts = value_counts[value_counts.index.isin(valid_cells_set)]

        logging.info(f"The number of cells that are kept: {kept_cells_num}")
        logging.info(f"Mean transcripts per cell: {np.mean(final_value_counts)}")
        logging.info(f"Total transcripts left: {np.sum(final_value_counts)}")
        logging.info(f"Gene number after filtering: {len(self.genes)}")
        
        # Create the HDF5 file
        with h5py.File(self.h5_file_path, 'w') as f:
            pass
        

def calculate_func(params):
    df = params[0]
    cell_id = params[1]
    try:
        data_graph = KNN_Radius_Graph(radius=radius, dataset=df, is_3D=True, cell_ID=cell_id, ref_gene=genes)
        gene_binary_matrix, gene_freq_matrix, trans_matrix = data_graph.get_gene_matrix(pair_threshold=pair_threshold, self_threshold=pair_threshold, plot=False)
        coo_matrix = binary_to_coo_matrix(gene_binary_matrix)
        pair_num = coo_matrix.nnz / 2  # Since it's symmetric, divide by 2
        
        del data_graph
        del gene_binary_matrix
        del gene_freq_matrix
        del trans_matrix
        del df  # Delete input dataframe
        gc.collect()
        return (cell_id, coo_matrix, pair_num)

    except Exception as e:
        logging.error(f"Error processing cell_id {cell_id}: {e}")
        return (cell_id, None, None)

def write_to_hdf5(results, h5_file_path):
    with h5py.File(h5_file_path, 'a') as f:
        for cell_id, coo_matrix, pair_num in results:
            if coo_matrix is not None:
                grp = f.create_group(str(cell_id))
                grp.create_dataset('data', data=coo_matrix.data, compression='gzip', compression_opts=1)
                grp.create_dataset('row', data=coo_matrix.row, compression='gzip', compression_opts=1)
                grp.create_dataset('col', data=coo_matrix.col, compression='gzip', compression_opts=1)
                grp.attrs['shape'] = coo_matrix.shape
                
def get_total_memory():
    """Get total memory used by this job (main + all workers)"""
    current_process = psutil.Process(os.getpid())
    
    # Get memory of main process
    total_mem = current_process.memory_info().rss
    
    # Add memory of all child processes
    children = current_process.children(recursive=True)
    for child in children:
        try:
            total_mem += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return total_mem / (1024**2)  # Convert to MB


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate the gene-gene interaction')
    parser.add_argument('--threshold', type=int, default=30, help='the threshold of transcripts for filtering the cells')
    parser.add_argument('--gene_threshold', type=int, default=10, help='the minimum number of gene for each cells')
    parser.add_argument('--gene_repeat', type=int, default=2, help='the number of transcript for each gene')
    parser.add_argument('--radius', type=int, default=5, help='the radius to separate compartments')
    parser.add_argument('--pair_threshold', type=int, default=3, help='the pair threshold for the same transcripts and different transcripts')
    parser.add_argument('--transcript_file', type=str, default="/scratch/project_465001027/nicheformer/src/nicheformer/data/raw/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/transcripts.csv", help='the file path of the transcript')
    parser.add_argument('--partition', type=int, default=1, help='The partition of cell_id that are used to run separately')
    parser.add_argument('--parquet_partition', type=int, default=1, help='The partition of cell_id that are used to run separately for the large dataset')
    parser.add_argument('--chunks', type=int, default=20000, help='The number of chunks for dividing the cell_ids')
    parser.add_argument('--dataname', type=str, default=None, help='The overall name of the dataset')
    args = parser.parse_args()
    erda_path = "/scratch/project_465001820/Spatialformer/data/processed"
    # erda_path = "/projects/sc_clip/data/spatialformer_gco"

    os.makedirs(erda_path, exist_ok=True)
    
    h5_file_path = os.path.join(erda_path, f"{args.dataname}_gene_interaction_{datetime.now()}.h5")
    
    h5_file_path_base = h5_file_path.rsplit(".", 1)[0]  # Everything before .h5
    h5_file_path = f"{h5_file_path_base}_{args.partition}.h5"
    
    global radius
    global pair_threshold
    global genes
    #adding the partitions information
    # h5_file_path = h5_file_path.split(".")[0] + "_" + str(args.partition) + "." + h5_file_path.split(".")[2]
    processor = GeneInteractionProcessor(args.threshold, args.gene_threshold, args.gene_repeat, args.radius, args.pair_threshold, args.transcript_file, h5_file_path)
    processor.load_and_preprocess_data()
    ddf_flt = processor.ddf
    genes = processor.genes
    del processor
    gc.collect()
    radius = args.radius
    pair_threshold = args.pair_threshold

    
    
    # Get cell IDs
    with ProgressBar():
        logging.info("Computing all cell IDs...")
        all_cell_ids = ddf_flt['cell_id'].unique().compute().tolist()
    
    logging.info(f"Total partitions needed: {len(all_cell_ids)//args.chunks + 1}")
    
    cell_ids = all_cell_ids[args.chunks * (args.partition - 1): args.chunks * args.partition]
    logging.info(f"Pre-filtering {len(cell_ids)} cells...")

    if not cell_ids:
        logging.info("No cells in this partition. Exiting.")
        sys.exit(0)


    MINI_BATCH_SIZE = 200  # Process 200 cells at a time
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    logging.info(f"Using {n_cpus} workers")
    
    results = []
    pairs_num = []
    
    # Split cell_ids into mini-batches
    for mini_batch_idx in range(0, len(cell_ids), MINI_BATCH_SIZE):
        mini_batch_cells = cell_ids[mini_batch_idx:mini_batch_idx + MINI_BATCH_SIZE]
        
        logging.info(f"Processing mini-batch {mini_batch_idx//MINI_BATCH_SIZE + 1}/{len(cell_ids)//MINI_BATCH_SIZE + 1} ({len(mini_batch_cells)} cells)...")
        
        # OPTIMIZATION 5: Load only THIS mini-batch into pandas
        with ProgressBar():
            df_pandas = ddf_flt[ddf_flt['cell_id'].isin(mini_batch_cells)].compute()
        
        # Convert to dict
        cell_dict = {cell_id: group.reset_index(drop=True) 
                     for cell_id, group in df_pandas.groupby('cell_id', observed=True)}
        
        # Prepare tasks for this mini-batch
        tasks = [
            (cell_dict[cell_id], cell_id)
            for cell_id in mini_batch_cells if cell_id in cell_dict
        ]
        
        # Free memory immediately
        del df_pandas
        del cell_dict
        gc.collect()
        
        # Process with multiprocessing
        with Pool(processes=n_cpus) as pool:
            # chunksize = max(1, len(tasks) // (n_cpus * 4))
            for result in tqdm(pool.imap_unordered(calculate_func, tasks), 
                             total=len(tasks), desc=f"Mini-batch {mini_batch_idx//MINI_BATCH_SIZE + 1}"):
                
                
                total_mem_mb = get_total_memory()
                mem_str = f"{total_mem_mb/1024:.2f}G" if total_mem_mb > 1024 else f"{total_mem_mb:.1f}M"
                logging.info(f"Processed {len(results)} cells, Total memory: {mem_str}")
                if result[2] is not None:
                    pairs_num.append(result[2])
                    results.append(result)
                
                # Write every 1000 results
                if len(results) >= 500:
                    write_to_hdf5(results, h5_file_path)
                    results = []
                    gc.collect()
        
        # Force garbage collection between mini-batches
        gc.collect()
    
    # Write remaining results
    if results:
        write_to_hdf5(results, h5_file_path)
    
    logging.info("Processing completed")
    if pairs_num:
        logging.info(f"Mean pairs: {np.mean(pairs_num):.4f}, Median: {np.median(pairs_num)}")


#for the new downloaded dataset

# python find_gene_interaction.py --transcript_file /tmp/erda/Spatialformer/downloaded_data/raw/Xenium_V1_Human_Ovary_Cancer_FF_xe_outs/transcript_processed --number_cell 200900 --partition 1 --dataname Xenium_V1_Human_Ovary_Cancer_FF_xe_outs



#python find_gene_interaction.py --number_cell 113460 --partition 6
#testing the david dataset
#for THD0008: 3 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/outs/transcripts.csv --number_cell 57889 --partition 1 --dataname THD0008
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/outs/transcripts.csv --number_cell 57889 --partition 2 --dataname THD0008
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/outs/transcripts.csv --number_cell 57889 --partition 3 --dataname THD0008


#for VUILD106: 6 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD106__20230313__191400/outs/transcripts.csv --number_cell 105595 --partition 1 --dataname VUILD106
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD106__20230313__191400/outs/transcripts.csv --number_cell 105595 --partition 2 --dataname VUILD106
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD106__20230313__191400/outs/transcripts.csv --number_cell 105595 --partition 3 --dataname VUILD106
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD106__20230313__191400/outs/transcripts.csv --number_cell 105595 --partition 4 --dataname VUILD106
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD106__20230313__191400/outs/transcripts.csv --number_cell 105595 --partition 5 --dataname VUILD106
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD106__20230313__191400/outs/transcripts.csv --number_cell 105595 --partition 6 --dataname VUILD106

#for VUILD110: 6 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD110__20230313__191400/outs/transcripts.csv --number_cell 106851 --partition 1 --dataname VUILD110
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD110__20230313__191400/outs/transcripts.csv --number_cell 106851 --partition 2 --dataname VUILD110
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD110__20230313__191400/outs/transcripts.csv --number_cell 106851 --partition 3 --dataname VUILD110
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD110__20230313__191400/outs/transcripts.csv --number_cell 106851 --partition 4 --dataname VUILD110
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD110__20230313__191400/outs/transcripts.csv --number_cell 106851 --partition 5 --dataname VUILD110
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD110__20230313__191400/outs/transcripts.csv --number_cell 106851 --partition 6 --dataname VUILD110

#for VUILD115: 4 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD115__20230313__191400/outs/transcripts.csv --number_cell 68718 --partition 1 --dataname VUILD115
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD115__20230313__191400/outs/transcripts.csv --number_cell 68718 --partition 2 --dataname VUILD115
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD115__20230313__191400/outs/transcripts.csv --number_cell 68718 --partition 3 --dataname VUILD115
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD115__20230313__191400/outs/transcripts.csv --number_cell 68718 --partition 4 --dataname VUILD115

#for THD0011: 1 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__THD0011__20230313__191400/outs/transcripts.csv --number_cell 14372 --partition 1 --dataname THD0011

#for TILD117LF: 2 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__TILD117LF__20230313__191400/outs/transcripts.csv --number_cell 33699 --partition 1 --dataname TILD117LF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__TILD117LF__20230313__191400/outs/transcripts.csv --number_cell 33699 --partition 2 --dataname TILD117LF

#for TILD117MF: 3 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__TILD117MF__20230313__191400/outs/transcripts.csv --number_cell 46075 --partition 1 --dataname TILD117MF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__TILD117MF__20230313__191400/outs/transcripts.csv --number_cell 46075 --partition 2 --dataname TILD117MF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__TILD117MF__20230313__191400/outs/transcripts.csv --number_cell 46075 --partition 3 --dataname TILD117MF

#for TILD175: 2 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__TILD175__20230313__191400/outs/transcripts.csv --number_cell 32849 --partition 1 --dataname TILD175
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__TILD175__20230313__191400/outs/transcripts.csv --number_cell 32849 --partition 2 --dataname TILD175

#for VUILD78LF: 1 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__VUILD78LF__20230313__191400/outs/transcripts.csv --number_cell 16292 --partition 1 --dataname VUILD78LF

#for VUILD78MF: 1 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__VUILD78MF__20230313__191400/outs/transcripts.csv --number_cell 17491 --partition 1 --dataname VUILD78MF

#for VUILD91LF: 1 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__VUILD91LF__20230313__191400/outs/transcripts.csv --number_cell 15232 --partition 1 --dataname VUILD91LF


#for VUILD91MF:  2 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__VUILD91MF__20230313__191400/outs/transcripts.csv --number_cell 23599 --partition 1 --dataname VUILD91MF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__VUILD91MF__20230313__191400/outs/transcripts.csv --number_cell 23599 --partition 2 --dataname VUILD91MF

#for VUHD069:  1 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUHD069__20230308__003731/outs/transcripts.csv --number_cell 16840 --partition 1 --dataname VUHD069

#for VUHD095:  1 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUHD095__20230308__003731/outs/transcripts.csv --number_cell 7875 --partition 1 --dataname VUHD095

#for VUHD113: 1 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUHD113__20230308__003731/outs/transcripts.csv --number_cell 11746 --partition 1 --dataname VUHD113

#for VUILD48MF: 2 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUILD48MF__20230308__003731/outs/transcripts.csv --number_cell 26485 --partition 1 --dataname VUILD48MF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUILD48MF__20230308__003731/outs/transcripts.csv --number_cell 26485 --partition 2 --dataname VUILD48MF

#for VUILD104LF: 2 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUILD104LF__20230308__003731/outs/transcripts.csv --number_cell 28243 --partition 1 --dataname VUILD104LF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUILD104LF__20230308__003731/outs/transcripts.csv --number_cell 28243 --partition 2 --dataname VUILD104LF

#for VUILD105MF: 1 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUILD105MF__20230308__003731/outs/transcripts.csv --number_cell 17434 --partition 1 --dataname VUILD105MF

#for VUHD116A: 1 partition
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUHD116A__20230308__003730/outs/transcripts.csv --number_cell 10914 --partition 1 --dataname VUHD116A

#for VUHD116B: 2 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUHD116B__20230308__003731/outs/transcripts.csv --number_cell 22671 --partition 1 --dataname VUHD116B
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUHD116B__20230308__003731/outs/transcripts.csv --number_cell 22671 --partition 2 --dataname VUHD116B

#for VUILD96LF: 3 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD96LF__20230308__003730/outs/transcripts.csv --number_cell 41156 --partition 1 --dataname VUILD96LF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD96LF__20230308__003730/outs/transcripts.csv --number_cell 41156 --partition 2 --dataname VUILD96LF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD96LF__20230308__003730/outs/transcripts.csv --number_cell 41156 --partition 3 --dataname VUILD96LF

#for VUILD96MF:  3 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD96MF__20230308__003730/outs/transcripts.csv --number_cell 50504 --partition 1 --dataname VUILD96MF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD96MF__20230308__003730/outs/transcripts.csv --number_cell 50504 --partition 2 --dataname VUILD96MF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD96MF__20230308__003730/outs/transcripts.csv --number_cell 50504 --partition 3 --dataname VUILD96MF


#for VUILD102LF: 2 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD102LF__20230308__003731/outs/transcripts.csv --number_cell 26017 --partition 1 --dataname VUILD102LF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD102LF__20230308__003731/outs/transcripts.csv --number_cell 26017 --partition 2 --dataname VUILD102LF

#for VUILD102MF: 2 partitions
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD102MF__20230308__003730/outs/transcripts.csv --number_cell 33247 --partition 1 --dataname VUILD102MF
#python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD102MF__20230308__003730/outs/transcripts.csv --number_cell 33247 --partition 2 --dataname VUILD102MF

#for VUILD107MF: 4 partitions
#python find_gene_interaction.py --transcript_file  /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD107MF__20230308__003731/outs/transcripts.csv --number_cell 60373 --partition 1 --dataname VUILD107MF
#python find_gene_interaction.py --transcript_file  /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD107MF__20230308__003731/outs/transcripts.csv --number_cell 60373 --partition 2 --dataname VUILD107MF
#python find_gene_interaction.py --transcript_file  /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD107MF__20230308__003731/outs/transcripts.csv --number_cell 60373 --partition 3 --dataname VUILD107MF
#python find_gene_interaction.py --transcript_file  /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD107MF__20230308__003731/outs/transcripts.csv --number_cell 60373 --partition 4 --dataname VUILD107MF