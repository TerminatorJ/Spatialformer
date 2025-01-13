import os 
import sys 
import h5py
import pandas as pd
import numpy as np
from multiprocessing import Pool
import multiprocessing
import argparse
import itertools
from pathlib import Path
import random
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]
sys.path.append("p_path")
sys.path.append(os.path.join(p_path, "utils"))
from process import KNN_Radius_Graph
import pickle
import argparse
from utils import *
import logging
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeneInteractionProcessor:
    '''
    A class to process gene interaction data, filtering and preparing it 
    for analysis of cell-cell interactions based on specified thresholds and 
    parameters.
    Attributes:
    ----------
    threshold : int
        The minimum number of transcripts in a cell required for a cell to be considered in the analysis.
    h5_file_path : str
        The file path to save the processed data in HDF5 format.
    transcript_file : str
        The path to the transcript CSV file (can be gzipped). This file originally from the Xenium platform.

    Methods:
    -------
    load_and_preprocess_data():
        Loads the transcript data and performs preprocessing, including filtering based on gene expression 
        thresholds and logging relevant statistics.
    '''

    def __init__(self, threshold, transcript_file, h5_file_path):
        '''
        Initializes the GeneInteractionProcessor with specified parameters.
        Parameters:
        ----------
        threshold : int
            Minimum transcript count threshold for keeping cells.
        transcript_file : str
            Path to the CSV file containing transcript data.
        h5_file_path : str
            Path to save the processed data as an HDF5 file.
        '''
        self.threshold = threshold
        self.lung_annot_3D_tx_filtered = None
        self.genes = None
        self.h5_file_path = h5_file_path
        self.transcript_file = transcript_file

    def load_and_preprocess_data(self):
        """
        Loads and preprocesses the transcript data from the specified file.

        This method reads the transcript file, filters the data based on expression thresholds, 
        and performs any necessary transformations. It logs relevant statistics regarding the 
        number of cells and genes retained after filtering. The processed DataFrame is stored in 
        the lung_annot_3D_tx_filtered attribute, and an empty HDF5 file is created at the specified path.

        Raises:
        ------
        FileNotFoundError:
            If the transcript file cannot be found.
        ValueError:
            If the filtering criteria do not match available data.
        """
        if self.transcript_file[-2:] == "gz":
            # import pdb; pdb.set_trace()
            lung_annot_3D_tx = pd.read_csv(self.transcript_file, compression='gzip') 
        else:
            lung_annot_3D_tx = pd.read_csv(self.transcript_file)
        lung_annot_3D_tx.rename(columns={'x_location': 'x', 'y_location': 'y', 'z_location': 'z', 'feature_name': 'gene'}, inplace=True)

        #filter by gene level
        self.lung_annot_3D_tx_filtered = lung_annot_3D_tx[~(lung_annot_3D_tx['gene'].str.startswith('Neg') | lung_annot_3D_tx['gene'].str.startswith('BLANK') | lung_annot_3D_tx['gene'].str.startswith('Unassigned'))]

        value_counts = self.lung_annot_3D_tx_filtered['cell_id'].value_counts()
        try:
            clean_value_counts = value_counts.drop("UNASSIGNED")
            self.lung_annot_3D_tx_filtered = self.lung_annot_3D_tx_filtered[self.lung_annot_3D_tx_filtered['cell_id'].isin(clean_value_counts.index[clean_value_counts >= self.threshold])]
        except:
            self.lung_annot_3D_tx_filtered = self.lung_annot_3D_tx_filtered[self.lung_annot_3D_tx_filtered['cell_id'].isin(value_counts.index[value_counts >= self.threshold])]
        kept_cells_num = len(self.lung_annot_3D_tx_filtered['cell_id'].unique())
        
        final_value_counts = self.lung_annot_3D_tx_filtered['cell_id'].value_counts()
        # print(self.lung_annot_3D_tx_filtered['gene'].unique())
        self.genes = list(self.lung_annot_3D_tx_filtered["gene"].unique())
        logging.info(f"The number of cells that are kept: {kept_cells_num}")
        logging.info(f"Mean transcripts per cell: {np.mean(final_value_counts)}")
        logging.info(f"Total transcripts left: {np.sum(final_value_counts)}")
        logging.info(f"Gene number after filtering: {len(self.lung_annot_3D_tx_filtered['gene'].unique())}")
        
        # Create the HDF5 file
        with h5py.File(self.h5_file_path, 'w') as f:
            pass 
        
    

def calculate_func(cell_id):
    """
    Calculate the K-Nearest Neighbors (KNN) graph and gene matrices for a given cell ID.

    This function constructs a KNN radius graph using the specified cell ID and extracts gene binary 
    and frequency matrices, as well as a coordinate matrix. 

    Parameters:
    ----------
    cell_id : str
        The identifier of the cell for which the KNN graph is to be calculated.

    Returns:
    -------
    tuple
        A tuple containing the cell ID, the coordinate matrix, and the total number of gene pairs 
        (divided by two) if successful; otherwise, a tuple of (cell_id, None) on failure.

    Raises:
    ------
    Exception:
        Logs an error if processing the specified cell ID fails.
    """
    try:
        data_graph = KNN_Radius_Graph(radius=radius, dataset=lung_annot_3D_tx_filtered, is_3D=True, cell_ID=cell_id, ref_gene=genes)
        gene_binary_matrix, gene_freq_matrix, trans_matrix = data_graph.get_gene_matrix(pair_threshold=pair_threshold, self_threshold=pair_threshold, plot=False)
        coo_matrix = binary_to_coo_matrix(gene_binary_matrix)
        pair_num = coo_matrix.toarray().sum()/2
        return (cell_id, coo_matrix, pair_num)

    except Exception as e:
        logging.error(f"Error processing cell_id {cell_id}: {e}")
        return (cell_id, None)

def write_to_hdf5(results, h5_file_path):
    """
    Write results to an HDF5 file.

    This function saves the coordinate matrix and associated data for each cell ID in the specified 
    HDF5 file path. Each cell ID has its own group within the file, containing datasets for the 
    coordinates and their shape.

    Parameters:
    ----------
    results : list of tuples
        A list where each tuple contains a cell ID, a coordinate matrix, and the number of gene pairs.
    h5_file_path : str
        The file path where the HDF5 file will be created or appended.

    Returns:
    -------
    None

    Raises:
    ------
    IOError:
        If there is an error opening or writing to the specified HDF5 file.
    """
    with h5py.File(h5_file_path, 'a') as f:
        for cell_id, coo_matrix, pair_num in results:
            if coo_matrix is not None:
                grp = f.create_group(str(cell_id))
                grp.create_dataset('data', data=coo_matrix.data)
                grp.create_dataset('row', data=coo_matrix.row)
                grp.create_dataset('col', data=coo_matrix.col)
                grp.attrs['shape'] = coo_matrix.shape
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate the gene-gene interaction')
    parser.add_argument('--threshold', type=int, default=30, help='the threshold of transcripts for filtering the cells')
    parser.add_argument('--radius', type=int, default=5, help='the radius to separate compartments')
    parser.add_argument('--pair_threshold', type=int, default=3, help='the pair threshold for the same transcripts and different transcripts')
    parser.add_argument('--number_cell', type=int, default=2, help='number of cells that are used to calculate, this can be useful for debugging the codes and gene-gene pipeline')
    parser.add_argument('--transcript_file', type=str, default="/scratch/project_465001027/nicheformer/src/nicheformer/data/raw/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/transcripts.csv", help='the file path of the transcript')
    parser.add_argument('--partition', type=int, default=1, help='The partition of cell_id that are used to run separately')
    parser.add_argument('--chunks', type=int, default=20000, help='The number of chunks for dividing the cell_ids')
    parser.add_argument('--dataname', type=str, default=None, help='The overall name of the dataset')
    args = parser.parse_args()
    erda_path = "/tmp/erda/Spatialformer/downloaded_data/processed"

    os.makedirs(erda_path, exist_ok=True)
    h5_file_path = os.path.join(erda_path, f"{args.dataname}_gene_interaction_{datetime.now()}.h5")
    #adding the partitions information
    
    h5_file_path = h5_file_path.split(".")[0] + "_" + str(args.partition) + "." + h5_file_path.split(".")[2]
    processor = GeneInteractionProcessor(args.threshold, args.transcript_file, h5_file_path)
    processor.load_and_preprocess_data()
    global lung_annot_3D_tx_filtered
    global radius
    global pair_threshold
    global genes
    
    #fetch parameters from the class
    lung_annot_3D_tx_filtered = processor.lung_annot_3D_tx_filtered
    radius = processor.radius
    pair_threshold = processor.pair_threshold
    genes = processor.genes

    cell_ids = list(lung_annot_3D_tx_filtered['cell_id'].unique())
    #calculating how many partitions you need
    logging.info(f"total partitions you need are: {len(cell_ids)//args.chunks + 1}")
    
    # import pdb; pdb.set_trace()
    cell_ids = random.sample(cell_ids, args.number_cell) if args.number_cell < 10 else cell_ids[:args.number_cell]
    cell_ids = cell_ids[args.chunks * (args.partition - 1): args.chunks * args.partition]
    # cell_ids = cell_ids[:2]
    batch_size = 200
    input_batches = [cell_ids[i:i + batch_size] for i in range(0, len(cell_ids), batch_size)]
    results = []
    pairs_num = []
    # with Pool(processes=multiprocessing.cpu_count()) as pool:
    with Pool(processes=64) as pool:
        for batch in tqdm(input_batches):
            result = list(pool.imap_unordered(calculate_func, batch))
            pairs_num.extend([i[2] for i in result])
            results.extend(result)
    # import pdb; pdb.set_trace()
    #get the mean and median number of gene pairs
    mean_pair = np.mean(pairs_num)
    median_pair = np.median(pairs_num)
    logging.info(f"mean number of the pairs is: {mean_pair:.4f}")
    logging.info(f"median number of the pairs is: {median_pair}")
    # import pdb; pdb.set_trace()
    # Write results to HDF5 file
    write_to_hdf5(results, h5_file_path)
    # Handle results after all jobs are done
    failure_count = sum(1 for result in results if result[1] == None)
    success_count = len(results) - failure_count
    logging.info(f"Processing completed. Success: {success_count}, Failure: {failure_count}")

    
