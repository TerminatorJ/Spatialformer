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
    def __init__(self, threshold, gene_threshold, gene_repeat, radius, pair_threshold, number_cell, transcript_file, h5_file_path):
        self.threshold = threshold
        self.radius = radius
        self.pair_threshold = pair_threshold
        self.number_cell = number_cell
        self.lung_annot_3D_tx_filtered = None
        self.genes = None
        self.h5_file_path = h5_file_path
        self.transcript_file = transcript_file
        self.gene_threshold = gene_threshold
        self.gene_repeat = gene_repeat

    def load_and_preprocess_data(self):
        if self.transcript_file[-2:] == "gz":
            # import pdb; pdb.set_trace()
            lung_annot_3D_tx = pd.read_csv(self.transcript_file, compression='gzip') 
        else:
            lung_annot_3D_tx = pd.read_csv(self.transcript_file)
        lung_annot_3D_tx.rename(columns={'x_location': 'x', 'y_location': 'y', 'z_location': 'z', 'feature_name': 'gene'}, inplace=True)
        
        
        #filter genes and cells level
        # import pdb; pdb.set_trace()

        #filter by gene level
        self.lung_annot_3D_tx_filtered = lung_annot_3D_tx[~(lung_annot_3D_tx['gene'].str.startswith('Neg') | lung_annot_3D_tx['gene'].str.startswith('BLANK') | lung_annot_3D_tx['gene'].str.startswith('Unassigned'))]
        #also filter out the cell with gene number less than 10
        # gene_counts = self.lung_annot_3D_tx_filtered.groupby('cell_id')['gene'].nunique().reset_index(name='unique_gene_count')
        # self.lung_annot_3D_tx_filtered = self.lung_annot_3D_tx_filtered[self.lung_annot_3D_tx_filtered['cell_id'].isin(gene_counts["cell_id"][gene_counts["unique_gene_count"] >= self.gene_threshold])]
        #get mean count of all gene in the same cell
        # genet_counts = self.lung_annot_3D_tx_filtered.groupby(['cell_id', 'gene']).size().reset_index(name='count')
        # mean_gene_count = genet_counts.groupby('cell_id')['count'].mean().reset_index(name='mean_gene_count')
        # self.lung_annot_3D_tx_filtered = self.lung_annot_3D_tx_filtered[self.lung_annot_3D_tx_filtered['cell_id'].isin(mean_gene_count["cell_id"][mean_gene_count["mean_gene_count"] >= self.gene_repeat])]

        #filter by transcript level
        # self.lung_annot_3D_tx_filtered = self.lung_annot_3D_tx_filtered[self.lung_annot_3D_tx_filtered["qv"] > 20]
        value_counts = self.lung_annot_3D_tx_filtered['cell_id'].value_counts()
        try:
            clean_value_counts = value_counts.drop("UNASSIGNED")
            self.lung_annot_3D_tx_filtered = self.lung_annot_3D_tx_filtered[self.lung_annot_3D_tx_filtered['cell_id'].isin(clean_value_counts.index[clean_value_counts >= self.threshold])]
        except:
            self.lung_annot_3D_tx_filtered = self.lung_annot_3D_tx_filtered[self.lung_annot_3D_tx_filtered['cell_id'].isin(value_counts.index[value_counts >= self.threshold])]
        # import pdb; pdb.set_trace()
        kept_cells_num = len(self.lung_annot_3D_tx_filtered['cell_id'].unique())
        
        final_value_counts = self.lung_annot_3D_tx_filtered['cell_id'].value_counts()
        # print(self.lung_annot_3D_tx_filtered['gene'].unique())
        self.genes = list(self.lung_annot_3D_tx_filtered["gene"].unique())
        # import pdb; pdb.set_trace()
        logging.info(f"The number of cells that are kept: {kept_cells_num}")
        logging.info(f"Mean transcripts per cell: {np.mean(final_value_counts)}")
        logging.info(f"Total transcripts left: {np.sum(final_value_counts)}")
        logging.info(f"Gene number after filtering: {len(self.lung_annot_3D_tx_filtered['gene'].unique())}")
        
        # Create the HDF5 file
        with h5py.File(self.h5_file_path, 'w') as f:
            pass 
        
    

def calculate_func(cell_id):
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
    parser.add_argument('--gene_threshold', type=int, default=10, help='the minimum number of gene for each cells')
    parser.add_argument('--gene_repeat', type=int, default=2, help='the number of transcript for each gene')
    parser.add_argument('--radius', type=int, default=5, help='the radius to separate compartments')
    parser.add_argument('--pair_threshold', type=int, default=3, help='the pair threshold for the same transcripts and different transcripts')
    parser.add_argument('--number_cell', type=int, default=2, help='number of cells that are used to calculate, this can be useful for debugging the codes and gene-gene pipeline')
    parser.add_argument('--transcript_file', type=str, default="/scratch/project_465001027/nicheformer/src/nicheformer/data/raw/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/transcripts.csv", help='the file path of the transcript')
    parser.add_argument('--partition', type=int, default=1, help='The partition of cell_id that are used to run separately')
    parser.add_argument('--chunks', type=int, default=20000, help='The number of chunks for dividing the cell_ids')
    parser.add_argument('--dataname', type=str, default=None, help='The overall name of the dataset')
    parser.add_argument('--datapath', type=str, default="/tmp/erda/Spatialformer/downloaded_data/processed", help='The data path that is used to store all the raw and processed dataset')
    args = parser.parse_args()
    data_path = args.datapath

    os.makedirs(data_path, exist_ok=True)
    h5_file_path = os.path.join(data_path, f"{args.dataname}_gene_interaction_{datetime.now()}.h5")
    #adding the partitions information
    
    h5_file_path = h5_file_path.split(".")[0] + "_" + str(args.partition) + "." + h5_file_path.split(".")[2]
    processor = GeneInteractionProcessor(args.threshold, args.gene_threshold, args.gene_repeat, args.radius, args.pair_threshold, args.number_cell, args.transcript_file, h5_file_path)
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

    


#for the new downloaded dataset

# python find_gene_interaction.py --transcript_file /tmp/erda/Spatialformer/downloaded_data/raw/Xenium_V1_hBoneMarrow_nondiseased_section_outs/transcripts.csv.gz --number_cell 84518 --partition 1 --dataname Xenium_50

#for simulated data
# python find_gene_interaction.py --transcript_file /home/sxr280/Spatialformer/downstream/subcellular_localization_prediction/data/transcripts_mod.csv --number_cell 10000 --partition 1 --dataname simulation --radius 10 --datapath /home/sxr280/Spatialformer/downstream/subcellular_localization_prediction/data/


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