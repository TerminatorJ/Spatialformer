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
from datasets import DatasetDict, load_dataset, concatenate_datasets, Dataset
import gzip
import shutil
import glob
import json
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeneInteractionProcessor:
    def __init__(self, threshold, radius, pair_threshold, transcript_file, h5_file_path):
        self.threshold = threshold
        self.radius = radius
        self.pair_threshold = pair_threshold
        self.number_cell = None
        self.lung_annot_3D_tx_filtered = None
        self.genes = None
        self.h5_file_path = h5_file_path
        self.transcript_file = transcript_file

    def load_and_preprocess_data(self):
        lung_annot_3D_tx = pd.read_csv(self.transcript_file)
        lung_annot_3D_tx.rename(columns={'x_location': 'x', 'y_location': 'y', 'z_location': 'z', 'feature_name': 'gene'}, inplace=True)
        
        value_counts = lung_annot_3D_tx['cell_id'].value_counts()
        clean_value_counts = value_counts.drop("UNASSIGNED")
        #filter genes and cells level
        self.lung_annot_3D_tx_filtered = lung_annot_3D_tx[lung_annot_3D_tx['cell_id'].isin(clean_value_counts.index[clean_value_counts >= self.threshold])]
        self.lung_annot_3D_tx_filtered = self.lung_annot_3D_tx_filtered[~(self.lung_annot_3D_tx_filtered['gene'].str.startswith('Neg') | self.lung_annot_3D_tx_filtered['gene'].str.startswith('BLANK') | self.lung_annot_3D_tx_filtered['gene'].str.startswith('Unassigned'))]
        #filter by the transcripts level
        self.genes = list(self.lung_annot_3D_tx_filtered["gene"].unique())
        self.lung_annot_3D_tx_filtered = self.lung_annot_3D_tx_filtered[self.lung_annot_3D_tx_filtered["qv"] > 20]
        kept_cells_num = len(self.lung_annot_3D_tx_filtered['cell_id'].unique())
        self.number_cell = kept_cells_num
        final_value_counts = self.lung_annot_3D_tx_filtered['cell_id'].value_counts()
        # print(self.lung_annot_3D_tx_filtered['gene'].unique())
        
        logging.info(f"The number of cells that are kept: {kept_cells_num}")
        logging.info(f"Mean transcripts per cell: {np.mean(final_value_counts)}")
        logging.info(f"Total transcripts left: {np.sum(final_value_counts)}")
        logging.info(f"Gene number after filtering: {len(self.lung_annot_3D_tx_filtered['gene'].unique())}")
        
        # Create the HDF5 file
        with h5py.File(self.h5_file_path, 'w') as f:
            pass 



def calculate_func(cell_id):
    data_graph = KNN_Radius_Graph(radius=radius, dataset=lung_annot_3D_tx_filtered, is_3D=True, cell_ID=cell_id, ref_gene=genes)
    # import pdb; pdb.set_trace()
    distance_matrix = data_graph.get_gene_dis_matrix()
    dis_coo_matrix = binary_to_coo_matrix(distance_matrix)
    pct_nucleus, total_num = data_graph.get_nucleus_info()

    return {cell_id: [dis_coo_matrix, pct_nucleus, total_num]}




def embed_matrix(samples):
    """
    All the results are aligned with the genes that are in the reference.
    """
    all_cellids = samples["Cell_Ids"]
    sample_ids = samples["Sample_Names"]
    refgenes_all = samples["Gene"]
    rankedgenes_all = samples["Ranked_Gene_Names"]
    # genes.index(gene) for gene in ranked_genes

    
    # Prepare default outputs for missing cells
    default_matrix = [[0.0]]  # Default to a 2D list
    default_nucleus = [0.0]      # Default to a list
    default_transcripts = [0]   # Default to a list

    outputs = {
        "Distance_Matrix": [],
        "pct_nucleus": [],
        "Transcript_Number": []
    }

    for i,(cell_id, sample_id) in enumerate(zip(all_cellids,sample_ids)):
        rankedgenes = rankedgenes_all[i]
        refgenes = refgenes_all[i]
        ranked_idx = [refgenes.index(gene) for gene in rankedgenes]



        this_key = sample_id+"_"+cell_id
        if this_key in all_results.keys():
            

            distance_matrix = all_results[this_key][0].toarray()  # Expecting this to be a 2D array

            distance_matrix = distance_matrix[np.ix_(ranked_idx, ranked_idx)]
            # We ensure distance_matrix is a 2D list
            if not isinstance(distance_matrix, list):
                distance_matrix = distance_matrix.tolist()  # Convert to list if necessary
            # import pdb; pdb.set_trace()
            pct_nucleus = all_results[this_key][1]  # Expecting this to be a list
            total_num = all_results[this_key][2]     # Expecting this to be a list
            
            # Handle output structure as lists directly for consistency
            outputs["Distance_Matrix"].append(distance_matrix)  # Append as a 2D list
            outputs["pct_nucleus"].append(pct_nucleus)       # Convert single value to a list
            outputs["Transcript_Number"].append(total_num)     # Assuming this is already a list
        else:
            # import pdb; pdb.set_trace()
            # Append default values for missing cells
            outputs["Distance_Matrix"].append(default_matrix)  # Append the 2D array default value
            outputs["pct_nucleus"].append(default_nucleus)     # Append the default nucleus as a list
            outputs["Transcript_Number"].append(default_transcripts)  # Append the default transcripts

    return outputs




def write_to_dataset(dataset, batch_size = 100):

    dataset = dataset.map(embed_matrix, batched = True, batch_size = batch_size)
    return dataset

# Function to save results to a temporary file
def save_results_to_file(results, filepath):
    # Save the dictionary as a JSON file
    import pdb; pdb.set_trace()
    with open(filepath, 'w') as json_file:
        json.dump(results, json_file, indent=1)  # The indent parameter is optional for pretty-printing

def load_results(filepath):
    with open(filepath, 'r') as json_file:
        results = json.load(json_file)
    return results

def write_to_hdf5(results, h5_file_path):
    with h5py.File(h5_file_path, 'a') as f:
        for cell_id, values in results.items():
            dis_coo_matrix, pct_nucleus, total_num = values[0], values[1], values[2]
            if dis_coo_matrix is not None:
                grp = f.create_group(str(cell_id))
                grp.create_dataset('data', data=dis_coo_matrix.data)
                grp.create_dataset('row', data=dis_coo_matrix.row)
                grp.create_dataset('col', data=dis_coo_matrix.col)
                grp.attrs['shape'] = dis_coo_matrix.shape

                # Store pct_nucleus and total_num as datasets
                grp.create_dataset('pct_nucleus', data=np.array(pct_nucleus))  # Convert to numpy array if needed
                grp.create_dataset('total_num', data=np.array(total_num)) 
                
def load_from_hdf5(h5_file_path):
    results = {}
    with h5py.File(h5_file_path, 'r') as f:
        # Iterate through all groups (cell IDs)
        for cell_id in f.keys():
            grp = f[cell_id]
            
            # Load the coordinate matrix
            data = grp['data'][:]  # Load the 'data' dataset
            row = grp['row'][:]    # Load the 'row' dataset
            col = grp['col'][:]    # Load the 'col' dataset
            
            # Get the shape of the matrix
            shape = grp.attrs['shape']
            
            # Create the COO matrix
            dis_coo = coo_matrix((data, (row, col)), shape=shape)
            
            # Convert COO to Dense format
            # dis_matrix = coo.toarray() 
            
            # Load pct_nucleus and total_num
            pct_nucleus = grp['pct_nucleus'][:]  # Load the 'pct_nucleus' dataset
            total_num = grp['total_num'][:]     # Load the 'total_num' dataset
            
            # Store in results dictionary
            results[cell_id] = [dis_coo, pct_nucleus, total_num]
    
    return results      





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate the gene-gene interaction')
    parser.add_argument('--threshold', type=int, default=100, help='the threshold of transcripts for filtering the cells')
    parser.add_argument('--radius', type=int, default=5, help='the radius to separate compartments')
    parser.add_argument('--pair_threshold', type=int, default=3, help='the pair threshold for the same transcripts and different transcripts')
    parser.add_argument('--transcript_file', type=str, default="/scratch/project_465001027/nicheformer/src/nicheformer/data/raw/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/transcripts.csv", help='the file path of the transcript')
    parser.add_argument('--number_cell', type=int, help='number of cells that are used to calculate, this can be useful for debugging the codes and gene-gene pipeline, default None, which use all the cells')
    parser.add_argument('--dataset_path', type=str, default="TerminatorJ/xenium_25_lung_dataset_update2", help='The name of the dataset in the huggingface repository')
    parser.add_argument('--partition', type=int, default=1, help='The partition of cell_id that are used to run separately')
    parser.add_argument('--chunks', type=int, default=20000, help='The number of chunks for dividing the cell_ids')
    parser.add_argument('--mode', type=str, default="get_result", help='The mode you want to run. default: get_result, alternatively, you can also set dataset_map to embed the results to the huggingface dataset')
    parser.add_argument('--split', type=int, default=1, help='The number of split of the results, ideally, 5 partities can be accessed, otherwise the programe will be killed')
    args = parser.parse_args()
    
    global lung_annot_3D_tx_filtered
    global radius
    global pair_threshold
    global genes
    global all_results
    global cell_id_list

    hf_cache = '/tmp/erda/Spatialformer/'
    prefix = "/home/sxr280/Spatialformer/david_data"
    results_file = f"results_{os.path.basename(args.transcript_file).replace('.csv', f'{args.partition}.h5')}"
    full_h5_file = os.path.join(prefix, results_file)


    all_h5_files = [os.path.join(prefix, file) for file in os.listdir(prefix) if ".h5" in file]
    dataset = load_dataset(args.dataset_path, cache_dir = hf_cache, num_proc = 1)



    boolean_list = np.any([results_file in file for file in os.listdir(prefix)])
    # import pdb; pdb.set_trace()
    if not boolean_list and args.mode == "get_result":
        logging.info(f"Running: {args.transcript_file}")
        processor = GeneInteractionProcessor(args.threshold, args.radius, args.pair_threshold, args.transcript_file, full_h5_file)
        processor.load_and_preprocess_data()
        
        #fetch parameters from the class
        lung_annot_3D_tx_filtered = processor.lung_annot_3D_tx_filtered
        radius = processor.radius
        pair_threshold = processor.pair_threshold
        genes = processor.genes

        cell_ids = list(lung_annot_3D_tx_filtered['cell_id'].unique())
        #calculating how many partitions you need
        # logging.info(f"total partitions you need are: {len(cell_ids)//args.chunks + 1}")
        
        if args.number_cell is not None:
            cell_ids = random.sample(cell_ids, args.number_cell)
        cell_ids = cell_ids[args.chunks * (args.partition - 1): args.chunks * args.partition]
        batch_size = 200
        input_batches = [cell_ids[i:i + batch_size] for i in range(0, len(cell_ids), batch_size)]
        results = {}

        # aa = [calculate_func(cell_ids[0]), calculate_func(cell_ids[1])]
        # run = [results.update(r) for r in aa]
        with Pool(processes=100) as pool:
            for batch in tqdm(input_batches):
                result = list(pool.imap_unordered(calculate_func, batch))
                run = [results.update(r) for r in result]
        # Save results to a temporary file
        # save_results_to_file(results, full_h5_file)
        write_to_hdf5(results, full_h5_file)
    else:
        # import pdb; pdb.set_trace()
        # results = load_results(full_h5_file)
        all_results = {}
        logging.info(f"Getting all the hdf5 files")
        if args.mode == "dataset_map":
            pt = all_h5_files[5 * (args.split - 1): 5 * args.split]
            for file in tqdm(pt):
                
                #map to the remote huggingface repository
                # import pdb; pdb.set_trace()
                results = load_from_hdf5(file)

                print("running:", file)
                print("number of cell:", len(list(results.keys())))
                
                sample_id = file.split("__")[-3]
                new_results = {f"{sample_id}_{cell_id}": value for cell_id, value in results.items()}
                # new_key = [sample_id + "_" + cell_id for cell_id in list(results.keys())]
                print("number of unique cell:", len(np.unique(list(new_results.keys()))))
                all_results.update(new_results)
                # import pdb; pdb.set_trace()

            # pickle.dump(all_results, open("/home/sxr280/Spatialformer/david_data/all_distance_results.pkl","wb"))
            # all_results = pickle.load(open("/home/sxr280/Spatialformer/david_data/all_distance_results.pkl","rb"))

            new_dataset = write_to_dataset(dataset, batch_size = 500)
            #filtering the data
            print("filtering the dataset")
            train_df = new_dataset['train'].to_pandas()
            test_df = new_dataset['test'].to_pandas()
            val_df = new_dataset['validation'].to_pandas()
            train_df = train_df[train_df['Distance_Matrix'].apply(lambda x: len(x[0]) > 2)]
            test_df = test_df[test_df['Distance_Matrix'].apply(lambda x: len(x[0]) > 2)]
            val_df = val_df[val_df['Distance_Matrix'].apply(lambda x: len(x[0]) > 2)]
            print("After filtering")
            # Assuming train_df, test_df, and val_df are your filtered DataFrames
            train_dataset = Dataset.from_pandas(train_df)
            test_dataset = Dataset.from_pandas(test_df)
            val_dataset = Dataset.from_pandas(val_df)

            # Optional: Create a DatasetDict if you want to package these splits
            new_dataset_dict = DatasetDict({
                'train': train_dataset,
                'test': test_dataset,
                'validation': val_dataset
            })
            file_name = args.dataset_path.split("/")[-1]
            # Save the new dataset to disk if needed
            new_dataset_path = os.path.join(hf_cache, file_name + "_" + str(args.split))
            new_dataset_dict.save_to_disk(new_dataset_path)
            
            # new_dataset.push_to_hub(new_dataset_path + str(split_num))
            # new_dataset_path = args.dataset_path[:-1] + str(args.dataset_path[-1])+"_"+str(args.split)
            # import pdb; pdb.set_trace()
            # new_dataset.push_to_hub(new_dataset_path)
            



            # import pdb; pdb.set_trace()
            
            





    # # import pdb; pdb.set_trace()
    # new_dataset_path = args.dataset_path[:-1] + str((int(args.dataset_path[-1])+1))
    # dataset.push_to_hub(new_dataset_path)
    # import pdb; pdb.set_trace()

    # # Handle results after all jobs are done
    # failure_count = sum(1 for result in results if result[1] == None)
    # success_count = len(results) - failure_count
    # logging.info(f"Processing completed. Success: {success_count}, Failure: {failure_count}")
#for dataset_map, please make sure all the cells have been used to calculate
# python find_gene_distance.py --mode dataset_map --split 12

#for THD0008
# python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400_transcripts.csv
# python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400_transcripts.csv
# python find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400_transcripts.csv


#for THD0011
# python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990533_output-XETG00048__0003400__THD0011__20230313__191400_transcripts.csv

#for TILD117MF
# python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990534_output-XETG00048__0003400__TILD117MF__20230313__191400_transcripts.csv
# python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990534_output-XETG00048__0003400__TILD117MF__20230313__191400_transcripts.csv
# python find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990534_output-XETG00048__0003400__TILD117MF__20230313__191400_transcripts.csv



#for TILD117LF
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990535_output-XETG00048__0003400__TILD117LF__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990535_output-XETG00048__0003400__TILD117LF__20230313__191400_transcripts.csv

#for TILD175
# python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990536_output-XETG00048__0003400__TILD175__20230313__191400_transcripts.csv
# python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990536_output-XETG00048__0003400__TILD175__20230313__191400_transcripts.csv



#for VUHD069
# python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990537_output-XETG00048__0003789__VUHD069__20230308__003731_transcripts.csv

#for VUHD095
#python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990538_output-XETG00048__0003789__VUHD095__20230308__003731_transcripts.csv

#for VUHD113
#python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990539_output-XETG00048__0003789__VUHD113__20230308__003731_transcripts.csv


#for VUHD116A
#python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990540_output-XETG00048__0003817__VUHD116A__20230308__003730_transcripts.csv

#for VUHD116B
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990541_output-XETG00048__0003817__VUHD116B__20230308__003731_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990541_output-XETG00048__0003817__VUHD116B__20230308__003731_transcripts.csv

#VUILD102MF
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990542_output-XETG00048__0003817__VUILD102MF__20230308__003730_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990542_output-XETG00048__0003817__VUILD102MF__20230308__003730_transcripts.csv

#VUILD102LF
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990543_output-XETG00048__0003817__VUILD102LF__20230308__003731_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990543_output-XETG00048__0003817__VUILD102LF__20230308__003731_transcripts.csv

#VUILD104MF
#python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990544_output-XETG00048__0003789__VUILD104MF__20230308__003731_transcripts.csv

#VUILD104LF
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990545_output-XETG00048__0003789__VUILD104LF__20230308__003731_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990545_output-XETG00048__0003789__VUILD104LF__20230308__003731_transcripts.csv

#VUILD105MF
#python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990546_output-XETG00048__0003789__VUILD105MF__20230308__003731_transcripts.csv

#VUILD105LF
#python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990547_output-XETG00048__0003789__VUILD105LF__20230308__003731_transcripts.csv


#VUILD106
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 4 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 5 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 6 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv



#VUILD107MF
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990549_output-XETG00048__0003817__VUILD107MF__20230308__003731_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990549_output-XETG00048__0003817__VUILD107MF__20230308__003731_transcripts.csv
#python find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990549_output-XETG00048__0003817__VUILD107MF__20230308__003731_transcripts.csv
#python find_gene_distance.py --partition 4 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990549_output-XETG00048__0003817__VUILD107MF__20230308__003731_transcripts.csv

#VUILD110
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 4 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 5 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 6 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv




#VUILD115
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990551_output-XETG00048__0003392__VUILD115__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990551_output-XETG00048__0003392__VUILD115__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990551_output-XETG00048__0003392__VUILD115__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 4 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990551_output-XETG00048__0003392__VUILD115__20230313__191400_transcripts.csv




#VUILD48MF
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990552_output-XETG00048__0003789__VUILD48MF__20230308__003731_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990552_output-XETG00048__0003789__VUILD48MF__20230308__003731_transcripts.csv

#VUILD48LF
#python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990553_output-XETG00048__0003789__VUILD48LF__20230308__003731_transcripts.csv

#VUILD78MF
#python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990554_output-XETG00048__0003400__VUILD78MF__20230313__191400_transcripts.csv

#VUILD78LF
#python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990555_output-XETG00048__0003400__VUILD78LF__20230313__191400_transcripts.csv

#VUILD91MF
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990556_output-XETG00048__0003400__VUILD91MF__20230313__191400_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990556_output-XETG00048__0003400__VUILD91MF__20230313__191400_transcripts.csv

#VUILD91LF
#python find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990557_output-XETG00048__0003400__VUILD91LF__20230313__191400_transcripts.csv

#VUILD96MF
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990558_output-XETG00048__0003817__VUILD96MF__20230308__003730_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990558_output-XETG00048__0003817__VUILD96MF__20230308__003730_transcripts.csv
#python find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990558_output-XETG00048__0003817__VUILD96MF__20230308__003730_transcripts.csv

#VUILD96LF
#python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990559_output-XETG00048__0003817__VUILD96LF__20230308__003730_transcripts.csv
#python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990559_output-XETG00048__0003817__VUILD96LF__20230308__003730_transcripts.csv
#python find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990559_output-XETG00048__0003817__VUILD96LF__20230308__003730_transcripts.csv
