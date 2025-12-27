import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from datasets import DatasetDict, load_dataset, concatenate_datasets
import scanpy as sc
from typing import List, Tuple, Dict, Union, Sequence
import logging
import time
import pandas as pd
from collections import namedtuple
from scipy.sparse import csr_matrix
import itertools
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import json
import argparse
from tqdm import tqdm
import psutil
from datasets import load_from_disk
from torch.utils.data import ConcatDataset, Sampler, IterableDataset
import gc
import math
# from .utils import uniform_quantile_global, binning
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]
data_dir = os.path.join(p_path, "david_data")#
tokenizer_dir = os.path.join(p_path, "tokenizer")
hf_cache = os.path.join(p_path, "cache")





logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GeneExpressionDataset:
    def __init__(self, adata, data_path, gene_median_dir, token_dir, erda):
        self.adata = adata
        self.adata_obs = adata.obs
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dataset_dict = None
        self.specie_dict = None
        self.ref_gene = None
        if token_dir is not None:
            with open(token_dir, "r") as file:
                token_indices = json.load(file)
            self.token_indices = token_indices
        else:
            self.token_indices = None

        self.push_to_hub = False
        self.g_g_dict = adata.uns
        self.data_path = data_path
        self.gene_median_dir = gene_median_dir
        self.data_name = data_path.split("/")[-1].split(".h5ad")[0]
        # import pdb; pdb.set_trace()
        if erda:
            self.save_path = os.path.join("/tmp/erda/Spatialformer/downloaded_data/", "processed", self.data_name + "_" + "arrow" + "_" + str(args.partition))

        else:
            parent_dir = os.path.dirname(self.data_path)
            self.save_path = os.path.join(parent_dir, self.data_name + "_" + "arrow" + "_" + str(args.partition))

        
    
    def adata_to_dict(self):
        data_list = []
        for idx in tqdm(range(self.adata.shape[0]), desc="Converting AnnData to list of dicts"):
            exp = np.array(self.adata.X[idx]) if isinstance(self.adata.X[idx], np.matrix) else self.adata.X[idx]
            cell_id = self.adata.obs.index[idx]
            genes = self.adata.var["gene_name"].values
            
            data_list.append({
                "Expression": exp.tolist(),
                "Cell_id": cell_id,
                "Gene": genes.tolist()
            })
        return data_list
    
    def run_in_batch_vectorized(self, samples: Dict[str, List]):
        '''
        Processes a batch of samples for tokenization and data preparation.
        
        The input 'samples' is a dictionary where each value is a list of 
        data points for the batch (e.g., samples["Cell_id"] is a list of Cell_ids).
        '''
        
        batch_size = len(samples["Cell_id"])
        
        # Initialize lists to store results for the entire batch
        batch_cell_ids = []
        batch_ranked_gene_names = []
        batch_full_tokens = []
        batch_row = []
        batch_col = []
        batch_data = []
        batch_shape = []

        # Iterate over each sample in the batch
        for i in range(batch_size):
            cell_id = samples["Cell_id"][i]
            # Assuming Expression is structured as [ [expression_array] ]
            expr = samples["Expression"][i][0]
            genes = np.array(samples["Gene"][i])
            
            # 1. Gene-Gene Matrix (g_g)
            g_g = self.g_g_dict[cell_id].toarray()
            
            # 2. Expression Filtering and Ranking
            # convert the nan to zeros
            nonnan_expr = np.nan_to_num(expr, nan=0)
            # get zero index
            zero_index = np.where(nonnan_expr == 0)[0]
            # filter the zeros and rank in descending way
            sorted_gene_idx = np.argsort(-nonnan_expr)
            # Only keep indices that correspond to non-zero expression values
            sorted_gene_nonzero_idx = sorted_gene_idx[~np.isin(sorted_gene_idx, zero_index)]
            # getting the descending genes (names)
            sorted_genes = genes[sorted_gene_nonzero_idx]
            #make sure the genes are in the token list
            sorted_genes = [g for g in sorted_genes if g in self.token_indices]
            
            # 3. Tokenization
            # get the sorted tokens from gene names
            sorted_tokens = list(map(lambda x: self.token_indices[x], sorted_genes))
            
            # getting other tokens (e.g., "Conditions", "Tissues", etc.)
            add_tokens = list(map(
                lambda x: self.token_indices[self.adata_obs.loc[cell_id, x]], 
                ["Conditions", "Tissues", "Species", "Assay"]
            ))             
            # concatenate all tokens
            full_tokens = add_tokens + sorted_tokens
            
            # 4. Gene-Gene Matrix Selection
            # get selected reference genes indices
            # Ensure 'self.ref_gene' is accessible and has been initialized (e.g., an array of all possible gene names)
            # This part assumes that all genes in 'sorted_genes' are present in 'self.ref_gene'
            selected_index = [np.where(self.ref_gene == g)[0][0] for g in sorted_genes]

            # get the corresponding gene x gene sub-matrix
            gene_gene_matrix = g_g[selected_index, :][:, selected_index]
            del g_g
            gc.collect()
            #convert the gene_gene_matrix to sparse representation
            rows, cols = np.nonzero(gene_gene_matrix)
            data = gene_gene_matrix[rows, cols]
            shape = gene_gene_matrix.shape
            
            
            
            # 5. Collect results for the current sample
            # Note: We collect the raw NumPy arrays/lists/strings here. 
            # The 'datasets' library will handle the final conversion/padding if needed.
            batch_cell_ids.append(cell_id)
            batch_ranked_gene_names.append(sorted_genes)
            batch_full_tokens.append(np.array(full_tokens, dtype=np.int32)) # Use appropriate dtype for tokens
            batch_row.append(rows)
            batch_col.append(cols)
            batch_data.append(data)
            batch_shape.append(shape)
            
        # 6. Return the batch dictionary
        # The output dict must contain lists/arrays of the collected results for all samples
        output = {
            "Cell_Ids": batch_cell_ids,
            # Using ragged list/object dtype for variable length sequences
            "Ranked_Gene_Names": batch_ranked_gene_names, 
            "Full_Tokens": batch_full_tokens,
            "Rows": batch_row,
            "Cols": batch_col,
            "Data": batch_data,
            "Shape": batch_shape
        }
        process = psutil.Process()
        print(f"Memory usage: {process.memory_info().rss / 1024 ** 3:.2f} GB")
        return output
    

    def preprocess_data(self):
        logging.info(f"The data are undergoing preprocessing, it will take a couple minutes")
        # Normalize by gene technique median
        # loading the overall gene median that has been calculated beforehand
        logging.info(f"Genes are normalized by the non-zero median value")
        gene_median_dict = pickle.load(open(self.gene_median_dir, "rb"))
        
        
        # Vectorized normalization
        gene_names = self.adata.var["gene_name"].values
        gene_technique_median = np.array([
        gene_median_dict.get(gene, 1) for gene in gene_names
        ])
        self.adata.X = self.adata.X / gene_technique_median
        # Normalize the cell to have 10,000 counts
        total_counts_per_cell = np.array(np.sum(self.adata.X, axis=1)).flatten()
        target_sum = 1e4
        normalized_expression = self.adata.X.copy()
        for i in range(self.adata.shape[0]):
            cell_sum = total_counts_per_cell[i]
            if cell_sum > 0:
                normalized_expression[i, :] *= target_sum / cell_sum
        self.adata.X = normalized_expression
        del normalized_expression
        gc.collect()
        
        cell_ids = self.adata.obs.index
        #get the ranked gene(non-zero), and gene x gene interacrtion matrix filtered by the gene order
        self.ref_gene = np.array(sorted(self.adata.var["gene_name"].unique()))
        #getting the token indices
        data_list = self.adata_to_dict()
        # Split the data
        logging.info("Create Hugging Face datasets ...")

        dataset = Dataset.from_list(data_list)
        del data_list, self.adata
        #delete the large variable
        gc.collect()
        # Combine into a DatasetDict
        tokenized_datasets = dataset.map(self.run_in_batch_vectorized, batched = True, batch_size = 1) 
        tokenized_datasets.set_format("torch")
        # import pdb; pdb.set_trace()

        if len(cell_ids) < 100:
            pickle.dump(tokenized_datasets, open(self.save_path + ".pkl", "wb"))
        else:
            tokenized_datasets.save_to_disk(self.save_path)
        #split the dataset into train test validation
        if self.push_to_hub:
            tokenized_datasets.push_to_hub(f"{self.data_name}")
        return tokenized_datasets

def get_rank_exp(raw_genes, raw_exps, ranked_genes):
    ranked_exp = []

    # Iterate over each gene list and its corresponding index in ranked_genes
    for i, gene_list in enumerate(ranked_genes):
        exp_values = []
        for gene in gene_list:
            # Attempt to find the index of the gene in the raw_genes
            if gene in raw_genes[i]:
                index = raw_genes[i].index(gene)
                exp_value = raw_exps[i][0][index]
                exp_values.append(exp_value)
            else:
                # Handle the case where the gene is not found
                print(f"Gene {gene} not found in raw_genes[{i}]")
                # Optionally append a placeholder value, e.g., torch.tensor(float('nan'))
        
        ranked_exp.append(torch.tensor(exp_values))
    return ranked_exp

class FilteredSampler(Sampler):
    def __init__(self, dataset):
        self.indices = [idx for idx in range(len(dataset)) if np.array(dataset[idx]['Gene_Gene_Matrix']).sum() != 0]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class filterdataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            # Check if the sum of 'adjmtx' is zero
            if np.array(item['Gene_Gene_Matrix']).sum() != 0:
                yield item  
            else:
                pass

def create_data_loaders(tokenized_datasets, cls_token = 1, sep_token = 1949,batch_size=1, context_length=1500, special_token_num = 4, split_num = 2, directionality = True, n_bins = 51):
    '''
    
    directionality: whether the pair-wise matrix should have the directionality. On the other word, the whether the token that is defined as co-localized
                    can have attention with all the other tokens. If so, this could be a fully attention matrix. If not, this should be a sparse binary matrix.
                    default: True

    '''
    # Create a Data Collator for batching
    class CustomDataCollator(object):
        def __init__(self, context_length, padding_idx=0):
            self.context_length = context_length
            self.padding_idx = padding_idx
            self.cls_token = torch.tensor([cls_token])
            self.sep_token = torch.tensor([sep_token])
            self.special_token_num = special_token_num
            # self.selection = selection

        def __call__(self, batch):
            # Extract sequences and matrices
            # import pdb; pdb.set_trace()
            # if self.selection != None:
            #     batch = [torch.tensor(item['Gene_Gene_Matrix']) for item in batch if item["Full_Tokens"]]
            # torch.cat((token_tensor, value_to_add))
            gg_mtx = [torch.tensor(item['Gene_Gene_Matrix']) for item in batch]
            Full_Tokens = [torch.tensor(item['Full_Tokens']) for item in batch]
            raw_exp = [torch.tensor(item['Expression'][0]) for item in batch]
            annotations = [item['Annotations'] for item in batch]
            # niche_annotation = [item['Niche_Annotations'] for item in batch]
            # Norm_Exp = [torch.tensor(item['Normalized_Exp']) for item in batch]
            raw_genes = [item["Gene"] for item in batch]
            raw_exps = [item["Expression"] for item in batch]
            ranked_genes = [item["Ranked_Gene_Names"] for item in batch]
            # nuc_pct = [item["pct_nucleus"] for item in batch]
            # rank_nuc_pct = [torch.tensor([nuc_pct[i][raw_genes[i].index(gene)] for gene in gene_list]) for i,gene_list in enumerate(ranked_genes)] #getting the nucleus expression percentage level


            # import pdb; pdb.set_trace()
            # ranked_exp = get_rank_exp(raw_genes, raw_exps, ranked_genes)
            ranked_exp = [torch.tensor([raw_exps[i][0][raw_genes[i].index(gene)] for gene in gene_list]) for i,gene_list in enumerate(ranked_genes)] #getting the ranked expression level
            # import pdb; pdb.set_trace()


            # dis_mtx = [torch.tensor(item['Distance_Matrix']) for item in batch]
            
            # import pdb; pdb.set_trace()
            full_tokens = torch.full((len(Full_Tokens), self.context_length), self.padding_idx, dtype=torch.int)
            for i, v in enumerate(Full_Tokens):
                full_tokens[i,:v.size(0)-(4-special_token_num)+len(self.cls_token)+len(self.sep_token)] = torch.cat([self.cls_token, v[4-special_token_num:], self.sep_token]) #add cls and sep to the token
            
            # import pdb; pdb.set_trace()
            full_exp = torch.full((len(Full_Tokens), self.context_length), self.padding_idx, dtype=torch.int)
            for i, v in enumerate(raw_exp):
                full_exp[i,: v.size(0)] = v
            

            # Pad sequences
            attention_masks = (full_tokens != self.padding_idx).bool()
            #token type ids
            token_type_ids = torch.ones_like(full_tokens, dtype=torch.int)
            token_type_ids = attention_masks*token_type_ids

            # Pad 2D matrices
            gg_mtx_p = torch.full((len(gg_mtx), self.context_length, self.context_length), self.padding_idx, dtype=torch.float)
            for i, mat in enumerate(gg_mtx):
                current_size = mat.shape[0]
                gg_mtx_p[i, len(self.cls_token)+self.special_token_num:(len(self.cls_token)+current_size+self.special_token_num), len(self.cls_token)+self.special_token_num:(len(self.cls_token)+self.special_token_num+current_size)] = mat#add head and tail

            if not directionality:
                rows_with_ones = torch.any(gg_mtx_p == 1, dim=2)
                cols_with_ones = torch.any(gg_mtx_p == 1, dim=1)

                # Expand dimensions to match the shape of the original matrix for broadcasting
                rows_with_ones = rows_with_ones.unsqueeze(2)
                cols_with_ones = cols_with_ones.unsqueeze(1)

                # Set the entire row and column to 1 if there is at least one 1
                gg_mtx_p = torch.logical_or(rows_with_ones, cols_with_ones)
                gg_mtx_p = gg_mtx_p.int()

            return {
                'adjmtx': gg_mtx_p,
                'indices': full_tokens,
                'attention_mask': attention_masks,
                "Expression": full_exp,
                "token_type_ids": token_type_ids,
                "Annotations": annotations

            }
    

    data_collator = CustomDataCollator(context_length, padding_idx=0)
    if split_num == 2:
        # Create DataLoaders
        # train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, collate_fn=data_collator, shuffle=True)
        val_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size)
        # test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=batch_size)
        combined_dataset = concatenate_datasets([tokenized_datasets["train"], tokenized_datasets["test"]])
        train_dataloader = DataLoader(combined_dataset, collate_fn=data_collator, batch_size=batch_size)
        return train_dataloader, val_dataloader
    elif split_num == 1:
        
        combined_dataloader = DataLoader(tokenized_datasets, collate_fn=data_collator, batch_size=batch_size, shuffle=True)
        return combined_dataloader
def get_dataset(data_path):
    data_name = data_path.split("/")[-1].split(".h5")[0]
    save_path = os.path.join(data_dir, data_name, "processed", data_name + "_" + "arrow")
    if not os.path.exists(save_path):
        logging.info(f"Reading the h5 data...")
        adata = sc.read_h5ad(data_path)
        
        mydataset = GeneExpressionDataset(adata)
        logging.info(f"{save_path} not exists, preprocessing the anndata to get the dataset...")
        tokenized_datasets = mydataset.preprocess_data()
        
    else:
        logging.info(f"{save_path} already exists, loading the dataset...")
        if os.path.exists(save_path + ".pkl"):
            tokenized_datasets = pickle.load(open(save_path + ".pkl"))
        else:
            tokenized_datasets = load_from_disk(save_path)
            # tokenized_datasets = load_dataset(save_path, cache_dir = hf_cache, num_proc = 1)
            
    return tokenized_datasets



def get_pair_num(adata):
    pair_list = []
    for cell_id in adata.uns.keys():
        pair_num = adata.uns[cell_id].toarray().sum()/2
        pair_list.append(pair_num)
    mean = np.mean(pair_list)
    median = np.median(pair_list)
    return mean, median
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='getting the dataloader for training the model')
    parser.add_argument('--data_path', type=str, default="None", help='The name of the processed anndata dataset with .h5ad suffix')
    parser.add_argument('--gene_median_dir', type=str, default="/scratch/project_465001820/Spatialformer/data/gene_median.pkl", help='The path of the gene technical median')
    parser.add_argument('--token_dir', type=str, default="/scratch/project_465001820/Spatialformer/tokenizer/tokenv5.json", help='The path of the token vocabulary')
    parser.add_argument('--erda', action = 'store_true', help='Whether to store the data into the ERDA system')
    parser.add_argument('--partitions', type = int, default=1, help='The partitions number of the data')
    parser.add_argument('--partition', type = int, default = None, help='The partition of the data')
    parser.add_argument('--chunk', type = int, default = None, help='The chunk size of each partition')
    args = parser.parse_args()

    start_time = time.time()
    data_path = args.data_path
    gene_median_dir = args.gene_median_dir
    token_dir = args.token_dir
    erda = args.erda
    adata = sc.read_h5ad(data_path, backed='r')
    # mean,median = get_pair_num(adata)
    
    if args.chunk is not None:
        if args.partitions == 1:

            partitions = math.ceil(adata.n_obs / args.chunk)
            logging.info(f"Running the partition {args.partition}/{partitions} with chunk size {args.chunk}")
        else:
            logging.info(f"Running the partition {args.partition}/{args.partitions} with chunk size {args.chunk}")
            
        # Start Index (0-indexed): chunk * (partition - 1)
        start_idx = args.chunk * (args.partition - 1)

        # End Index (exclusive): chunk * partition. Ensure it does not exceed the total number of observations.
        end_idx = min(args.chunk * args.partition, adata.n_obs)

        # Ensure the partition argument is valid
        if start_idx >= adata.n_obs:
            logging.warning(f"Partition {args.partition} is out of bounds (max index: {adata.n_obs-1}). Skipping.")
            adata_flt = None # Or raise an error, depending on desired behavior
        else:
            # Slice the AnnData object using integer indices
            # We use .X or .var_names to access the index directly when slicing
            # Note: Slicing by index *numbers* is faster and safer than by .obs.index values
            adata_flt = adata[start_idx:end_idx, :].to_memory() 
            # Extract ONLY needed uns entries (gene-gene matrices for this chunk)
            adata_flt.uns = {
                key: adata.uns[key] 
                for key in adata_flt.obs.index  # ✓ Only chunk's cell IDs
                if key in adata.uns
            }
        #delete the large variable
        del adata
        gc.collect()
        logging.info(f"Selected slice indices: [{start_idx}:{end_idx}] (Size: {end_idx - start_idx} observations)")
    
    
        mydataset = GeneExpressionDataset(adata_flt, data_path, gene_median_dir, token_dir, erda)
    else:
        mydataset = GeneExpressionDataset(adata, data_path, gene_median_dir, token_dir, erda)
    tokenized_datasets = mydataset.preprocess_data()
    
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"The pyarrow data have been generated. Time taken: {duration:.2f} seconds")


#demo
# python h5toloader.py --data_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/processed/relabel_output-XETG00048__0003392__THD0008__20230313__191400.h5ad 

#for the pandataset
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs.h5ad --erda

#Xenium_V1_hLung_cancer_section_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hLung_cancer_section_outs.h5ad --erda

#Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.h5ad --erda

#Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs.h5ad --erda

#Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon_outs.h5ad --erda

#Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon_outs.h5ad --erda

#Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs.h5ad --erda

#Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.h5ad --erda

#Xenium_V1_human_Pancreas_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_human_Pancreas_FFPE_outs.h5ad --erda

#Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs.h5ad --erda

#Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs.h5ad --erda

#Xenium_V1_hLiver_nondiseased_section_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hLiver_nondiseased_section_FFPE_outs.h5ad --erda

#Xenium_V1_hLiver_cancer_section_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hLiver_cancer_section_FFPE_outs.h5ad --erda

#Xenium_V1_hHeart_nondiseased_section_FFPE_outs
#python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hHeart_nondiseased_section_FFPE_outs.h5ad --erda

#Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.h5ad --erda

#Xenium_V1_hSkin_Melanoma_Base_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hSkin_Melanoma_Base_FFPE_outs.h5ad --erda

#Xenium_V1_hColon_Non_diseased_Base_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hColon_Non_diseased_Base_FFPE_outs.h5ad --erda

#Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs.h5ad --erda

#Xenium_V1_hColon_Cancer_Base_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hColon_Cancer_Base_FFPE_outs.h5ad --erda

#Xenium_V1_hLung_cancer_section_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hLung_cancer_section_outs.h5ad --erda

#Xenium_V1_hLymphNode_nondiseased_section_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hLymphNode_nondiseased_section_outs.h5ad --erda

#Xenium_V1_humanLung_Cancer_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_humanLung_Cancer_FFPE_outs.h5ad --erda

#Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs.h5ad --erda

#Xenium_V1_Human_Lung_Cancer_Addon_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_Human_Lung_Cancer_Addon_FFPE_outs.h5ad --erda

#Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs.h5ad --erda

#Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_outs.h5ad --erda

#Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs.h5ad --erda

# Xenium_V1_FFPE_Human_Breast_IDC_Big_1_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_FFPE_Human_Breast_IDC_Big_1_outs.h5ad --erda

#Xenium_V1_FFPE_Human_Breast_IDC_Big_2_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_FFPE_Human_Breast_IDC_Big_2_outs.h5ad --erda

#Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs.h5ad --erda

#Xenium_V1_FFPE_Human_Breast_ILC_With_Addon_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_FFPE_Human_Breast_ILC_With_Addon_outs.h5ad --erda

#Xenium_V1_FFPE_Human_Breast_IDC_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_FFPE_Human_Breast_IDC_outs.h5ad --erda

#Xenium_V1_FFPE_Human_Breast_ILC_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_FFPE_Human_Breast_ILC_outs.h5ad --erda

#Xenium_V1_hPancreas_nondiseased_section_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hPancreas_nondiseased_section_outs.h5ad --erda

#Xenium_V1_hKidney_nondiseased_section_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hKidney_nondiseased_section_outs.h5ad --erda

#Xenium_V1_hKidney_cancer_section_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hKidney_cancer_section_outs.h5ad --erda

#Xenium_V1_hColon_Cancer_Add_on_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hColon_Cancer_Add_on_FFPE_outs.h5ad --erda

#Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs.h5ad --erda

#Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs
# python h5toloader.py --data_path /tmp/erda/Spatialformer/downloaded_data/processed/Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs.h5ad --erda
