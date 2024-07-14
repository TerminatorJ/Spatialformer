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
from datasets import load_from_disk
from torch.utils.data import ConcatDataset
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]
data_dir = os.path.join(p_path, "david_data")#
tokenizer_dir = os.path.join(p_path, "tokenizer")
hf_cache = os.path.join(p_path, "cache")





logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GeneExpressionDataset:
    def __init__(self, adata, data_path):
        self.adata = adata
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dataset_dict = None
        self.specie_dict = None
        self.ref_gene = None
        self.token_indices = None
        self.push_to_hub = True
        self.g_g_dict = adata.uns
        self.data_path = data_path
        self.data_name = data_path.split("/")[-1].split(".h5")[0]
        # import pdb; pdb.set_trace()
        self.save_path = os.path.join(data_dir, self.data_name, "processed", self.data_name + "_" + "arrow")

        
    def define_all_tokens(self):

        #define the data dictionary
        Dictionary = namedtuple('Dictionary', ['symbols'])
        special_tokens = ["<pad>", "<mask>", '<CLS>']
        other_tokens = ['Human', 'Mouse', "Merfish", "Xenium", "Lung", "Healthy", "Disease"]
        gene_tokens = list(self.ref_gene)
        # import pdb; pdb.set_trace()
        vocab = Dictionary(
            symbols = special_tokens + other_tokens + gene_tokens,
        )
        self.token_indices = {token: idx for idx, token in enumerate(vocab.symbols)}
        
        with open(os.path.join(tokenizer_dir, "token.json"), 'w') as json_file:
            json.dump(self.token_indices, json_file, indent=4)
    
    def adata_to_dict(self):
        data_list = []
        for idx in range(self.adata.shape[0]):
            exp = np.array(self.adata.X[idx]) if isinstance(self.adata.X[idx], np.matrix) else self.adata.X[idx]
            split = self.adata.obs["Split"][idx]
            cell_id = self.adata.obs.index[idx]
            genes = self.adata.var["gene_name"].values
            
            data_list.append({
                "Expression": exp.tolist(),
                "Split": split,
                "Cell_id": cell_id,
                "Gene": genes.tolist()
            })
        return data_list
    
    def run_in_batch(self, sample : DatasetDict) -> Dict:
        '''
        The input is a batch of samples, which allows for faster preprocessing and storing the data.
        This function is used to do the tokenization by predefined token dict along with the special tokens and the gene tokens.
        '''
        cell_id = sample["Cell_id"][0]
        expr = sample["Expression"][0][0]
        split = sample["Split"][0]
        genes = np.array(sample["Gene"][0])
        # import pdb; pdb.set_trace()
        g_g = self.g_g_dict[cell_id].toarray()
        
        #convert the nan to zeros
        nonnan_expr = np.nan_to_num(expr, nan=0)
        #get zero index
        zero_index = np.where(nonnan_expr == 0)[0]
        #filter the zeros and rank in descending way
        sorted_gene_idx = np.argsort(-nonnan_expr)
        sorted_gene_nonzero_idx = sorted_gene_idx[~np.isin(sorted_gene_idx,zero_index)]
        #getting the descending genes
        sorted_genes = genes[sorted_gene_nonzero_idx]
        #get the sorted tokens
        sorted_tokens = list(map(lambda x: self.token_indices[x], sorted_genes))
        #getting other tokens
        #replace the name to token
        add_tokens = list(map(lambda x: self.token_indices[self.adata.obs.loc[cell_id, x]], ["Conditions", "Tissues", "Species", "Assay"]))
        # import pdb; pdb.set_trace()                
        #concate all tokens
        full_tokens = add_tokens + sorted_tokens
        #get selected reference genes
        selected_index = [np.where(self.ref_gene == g)[0][0] for g in sorted_genes]
        # import pdb; pdb.set_trace()
        #get the corresponding gene x gene
        # import pdb; pdb.set_trace()
        gene_gene_matrix = g_g[selected_index, :][:, selected_index]
        
        #preparing for storing
        sorted_genes = np.expand_dims(sorted_genes, axis=0)
        full_tokens = np.expand_dims(np.array(full_tokens), axis=0)
        gene_gene_matrix = np.expand_dims(gene_gene_matrix, axis = 0)
        cell_id = np.expand_dims(np.array(cell_id), axis = 0)
        
        output = {"Cell_Ids" : cell_id, "Ranked_Gene_Names" : sorted_genes, "Full_Tokens" : full_tokens, "Gene_Gene_Matrix": gene_gene_matrix}#array with variant lengths
        return output
    
        
        
    
    def preprocess_data(self):
        logging.info(f"The data are undergoing preprocessing, it will take a couple minutes")
        

        # Normalize the cell to have 10,000 counts
        total_counts_per_cell = np.array(np.sum(self.adata.X, axis=1)).flatten()
        target_sum = 1e4
        normalized_expression = self.adata.X.copy()
        for i in range(self.adata.shape[0]):
            cell_sum = total_counts_per_cell[i]
            if cell_sum > 0:
                normalized_expression[i, :] *= target_sum / cell_sum
        # import pdb; pdb.set_trace()
        self.adata.X = normalized_expression
        # Normalize by gene technique median
        gene_technique_mean = [self.adata.X[:, i][self.adata.X[:, i].nonzero()[0]].mean() for i in range(self.adata.shape[1])]
        self.adata.X = self.adata.X / np.array(gene_technique_mean)

        cell_ids = self.adata.obs.index
        #get the ranked gene(non-zero), and gene x gene interacrtion matrix filtered by the gene order
        self.ref_gene = np.array(sorted(adata.var["gene_name"].unique()))
        #getting the token indices
        # import pdb; pdb.set_trace()
        self.define_all_tokens()
        data_list = self.adata_to_dict()
        # Split the data
        train_data = [d for d in data_list if d["Split"] == "train"]
        test_data = [d for d in data_list if d["Split"] == "test"]
        validation_data = [d for d in data_list if d["Split"] == "validation"]
        # Create Hugging Face datasets
        # import pdb; pdb.set_trace()
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        validation_dataset = Dataset.from_list(validation_data)
        # Combine into a DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
            "validation": validation_dataset
        })
        tokenized_datasets = dataset_dict.map(self.run_in_batch, batched = True, batch_size = 1)
        tokenized_datasets.set_format("torch")
        
        # import pdb; pdb.set_trace()
        if len(cell_ids) < 100:
            pickle.dump(tokenized_datasets, open(self.save_path + ".pkl", "wb"))
        else:
            tokenized_datasets.save_to_disk(self.save_path)
        #split the dataset into train test validation
        # import pdb; pdb.set_trace()
        if self.push_to_hub:
            tokenized_datasets.push_to_hub(f"{self.data_name}")
        return tokenized_datasets

        

def create_data_loaders(tokenized_datasets, batch_size=1, context_length=1500):
    # Create a Data Collator for batching
    class CustomDataCollator(object):
        def __init__(self, context_length, padding_idx=0):
            self.context_length = context_length
            self.padding_idx = padding_idx

        def __call__(self, batch):
            # Extract sequences and matrices
            # import pdb; pdb.set_trace()
            gg_mtx = [torch.tensor(item['Gene_Gene_Matrix']) for item in batch]
            Full_Tokens = [torch.tensor(item['Full_Tokens']) for item in batch]
            
            # import pdb; pdb.set_trace()
            full_tokens = torch.full((len(Full_Tokens), self.context_length), self.padding_idx, dtype=torch.int)
            for i, v in enumerate(Full_Tokens):
                full_tokens[i,:v.size(0)] = v
            # import pdb; pdb.set_trace()
            # Pad sequences
            attention_masks = (full_tokens != self.padding_idx).bool()
            # import pdb; pdb.set_trace()

            # Pad 2D matrices
            gg_mtx_p = torch.full((len(gg_mtx), self.context_length, self.context_length), self.padding_idx, dtype=torch.float)
            for i, mat in enumerate(gg_mtx):
                current_size = mat.shape[0]
                gg_mtx_p[i, :current_size, :current_size] = mat
            # import pdb; pdb.set_trace()
            return {
                'adjmtx': gg_mtx_p,
                'indices': full_tokens,
                'attention_mask': attention_masks
            }

    data_collator = CustomDataCollator(context_length, padding_idx=0)
    # Create DataLoaders
    # train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, collate_fn=data_collator, shuffle=True)
    val_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size)
    # test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=batch_size)
    combined_dataset = concatenate_datasets([tokenized_datasets["train"], tokenized_datasets["test"]])
    train_dataloader = DataLoader(combined_dataset, collate_fn=data_collator, batch_size=batch_size)
    return train_dataloader, val_dataloader

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
    parser.add_argument('--data_path', type=str, default="None", help='The name of the processed h5 dataset')
    # parser.add_argument('--data_name', type=str, default="None", help='The name of the raw dataset')
    args = parser.parse_args()

    start_time = time.time()
    # adata = sc.read_h5ad("/scratch/project_465001027/spatialformer/data/processed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs_toy.h5ad")
    # import pdb; pdb.set_trace()
    #save to a tiny example h5 file to test the pipeline
    # test_adata = adata[:10,:10]
    # test_adata.uns =  {cell_id:test_adata.uns[cell_id] for cell_id in test_adata.obs.index}
    #save the toy example
    # test_adata.write("/scratch/project_465001027/spatialformer/data/processed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs_toy.h5ad")
    
    data_path = args.data_path
    adata = sc.read_h5ad(data_path)
    mean,median = get_pair_num(adata)
    # import pdb; pdb.set_trace()
    
    logging.info(f"The mean number of the gene pairs is:{mean}; the median is: {median}")
    
    
    mydataset = GeneExpressionDataset(adata, data_path)
    tokenized_datasets = mydataset.preprocess_data()
    
    # data_path = "/scratch/project_465001027/spatialformer/data/processed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.h5ad"
    # tokenized_datasets = get_dataset(data_path)
    
    # train_dataloader, val_dataloader, test_dataloader = create_data_loaders(tokenized_datasets, batch_size=1)
        
    # for batch in train_dataloader:
    #     matrix = batch["Gene_Gene_Matrix"]
    # import pdb; pdb.set_trace()
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"The dataloader has been generated. Time taken: {duration:.2f} seconds")


#demo
# python h5toloader.py --data_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/processed/relabel_output-XETG00048__0003392__THD0008__20230313__191400.h5ad 