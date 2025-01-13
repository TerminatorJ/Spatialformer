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
from torch.utils.data import ConcatDataset, Sampler, IterableDataset
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
        self.gene_median = gene_median_dir
        self.data_name = data_path.split("/")[-1].split(".h5")[0]
        # import pdb; pdb.set_trace()
        if erda:
            self.save_path = os.path.join("/tmp/erda/Spatialformer/downloaded_data/", "processed", self.data_name + "_" + "arrow")

        else:
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
        # import pdb; pdb.set_trace()
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
        
        # Normalize by gene technique median
        # loading the overall gene median that has been calculated beforehand
        # gene_technique_mean = [self.adata.X[:, i][self.adata.X[:, i].nonzero()[0]].median() for i in range(self.adata.shape[1])]
        gene_median_dict = pickle.load(open("/home/sxr280/Spatialformer/data/Xenium_median_gene_final_exp.pkl", "rb"))
        # import pdb; pdb.set_trace()
        gene_technique_mean = [gene_median_dict[gene] for gene in self.adata.var["gene_name"].values]
        # import pdb; pdb.set_trace()
        self.adata.X = self.adata.X / np.array(gene_technique_mean)
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
        

        cell_ids = self.adata.obs.index
        #get the ranked gene(non-zero), and gene x gene interacrtion matrix filtered by the gene order
        self.ref_gene = np.array(sorted(adata.var["gene_name"].unique()))
        #getting the token indices
        # import pdb; pdb.set_trace()
        if self.token_indices is None:
            self.define_all_tokens()
        data_list = self.adata_to_dict()
        # Split the data
        train_data = [d for d in data_list if d["Split"] == "train"]
        test_data = [d for d in data_list if d["Split"] == "test"]
        validation_data = [d for d in data_list if d["Split"] == "validation"]
        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        tokenized_datasets = dataset_dict.map(self.run_in_batch, batched = True, batch_size = 1)
        tokenized_datasets.set_format("torch")
        # import pdb; pdb.set_trace()
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
            


            # norm_exp = torch.full((len(ranked_exp), self.context_length), self.padding_idx, dtype=torch.float)
            # try:
            #     for i, e in enumerate(ranked_exp):
            #         # import pdb; pdb.set_trace()
            #         e = binning(
            #             row=e,
            #             n_bins=n_bins,
            #         )

            #         norm_exp[i,self.special_token_num:self.special_token_num+e.size(0)] = e
            # except:
            #     import pdb; pdb.set_trace()
            #     pass

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
    parser.add_argument('--data_path', type=str, default="None", help='The name of the processed h5 dataset')
    parser.add_argument('--gene_median_dir', type=str, default="/home/sxr280/Spatialformer/data/Xenium_median_gene_final_exp.pkl", help='The path of the gene technical median')
    parser.add_argument('--token_dir', type=str, default="/home/sxr280/Spatialformer/tokenizer/tokenv2.json", help='The path of the token vocabulary')
    parser.add_argument('--erda', action = 'store_true', help='Whether to store the data into the ERDA system')

    # parser.add_argument('--data_name', type=str, default="None", help='The name of the raw dataset')
    args = parser.parse_args()

    start_time = time.time()
    data_path = args.data_path
    gene_median_dir = args.gene_median_dir
    token_dir = args.token_dir
    erda = args.erda
    adata = sc.read_h5ad(data_path)
    mean,median = get_pair_num(adata)
    # import pdb; pdb.set_trace()
    
    logging.info(f"The mean number of the gene pairs is:{mean}; the median is: {median}")
    
    
    mydataset = GeneExpressionDataset(adata, data_path, gene_median_dir, token_dir, erda)
    tokenized_datasets = mydataset.preprocess_data()
    
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"The dataloader has been generated. Time taken: {duration:.2f} seconds")


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
