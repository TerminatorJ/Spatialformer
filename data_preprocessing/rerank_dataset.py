from datasets import DatasetDict, load_dataset, concatenate_datasets, load_from_disk
import pickle
from typing import List
import json
import psutil
import numpy as np
from typing import Dict




def rerank(ori_exp: List, gene_names: List[str]):
    """
    Rerank the gene expression according to the gene median expression for Xenium technique
    Args:
    ori_exp: List. the original gene expression values after normalization.
    gene_names: List[str].list of gene names corresponding to the expression values, which is not normalized.
    
    Return:
    the reranked gene tokens
    """
    gene_median_dict = pickle.load(open(gene_median_dir, "rb"))
    gene_technique_median = np.array([
        gene_median_dict.get(gene, 1) for gene in gene_names
        ])
    
    assert len(ori_exp) == len(gene_technique_median), "The gene names should have the same length as expression velues"
    norm_exp = np.array(ori_exp) / gene_technique_median
    # 1. Expression Filtering and Ranking
    # convert the nan to zeros
    nonnan_expr = np.nan_to_num(norm_exp, nan=0)
    # get zero index
    zero_index = np.where(nonnan_expr == 0)[0]
    # filter the zeros and rank in descending way
    sorted_gene_idx = np.argsort(-nonnan_expr)
    # Only keep indices that correspond to non-zero expression values
    sorted_gene_nonzero_idx = sorted_gene_idx[~np.isin(sorted_gene_idx, zero_index)]
    # getting the descending genes (names)
    sorted_genes = np.array(gene_names)[sorted_gene_nonzero_idx]
    #make sure the genes are in the token list
    sorted_genes = [g for g in sorted_genes if g in token_indices]
    #get the sorted gene expression
    sorted_exp = np.array(norm_exp)[sorted_gene_nonzero_idx]
    
    # 3. Tokenization
    # get the sorted tokens from gene names
    sorted_tokens = list(map(lambda x: token_indices[x], sorted_genes))
    
    return sorted_genes, sorted_tokens, sorted_exp
    


#rerank the gene according to the new expression median values
def run_in_batch_vectorized(samples: Dict[str, List]):
        '''
        Processes a batch of samples for tokenization and data preparation.
        
        The input 'samples' is a dictionary where each value is a list of 
        data points for the batch (e.g., samples["Cell_id"] is a list of Cell_ids).
        '''
        
        batch_size = len(samples["Cell_id"])
        batch_cell_ids = []
        batch_ranked_gene_names = []
        batch_ranked_gene_exp = []
        batch_gene_names = []
        batch_full_tokens = []
        batch_row = []
        batch_col = []
        batch_data = []
        batch_shape = []
        batch_expression = []
        batch_split = []
        batch_species = []
        batch_assay = []
        
        
        
        for i in range(batch_size):
            cell_id = samples["Cell_id"][i]
            split = samples["Split"][i]
            ori_exp = samples["Expression"][i][0]
            gene_names = samples["Gene"][i]
            condition = samples["Conditions"][i]
            tissue = samples["Tissues"][i]  
            if tissue == "Bone Marrow":
                tissue = "BoneMarrow"
            species = "Human"
            assay = "Xenium"
            g_g = np.array(samples["Gene_Gene_Matrix"][i])
            ranked_gene_names = samples["Ranked_Gene_Names"][i]
            #Getting the sorted gene tokens and gene names
            sorted_genes, sorted_tokens, sorted_exp = rerank(ori_exp = ori_exp, gene_names = gene_names)
            # getting other tokens (e.g., "Conditions", "Tissues", etc.)
            add_tokens = list(map(
                lambda x: token_indices[x], 
                [condition, tissue, species, assay]
            ))             
            # concatenate all tokens
            full_tokens = add_tokens + sorted_tokens
        
            
            # 4. Gene-Gene Matrix Selection
            # get the reranked gene-gene matrix according to the sorted genes
                       
            selected_index = [ranked_gene_names.index(gene) for gene in sorted_genes]
            # get the corresponding gene x gene sub-matrix
            gene_gene_matrix = g_g[selected_index, :][:, selected_index]
            #convert the gene_gene_matrix to sparse representation
            rows, cols = np.nonzero(gene_gene_matrix)
            data = gene_gene_matrix[rows, cols]
            shape = gene_gene_matrix.shape
            #collect all the data in batch
            batch_expression.append(ori_exp)
            batch_split.append(split)
            batch_cell_ids.append(cell_id)
            batch_gene_names.append(gene_names)
            batch_ranked_gene_names.append(sorted_genes)
            batch_full_tokens.append(full_tokens)
            batch_row.append(rows)
            batch_col.append(cols)
            batch_data.append(data)
            batch_shape.append(shape)
            batch_ranked_gene_exp.append(sorted_exp)
            batch_species.append(species)
            batch_assay.append(assay)
            
            
        # 5. Return the batch dictionary
        # The output dict must contain lists/arrays of the collected results for all samples
        output = {
            "Expression": batch_expression,
            "Split": batch_split,
            "Cell_Ids": batch_cell_ids,
            "Gene": batch_gene_names,
            "Ranked_Gene_Names": batch_ranked_gene_names, 
            "Full_Tokens": batch_full_tokens,
            "Rows": batch_row,
            "Cols": batch_col,
            "Data": batch_data,
            "Shape": batch_shape,
            "Annotations": [str(ann) if ann is not None else "" for ann in samples["Annotations"]],
            "Sample_Names": samples["Sample_Names"],
            "compound_key": [key if key is not None else "" for key in samples["compound_key"]],
            "Niche_Annotation": [na if na is not None else "" for na in samples["Niche_Annotations"]],
            "centroid_x": samples["centroid_x"],
            "centroid_y": samples["centroid_y"],
            "Tissues": samples["Tissues"],
            "Species": batch_species,
            "Assay": batch_assay,
            "Conditions": samples["Conditions"]
            
        }
        return output
    
    
    
gene_median_dir = "/scratch/project_465001820/Spatialformer/data/gene_median.pkl"
with open("/scratch/project_465001820/Spatialformer/tokenizer/tokenv5.json", "r") as f: 
    token_indices = json.load(f)
    
    
combined_dataset = load_dataset("TerminatorJ/xenium_pandavid_dataset5",cache_dir = "/scratch/project_465001820/Spatialformer/cache", num_proc=64)
new_combined_dataset = combined_dataset.map(run_in_batch_vectorized, 
                                            batched=True, 
                                            batch_size=1000, 
                                            num_proc=128,
                                            remove_columns=[combined_dataset["train"].column_names])
new_combined_dataset.save_to_disk("/scratch/project_465001820/Spatialformer/cache/xenium_pandavid_dataset6")