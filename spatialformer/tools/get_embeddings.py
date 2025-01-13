import torch
import json
import numpy as np
from tqdm import tqdm
import os
import random
from datetime import datetime
import logging
import json
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
# import pdb; pdb.set_trace()
from ..train import manual_train_fm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger("Spatialformer")


#To get the file from the other path as the parent directory

get_file_path = lambda path, filename: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path, filename)

def embed_data(adata,
               tissue, 
               condition,
               method,
               model_ckp_path, 
               batch_size,
               config_path = get_file_path("config", "_config_train_large_pair.json"),
               token_path = get_file_path("tokenizer", "tokenv4.json"),
               mode = "single",
               threshold = 0.8,
               left_cell: Optional[List] = None,
               right_cell: Optional[List] = None,
               num_workers = 0,
               ):
    #fetch the config
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    #loading the model
    logger.info("Loading the SpatialFormer model...")
    model = manual_train_fm(config = config)
    ckp = torch.load(model_ckp_path, map_location=torch.device(device))
    params = ckp["state_dict"]
    model.load_state_dict(params)
    logger.info("Setting the model to the evaluation mode...")
    model.eval()
    logger.info(f"The model is mapped into {device}")
    model.to(device)

    #encoding the data
    logger.info("Encoding the data into the batch...")
    if mode == "single":
        tokenizer = GeneTokenizer(token_path, mode = mode, tissue = tissue, condition = condition)
        dataset = GeneExpressionDataset(adata, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers = num_workers, collate_fn=collate_fn)
        all_embeddings = []
        all_pairs = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):
                
                #getting the last layer
                last_hidden_repr, co_adj_prob = model.get_embeddings(batch, [-1], False, True)#return the probabilities of the gene gene cooccurrence
                if method == "cls":
                    embeddings = last_hidden_repr[0][:,0] #getting the embeddings of cls tokens
                elif method == "gene":
                    #optional, we can use the gene embeddings
                    embeddings = torch.mean(last_hidden_repr[0][:,5:], dim=1) 
                else:
                    raise ValueError(f"Unsupported method: '{method}'. Please use 'cls' or 'gene'.")
                #reveal the gene pair names:
                batch_pairs = reveal_name(GeneTokenizer = tokenizer, co_adj_prob = co_adj_prob, threshold = threshold, batch = batch)
                # import pdb; pdb.set_trace()
                # logger.info(f"{len(all_pairs)} pairs selected")
                all_embeddings.append(embeddings)
                all_pairs += batch_pairs
        # import pdb; pdb.set_trace()
        combined_embeddings = torch.concat(all_embeddings).detach().cpu().numpy()
        adata.obsm["X_SpaF"] = combined_embeddings
        adata.obs["Gene_Pairs"] = all_pairs
        return adata
    elif mode == "pair":
        tokenizer = GeneTokenizer(token_path, mode = mode, tissue = tissue, condition = condition)
        dataset = GeneExpressionPairDataset(adata, left_cell, right_cell, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        all_embeddings = []
        all_prob = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):
                
                #getting the last layer
                last_hidden_repr, probabilities = model.get_embeddings(batch, [-1], True, False)
                cls_embeddings = last_hidden_repr[0][:,0]
                all_embeddings.append(cls_embeddings)
                # probabilities = probabilities.detach().cpu().numpy()
                #calculating the reverse pair results
                rev_batch = rearrange_sentences(batch)
                rev_last_hidden_repr, rev_probabilities = model.get_embeddings(rev_batch, [-1], True, False)
                # import pdb; pdb.set_trace()
                result_tensors = list(map(lambda x, y: process_tensors(x, y), probabilities, rev_probabilities))
                confirm_prob = torch.stack(result_tensors).detach().cpu().numpy()
                all_prob.append(confirm_prob)
            combined_embeddings = torch.concat(all_embeddings).detach().cpu().numpy()
        # import pdb; pdb.set_trace()
        # adata.obsm["X_SpaF"] = combined_embeddings

        return combined_embeddings, all_prob
def process_tensors(tensor1, tensor2):
    # Ensure inputs are tensors
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        raise ValueError("Both inputs must be PyTorch tensors.")

    # Verify that both tensors have exactly two elements
    if tensor1.numel() != 2 or tensor2.numel() != 2:
        raise ValueError("Both tensors must contain exactly two elements.")

    # Check if both tensors have the order element2 > element1
    are_tensor1_aligned = tensor1[1] > tensor1[0]
    are_tensor2_aligned = tensor2[1] > tensor2[0]

    # Process based on the conditions
    if are_tensor1_aligned and are_tensor2_aligned:
        # Both tensors are aligned, so randomly choose one
        chosen_tensor = random.choice([tensor1, tensor2])
    else:
        # Neither tensor is aligned, choose based on the first element
        if tensor1[0] > tensor1[1]:
            chosen_tensor = tensor1
        elif tensor2[0] > tensor2[1]:
            chosen_tensor = tensor2

    return chosen_tensor

def rearrange_sentences(batch):
    # Extract the relevant tensors from the batch dictionary
    batch_size = len(batch['indices'])
    max_length = 500
    indices = batch['indices']
    mask_attentions = batch['attention_mask']  # Corrected to match your input
    token_types = batch['token_type_ids']  # Corrected field names
    new_indices = torch.full((batch_size, max_length), 0, dtype=torch.int)
    new_mask_attention = torch.full((batch_size, max_length), 0, dtype=torch.int)
    new_token_types = torch.full((batch_size, max_length), 0, dtype=torch.int)

    sep_token = 1949  # This is the token ID for the separator

    for i, (indice, mask_attention, token_type) in enumerate(zip(indices, mask_attentions, token_types)):
        # Find the index of the separator token (1949)
#         print("batch_size", batch_size)
        mid_index = (indice == sep_token).nonzero(as_tuple=True)[0][0]
        end_index = (indice == sep_token).nonzero(as_tuple=True)[0][1]
        if mid_index.numel() == 0:  # Check if SEP token is present
            print(f"Warning: SEP token not found in batch index {i}.")
            continue  # Skip to the next batch if no SEP token is found
        
        # Split the indices into two sentences
        cls = indice[:1]  # Include the CLS token
        sentence1 = indice[1:mid_index + 1]  # From index 1 to the SEP token (inclusive)
        sentence2 = indice[mid_index + 1:end_index+1]   # From the token after the SEP to the end
        # Combine to form the new order: sentence2 followed by sentence1
        combined = torch.cat((cls, sentence2, sentence1))
        
        # Pad to the max_length
        new_sequence = torch.cat((combined, torch.zeros(max_length - combined.size(0), dtype=torch.int)))
#         print(new_sequence)
        # Update new_indices
        new_indices[i, :] = new_sequence[:max_length]  # Ensure to only keep the first max_length tokens

        # Update the attention mask
        new_mask_attention[i, :] = mask_attention  

        # Update the token type IDs
        left_token_type = torch.full((len(cls)+len(sentence2),), 1)  # Token type for the second sentence
        right_token_type = torch.full((len(sentence1),), 2)  # Token type for the first sentence
        pad_token_type = torch.full((max_length - combined.size(0),), 0)  # Padding type
        # Combine token types
        new_token_type = torch.cat((left_token_type, right_token_type, pad_token_type))

        # Update new_token_types
        new_token_types[i, :] = new_token_type[:max_length]

    # Put it all back in a dictionary
    new_batch = {
        "indices": new_indices,  # Add batch dimension
        "attention_mask": new_mask_attention.bool(),  # Add batch dimension
        "token_type_ids": new_token_types,  # Add batch dimension
    }
    
    return new_batch



def reveal_name(GeneTokenizer, co_adj_prob, threshold, batch):
    token_to_id = GeneTokenizer.token_to_id
    id_to_token = {v: k for k, v in token_to_id.items()}  # Create a reverse lookup dictionary

    all_pairs = []
    for i, indice in enumerate(batch["indices"]):
        length = len(indice)

        # Get indices for the upper triangle of the adjacency probability matrix
        upper_indices = torch.triu_indices(length, length, offset=1, device = device)  # offset=1 excludes the diagonal

        # Filter based on the threshold
        # import pdb; pdb.set_trace()
        filtered = co_adj_prob[i, upper_indices[0], upper_indices[1]] > threshold
        filtered_indices = upper_indices[0][filtered], upper_indices[1][filtered]
        # import pdb; pdb.set_trace()
        sample_pairs = []
        for pair1, pair2 in zip(*filtered_indices):  # Unpack filtered_indices using * operator
            
            if pair1.item() > 26 and pair2.item() > 26:
                gene_a = id_to_token[indice[pair1.item()].item()]  # Access the gene name from the index
                gene_b = id_to_token[indice[pair2.item()].item()]  # Access the gene name from the index
                sample_pairs.append([gene_a, gene_b])  # Store the pair of gene names
        # import pdb; pdb.set_trace()
        all_pairs.append(sample_pairs)  # Append the list of gene pairs for this batch index

    return all_pairs
        






class GeneTokenizer:
    def __init__(self, token_file, mode = "single", tissue = None, condition = None):
        # Load the token vocabulary
        with open(token_file, 'r') as f:
            self.token_vocab = json.load(f)
        self.token_to_id = {gene: idx for gene, idx in self.token_vocab.items()}
        self.mode = mode
        self.tissue_id = self.token_to_id[tissue]
        self.condition_id = self.token_to_id[condition]

    def single_cell(self, expression_vector):
        # Get indices of ranked genes for the current cell based on expression level
        ranked_genes = np.argsort(expression_vector)[::-1]  # Get indices of ranked genes (high to low)
        # Generate tokens based on ranked genes
        gene_tokens = []
        for gene_index in ranked_genes:
            gene_name = self.genes[gene_index]  # Get the gene name using indices
            if gene_name in self.token_to_id:
                gene_tokens.append(self.token_to_id[gene_name])
        return gene_tokens
    def encode(self, expression_vector1, expression_vector2 = None):
        
        #handling the auxiliary tokens
        cls_token = self.token_to_id["<CLS>"]
        sep_token = self.token_to_id["<SEP>"]
        prefix = [cls_token, self.condition_id, self.tissue_id, 3, 6]
        end = [sep_token]
        if self.mode == "single":
            gene_tokens = self.single_cell(expression_vector1)
            return gene_tokens, prefix, end
        elif self.mode == "pair":
            #for first cell
            gene_token1 = self.single_cell(expression_vector1)
            #for second cell
            gene_token2 = self.single_cell(expression_vector2)
            return (gene_token1, gene_token2), prefix, end

       
        # import pdb; pdb.set_trace()
        

class GeneExpressionDataset(Dataset):
    def __init__(self, adata, tokenizer):
        self.tokenizer = tokenizer
        self.mode = self.tokenizer.mode
        self.expression_data = np.array(adata.X.todense())
        self.genes = adata.var["gene_name"].to_numpy() 

    def __len__(self):
        return self.expression_data.shape[0]  # Number of cells (rows)

    def __getitem__(self, idx):
        expression_vector = self.expression_data[idx]
        self.tokenizer.genes = self.genes  # Assign genes to tokenizer for access
        tokens, prefix, end = self.tokenizer.encode(expression_vector)

        
        return tokens, prefix, end
    

class GeneExpressionPairDataset(Dataset):
    def __init__(self, adata, left_cells, right_cells, tokenizer):
        self.tokenizer = tokenizer
        self.mode = self.tokenizer.mode
        all_cell_names = list(adata.obs.index)
        self.left_indices = [all_cell_names.index(left_cell) for left_cell in left_cells]
        self.right_indices = [all_cell_names.index(right_cell) for right_cell in right_cells]
        # self.right_indices = adata.obs[adata.obs.index.isin(right_cell)].index.tolist()
        try:
            self.left_expression_data = np.array(adata.X[self.left_indices, :].todense())
            self.right_expression_data = np.array(adata.X[self.right_indices, :].todense())
            self.genes = adata.var["gene_name"].to_numpy() 
        except AttributeError:
            self.left_expression_data = np.array(adata.X[self.left_indices, :])
            self.right_expression_data = np.array(adata.X[self.right_indices, :])
            self.genes = adata.var["gene_name"].to_numpy() 

    def __len__(self):
        return self.left_expression_data.shape[0]  # Number of cells (rows)

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        left_expression_vector = self.left_expression_data[idx]
        right_expression_vector = self.right_expression_data[idx]
        self.tokenizer.genes = self.genes  # Assign genes to tokenizer for access
        tokens, prefix, end = self.tokenizer.encode(left_expression_vector, right_expression_vector)

        
        return tokens, prefix, end

def collate_fn(batch):
    def seg_id(token, sep_token):
        before_sep = True
        assigned_tokens = []
        for token in tokens:
            if token == sep_token:  # Check for the separator
                assigned_tokens.append(1)  # Assign 1 to the sep token
                before_sep = False  # Switch flag 
            elif before_sep:
                assigned_tokens.append(1)  # Assign 1 to tokens before the first 2
            else:
                assigned_tokens.append(2 if token != 0 else 0)  # Assign 2 to following tokens or keep pad as 0
        import pdb; pdb.set_trace()
        return assigned_tokens
    
    # Pad the sequences to the max length and create attention masks
    indices = list(map(lambda x: x[0], batch))
    prefix_tokens = list(map(lambda x: x[1], batch))[0]
    end_tokens = list(map(lambda x: x[2], batch))[0]
    end_token = end_tokens[0]
    auxi_lenght = len(prefix_tokens) + len(end_tokens)
    padded_indices = []
    attention_masks = []
    token_type_ids = []
    #single mode
    # import pdb; pdb.set_trace()
    if len(indices[0]) != 2:
        for i, tokens in enumerate(indices):
            tokens_len = len(tokens)
            if tokens_len > (500 - auxi_lenght):
                warn = True
                # import pdb; pdb.set_trace()
                tokens = tokens[:(500 - auxi_lenght)]
                padded_indice = prefix_tokens + tokens + end_tokens
                attention_mask = [1]*len(padded_indice)
                token_type_id = [1]*len(padded_indice)
            else:
                
                pad_size = (500 - auxi_lenght) - tokens_len
                padded_indice = tokens + [0]*pad_size
                attention_mask = [1]*len(tokens) + [0]*pad_size
            padded_indices.append(padded_indice)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
    elif len(indices[0]) == 2: #if pair, there mush be pairs indices returned
        
        for i, (token1,token2) in enumerate(indices):
            token1_len = len(token1)
            token2_len = len(token2)
            fix_length = int((500/2 - auxi_lenght))
            if np.any([token1_len > fix_length, token2_len > fix_length]):
                warn = True
                
                token1 = token1[:fix_length+1]
                padded_indice = prefix_tokens + token1 + end_tokens
                token_type_id = [1]*len(padded_indice)
                # import pdb; pdb.set_trace()
                exclude_cls = prefix_tokens[1:]
                token2 = token2[:fix_length]
                padded_indice += exclude_cls + token2 + end_tokens
                token_type_id += [2]*(len(exclude_cls) + len(token2) + len(end_tokens))#add the second sentence
                # import pdb; pdb.set_trace()
                attention_mask = [1]*len(padded_indice)

            else:
                import pdb; pdb.set_trace()
                # pad_size = (500 - auxi_lenght) - tokens_len

                exclude_cls = prefix_tokens[1:]
                padded_indice = prefix_tokens + token1 + end_tokens + exclude_cls + end_tokens
                pad_size = 500 - padded_indice
                attention_mask = [1]*len(padded_indice) + [0]*pad_size
                padded_indice += [0] * pad_size 
                
            padded_indices.append(padded_indice)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)

    # if warn:
    #     logger.warning("The input is longer than 500 genes, manually set to 500")

    return {"indices": torch.tensor(padded_indices), 
            "attention_mask": torch.tensor(attention_masks).to(torch.bool),
            "token_type_ids": torch.tensor(token_type_ids)}




