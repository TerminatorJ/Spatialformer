#This dataloader support the pairwise cell input, which means the data can be used to train the model with cell level spatial information
#author: Jun
#31/10/2024

from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import numpy as np
import h5py
import os
import sys
from utils.utils import *
import pickle
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataCollator(object):
    def __init__(self, directionality, context_length = 1000, padding_idx=0, special_token_num = 4, n_bins = 51, sep_token = 1949, cls_token = 1, cell_id = False):
        self.context_length = context_length
        self.padding_idx = padding_idx
        self.directionality = directionality
        self.special_token_num = special_token_num
        self.n_bins = n_bins
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pair1_length = None
        self.last_mtx_length = None
        self.batch = None
        self.full_tokens = None
        self.full_exp = None
        self.norm_exp = None
        self.gg_mtx_p = None
        self.cell_id = cell_id
    def __call__(self, batch):
        '''
        batch is the index and label, we need to get the rows of dataset
        '''
        #Getting the rows from the datasets
        # import pdb; pdb.set_trace()
        # filtered_batch = [item for item in batch if len(item[0]) < 500]
        pair_labels =  torch.tensor([data["Labels"] for data in batch])
        if self.cell_id:
            left_cell_ids =  [data["left_Cell_Ids"] for data in batch]
            right_cell_ids =  [data["right_Cell_Ids"] for data in batch]
        #define which data you want to extract from the dataset
        
        self.full_tokens = torch.full((len(batch), self.context_length), self.padding_idx, dtype=torch.int)#, device = device)
        self.token_type_ids =  torch.full((len(batch), self.context_length), self.padding_idx, dtype=torch.int)#, device = device)
        self.full_exp = torch.ones((len(batch), self.context_length), dtype=torch.int)#, device = device)
        self.norm_exp = torch.ones((len(batch), self.context_length),dtype=torch.float)#, device = device)
        self.gg_mtx = torch.full((len(batch), self.context_length, self.context_length), self.padding_idx, dtype=torch.float)#, device = device)
         
        #fill the data 

        self.one_site(batch)

       
        

        # Filtering: Keep samples with at least 5 non-padding elements
        valid_mask = (self.full_tokens != self.padding_idx)  # Create a mask for non-padding elements
        non_padding_counts = valid_mask.sum(dim=1)  # Count non-padding elements for each sample
        # Filter based on the count of non-padding elements
        non_zero_indices = non_padding_counts >= 5  # Keep only samples with at least 5 non-padding elements
        # Now filter gg_mtx to exclude 2D matrices that are all zeros
        gg_mtx_nonzero_mask = (self.gg_mtx.sum(dim=(1, 2)) != 0)  # Check if sum across the last two dimensions is not zero
        non_zero_indices = non_zero_indices & gg_mtx_nonzero_mask  # Combine both conditions



        # Check if there are valid samples
        if not non_zero_indices.any():
            print("Skipping batch: no valid samples with at least 5 non-padding elements.")
            return None 

        # Apply the filter to all relevant tensors
        self.full_tokens = self.full_tokens[non_zero_indices]
        self.gg_mtx = self.gg_mtx[non_zero_indices]
        self.token_type_ids = self.token_type_ids[non_zero_indices]
        self.full_exp = self.full_exp[non_zero_indices]
        self.norm_exp = self.norm_exp[non_zero_indices]
        pair_labels = pair_labels[non_zero_indices]


        # Pad sequences after filling the data
        attention_masks = (self.full_tokens != self.padding_idx).bool()
        # print(attention_masks.sum())
        # pair_labels =  torch.tensor(self.pair_labels)
        # import pdb; pdb.set_trace()
        if self.cell_id:
            return {
                'adjmtx': self.gg_mtx,
                'indices': self.full_tokens,
                'attention_mask': attention_masks,
                'normalized_exp': self.norm_exp,
                "Expression": self.full_exp,
                "pair_label": pair_labels,
                "token_type_ids": self.token_type_ids,
                "left_cell_ids": left_cell_ids,
                "right_cell_ids": right_cell_ids
            }
        else:
            return {
                'adjmtx': self.gg_mtx,
                'indices': self.full_tokens,
                'attention_mask': attention_masks,
                'normalized_exp': self.norm_exp,
                "Expression": self.full_exp,
                "pair_label": pair_labels,
                "token_type_ids": self.token_type_ids,
            }
    def filldata(self, sample_index, full_tokens, gg_mtx, raw_exp, side):

        # full_tokens = torch.tensor(batch["Full_Tokens"][0])
        # gg_mtx = torch.tensor(batch["Gene_Gene_Matrix"][0])
        # raw_exp = torch.tensor(batch["Expression"][0][0])

        #make sure all the data shorter than the context_length
        if len(full_tokens) > self.context_length:
            full_tokens = full_tokens[:self.context_length]
        if gg_mtx.shape[0] > self.context_length:
            gg_mtx = gg_mtx[:self.context_length, :self.context_length]
        if len(raw_exp) > self.context_length:
            raw_exp = raw_exp[:self.context_length]
        # import pdb; pdb.set_trace()
        self.full_exp[sample_index,1: raw_exp.size(0)+1] = raw_exp #1 for cls token
        # import pdb; pdb.set_trace()
        cls_site = 1
        sep_site = 1
        current_size = gg_mtx.shape[0]
        if side == 0: #for the left pair
            # import pdb; pdb.set_trace()
            prefix_length = cls_site + self.special_token_num
            self.pair1_length = cls_site + full_tokens.size(0)-(4-self.special_token_num)
            #adding the cls token first
            self.full_tokens[sample_index,0] = self.cls_token
            #adding the tokens 
            self.full_tokens[sample_index,cls_site:self.pair1_length] = full_tokens[4-self.special_token_num:] #add the special token for the left and right sequence
            #adding the sep in the middle of pair
            self.full_tokens[sample_index, self.pair1_length] = self.sep_token
            #adding the gene pair matrix
            self.last_mtx_length = gg_mtx.shape[0]
            self.gg_mtx[sample_index, prefix_length:(prefix_length+current_size), prefix_length:(prefix_length+current_size)] = gg_mtx

            self.token_type_ids[sample_index, :self.pair1_length+cls_site] = 1
        elif side == 1: #for the right pair
            # import pdb; pdb.set_trace()
            prefix_length = cls_site + self.special_token_num
            
            #adding the right cell
            start = (self.pair1_length + sep_site)
            seq_length = full_tokens.size(0)-(4-self.special_token_num)
            self.full_tokens[sample_index, start:(start + seq_length)] = full_tokens[4-self.special_token_num:]
            #add the sep to the right end
            self.full_tokens[sample_index, (start + seq_length): (start + seq_length + sep_site) ] = self.sep_token
            #add the gene pairs to the right cell
            self.gg_mtx[sample_index, (prefix_length + self.last_mtx_length + cls_site) : (prefix_length + self.last_mtx_length + cls_site + current_size), (prefix_length + self.last_mtx_length + cls_site) : (prefix_length + self.last_mtx_length + cls_site + current_size)] = gg_mtx

            self.token_type_ids[sample_index, start:(start + seq_length + sep_site)] = 2
        # import pdb; pdb.set_trace()

    

    def rebuild_adj(self, data, row, col, shape):
        # import pdb; pdb.set_trace()
        sparse_matrix = coo_matrix((data, (row, col)), shape=shape)
        adj = sparse_matrix.toarray()
        return adj


    def one_site(self, dataset_batch):

        for i, sample in enumerate(dataset_batch):
            # import pdb; pdb.set_trace()
            # pair = sample["Labels"]
            # self.pair_labels.append(pair)
            #for left
            full_tokens = torch.tensor(sample["left_Full_Tokens"])
            row = sample["left_row"]
            col = sample["left_col"]
            data = sample["left_data"]
            shape = sample["left_shape"]
            gg_mtx = self.rebuild_adj(data, row, col, shape)
            gg_mtx = torch.tensor(gg_mtx)
            raw_exp = torch.tensor(sample["left_Expression"][0])
            # import pdb; pdb.set_trace()
            self.filldata(i, full_tokens, gg_mtx, raw_exp, side = 0)
            # import pdb; pdb.set_trace()
            #for right
            full_tokens = torch.tensor(sample["right_Full_Tokens"])
            row = sample["right_row"]
            col = sample["right_col"]
            data = sample["right_data"]
            shape = sample["right_shape"]
            gg_mtx = self.rebuild_adj(data, row, col, shape)
            gg_mtx = torch.tensor(gg_mtx)
            raw_exp = torch.tensor(sample["right_Expression"][0])
            self.filldata(i, full_tokens, gg_mtx, raw_exp, side = 1)
            # import pdb; pdb.set_trace()







class CustomFastDataCollator(object):
    def __init__(self, directionality, context_length = 1000, padding_idx=0, special_token_num = 4, n_bins = 51, sep_token = 1949, cls_token = 1, load_save = True):
        """
        load_save: load the saved dataset, if False, the left and right dataset should be loaded without combination, which is more straightforward and effecient!!!
        """
        self.context_length = context_length
        self.padding_idx = padding_idx
        self.directionality = directionality
        self.special_token_num = special_token_num
        self.n_bins = n_bins
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pair1_length = None
        self.batch = None
        self.full_tokens = None
        self.full_exp = None
        self.norm_exp = None
        self.load_save = load_save
    def __call__(self, batch):
        '''
        batch is the index and label, we need to get the rows of dataset
        '''

        
        #define which data you want to extract from the dataset
        
        self.full_tokens = torch.full((len(batch), self.context_length), self.padding_idx, dtype=torch.int)#, device = device)
        self.token_type_ids =  torch.full((len(batch), self.context_length), self.padding_idx, dtype=torch.int)#, device = device)
        self.full_exp = torch.ones((len(batch), self.context_length), dtype=torch.int)#, device = device)
        self.norm_exp = torch.ones((len(batch), self.context_length),dtype=torch.float)#, device = device)
        self.gg_mtx = torch.full((len(batch), self.context_length, self.context_length), self.padding_idx, dtype=torch.float)#, device = device)
        #fill the data 
        if self.load_save:
            self.one_site(batch)
            pair_labels =  torch.tensor([data["Labels"] for data in batch])
            left_cell_ids =  [data["left_Cell_Ids"] for data in batch]
            right_cell_ids =  [data["right_Cell_Ids"] for data in batch]
        else:
            self.one_site_fast(batch)
            pair_labels =  torch.tensor([data[2] for data in batch], dtype=torch.int64)
            left_index = torch.tensor([data[3] for data in batch], dtype=torch.int64)
            right_index = torch.tensor([data[4] for data in batch], dtype=torch.int64)
        

        # Filtering: Keep samples with at least 5 non-padding elements
        valid_mask = (self.full_tokens != self.padding_idx)  # Create a mask for non-padding elements
        non_padding_counts = valid_mask.sum(dim=1)  # Count non-padding elements for each sample
        # Filter based on the count of non-padding elements
        non_zero_indices = non_padding_counts >= 5  # Keep only samples with at least 5 non-padding elements
        gg_mtx_nonzero_mask = (self.gg_mtx.sum(dim=(1, 2)) != 0)
        non_zero_indices = non_zero_indices & gg_mtx_nonzero_mask

        # Check if there are valid samples
        if not non_zero_indices.any():
            print("Skipping batch: no valid samples with at least 5 non-padding elements.")
            return None 

        # Apply the filter to all relevant tensors
        self.full_tokens = self.full_tokens[non_zero_indices]
        self.token_type_ids = self.token_type_ids[non_zero_indices]
        self.full_exp = self.full_exp[non_zero_indices]
        self.gg_mtx = self.gg_mtx[non_zero_indices]
        self.norm_exp = self.norm_exp[non_zero_indices]
        pair_labels = pair_labels[non_zero_indices]
        left_index = left_index[non_zero_indices]
        right_index = right_index[non_zero_indices]


        # Pad sequences after filling the data
        attention_masks = (self.full_tokens != self.padding_idx).bool()
        # print(attention_masks.sum())
        # pair_labels =  torch.tensor(self.pair_labels)
        # import pdb; pdb.set_trace()
        return {
            'adjmtx': self.gg_mtx,
            'indices': self.full_tokens,
            'attention_mask': attention_masks,
            'normalized_exp': self.norm_exp,
            "Expression": self.full_exp,
            "pair_label": pair_labels,
            "token_type_ids": self.token_type_ids,
            "left_index": left_index,
            "right_index": right_index
        }
    def filldata(self, sample_index, full_tokens, gg_mtx, raw_exp, side):

        # full_tokens = torch.tensor(batch["Full_Tokens"][0])
        # gg_mtx = torch.tensor(batch["Gene_Gene_Matrix"][0])
        # raw_exp = torch.tensor(batch["Expression"][0][0])

        #make sure all the data shorter than the context_length
        if len(full_tokens) > self.context_length:
            full_tokens = full_tokens[:self.context_length]
        if gg_mtx.shape[0] > self.context_length:
            gg_mtx = gg_mtx[:self.context_length, :self.context_length]
        if len(raw_exp) > self.context_length:
            raw_exp = raw_exp[:self.context_length]
        # import pdb; pdb.set_trace()
        self.full_exp[sample_index,1: raw_exp.size(0)+1] = raw_exp #1 for cls token
        # import pdb; pdb.set_trace()
        cls_site = 1
        sep_site = 1
        current_size = gg_mtx.shape[0]
        if side == 0: #for the left pair
            # import pdb; pdb.set_trace()
            prefix_length = cls_site + self.special_token_num
            self.pair1_length = cls_site + full_tokens.size(0)-(4-self.special_token_num)
            #adding the cls token first
            self.full_tokens[sample_index,0] = self.cls_token
            #adding the tokens 
            self.full_tokens[sample_index,cls_site:self.pair1_length] = full_tokens[4-self.special_token_num:] #add the special token for the left and right sequence
            #adding the sep in the middle of pair
            self.full_tokens[sample_index, self.pair1_length] = self.sep_token
            #adding the gene pair matrix
            self.last_mtx_length = gg_mtx.shape[0]
            self.gg_mtx[sample_index, prefix_length:(prefix_length+current_size), prefix_length:(prefix_length+current_size)] = gg_mtx

            self.token_type_ids[sample_index, :self.pair1_length+cls_site] = 1
        elif side == 1: #for the right pair
            # import pdb; pdb.set_trace()
            prefix_length = cls_site + self.special_token_num
            
            #adding the right cell
            start = (self.pair1_length + sep_site)
            seq_length = full_tokens.size(0)-(4-self.special_token_num)
            self.full_tokens[sample_index, start:(start + seq_length)] = full_tokens[4-self.special_token_num:]
            #add the sep to the right end
            self.full_tokens[sample_index, (start + seq_length): (start + seq_length + sep_site) ] = self.sep_token
            #add the gene pairs to the right cell
            self.gg_mtx[sample_index, (prefix_length + self.last_mtx_length + cls_site) : (prefix_length + self.last_mtx_length + cls_site + current_size), (prefix_length + self.last_mtx_length + cls_site) : (prefix_length + self.last_mtx_length + cls_site + current_size)] = gg_mtx

            self.token_type_ids[sample_index, start:(start + seq_length + sep_site)] = 2
        # import pdb; pdb.set_trace()

            
    def one_site_fast(self, dataset_batch):

        for i, (left_sample, right_sample, label, left_index, right_index) in enumerate(dataset_batch):

            #for left
            # import pdb; pdb.set_trace()
            full_tokens = torch.tensor(left_sample["Full_Tokens"])
            gg_mtx = torch.tensor(left_sample["Gene_Gene_Matrix"])
            raw_exp = torch.tensor(left_sample["Expression"][0])
#             print(raw_exp)
            # import pdb; pdb.set_trace()
#             print(full_tokens)
            self.filldata(i, full_tokens, gg_mtx, raw_exp, side = 0)
            # import pdb; pdb.set_trace()
            #for right
            full_tokens = torch.tensor(right_sample["Full_Tokens"])
            gg_mtx = torch.tensor(right_sample["Gene_Gene_Matrix"])
            raw_exp = torch.tensor(right_sample["Expression"][0])
            self.filldata(i, full_tokens, gg_mtx, raw_exp, side = 1)
            # import pdb; pdb.set_trace()


class CustomSingleDataCollator(object):
    def __init__(self, context_length, cls_token, sep_token, special_token_num, padding_idx=0):
        self.context_length = context_length
        self.padding_idx = padding_idx
        self.cls_token = torch.tensor([cls_token])
        self.sep_token = torch.tensor([sep_token])
        self.special_token_num = special_token_num
    def __call__(self, batch):
        # Extract sequences and matrices
        gg_mtx = [torch.tensor(item['Gene_Gene_Matrix']) for item in batch]
        Full_Tokens = [torch.tensor(item['Full_Tokens']) for item in batch]
        raw_exp = [torch.tensor(item['Expression'][0]) for item in batch]
        annotations = [item['Annotations'] for item in batch]
        
        niche_annotations = [item['Niche_Annotations'] for item in batch]
        
        pair_label = torch.tensor([float('nan') for item in batch]) #placehold for pair label
        tissues = [item['Tissues'] for item in batch]
        conditions = [item['Conditions'] for item in batch]

        full_tokens = torch.full((len(Full_Tokens), self.context_length), self.padding_idx, dtype=torch.int)
        for i, v in enumerate(Full_Tokens):
            if not self.cls_token:
                full_tokens[i,:v.size(0)-(4-self.special_token_num)] = v[4-self.special_token_num:] #withour cls and sep token
            else:
                full_tokens[i,:v.size(0)-(4-self.special_token_num)+len(self.cls_token)+len(self.sep_token)] = torch.cat([self.cls_token, v[4-self.special_token_num:], self.sep_token]) #add cls and sep to the token
        
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
            if not self.cls_token:
                gg_mtx_p[i, self.special_token_num: (current_size+self.special_token_num), self.special_token_num:(self.special_token_num+current_size)] = mat#add head and tail
            else:
                gg_mtx_p[i, len(self.cls_token)+self.special_token_num:(len(self.cls_token)+current_size+self.special_token_num), len(self.cls_token)+self.special_token_num:(len(self.cls_token)+self.special_token_num+current_size)] = mat#add head and tail

        # import pdb; pdb.set_trace()
        # print("adjmtx sum", gg_mtx_p.sum(dim=(1,2)))
        # print("indices sum", (full_tokens !=0).sum(dim=1))
        # print("attention_masks sum", attention_masks.sum(dim=1))
        # print("token_type_ids sum", token_type_ids.sum(dim=1))
        return {
            'adjmtx': gg_mtx_p,
            'indices': full_tokens,
            'attention_mask': attention_masks,
            "Expression": full_exp,
            "token_type_ids": token_type_ids,
            "Annotations": annotations,
            "pair_label": pair_label,
            "Niche_Annotations": niche_annotations,
            "Tissues": tissues,
            "Conditions": conditions
        }

           
def binary_to_coo_matrix(example):
    adj_mtx = np.array(example["Gene_Gene_Matrix"])
    # import pdb; pdb.set_trace()
    # Find the indices where the elements are non-zero
    row, col = np.nonzero(adj_mtx)

    # Gather the non-zero elements. Since it's a binary matrix, these will all be 1s.
    data = adj_mtx[row, col]
    shape = adj_mtx.shape
    # Create the COO format sparse matrix
    # sparse_matrix = coo_matrix((data, (row, col)), shape=binary_matrix.shape)

    return {"row": row, "col": col, "data": data, "shape": shape}


def save_pair_dataset(dataset, sample_cell_index, radius, num_workers, chunk, res):
    '''
    dataset: huggingface dataset
    index_path: the index build for huggingface dataset for fast retrive
    radius: the radius for getting the cell connection
    
    '''

    combined_dataset_all = concatenate_datasets([dataset["train"], dataset["test"], dataset["validation"]])
    print("building all the datasets from all the samples!!!")
    # import pdb; pdb.set_trace()
    num_sample = len(list(sample_cell_index.keys()))
    target_dirs = os.listdir("/scratch/project_465001027/Spatialformer/cache")
    bs = 6
    pair_files = [file for file in target_dirs if "pair" in file]
    file_left = [file for file in list(sample_cell_index.keys()) if "xenium_" + file + "_pair" not in pair_files]
    # import pdb; pdb.set_trace()
    # for sample_name in tqdm(list(sample_cell_index.keys())[(chunk-1)*bs + res:min(chunk*bs, num_sample)]):
    for sample_name in ['Xenium_V1_hColon_Non_diseased_Base_FFPE_outs', 'Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs', 'Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs'][2:3]:
        save_path = f"xenium_{sample_name}_pair"
        # import pdb; pdb.set_trace()
        if save_path not in target_dirs:

            # if save_path not in target_dirs:
            print(f"running {save_path}")
            # Filter the dataset for rows corresponding to the current sample_name
            sample_index = list(sample_cell_index[sample_name].values())
            sample_data = combined_dataset_all.select(sample_index)
            
            sparse_adjmtx,cell_ids = get_adj(sample_data, radius = radius, plot = False)
            # import pdb; pdb.set_trace()
            sample_data = sample_data.select_columns(["Full_Tokens","Gene_Gene_Matrix","Expression","Cell_Ids"])
            # import pdb;pdb.set_trace()
            sample_data = sample_data.map(binary_to_coo_matrix, num_proc = num_workers)
            sample_data = sample_data.remove_columns("Gene_Gene_Matrix")
            Pairs = GetPairs(sparse_adjmtx, num_workers = num_workers) #sample_index, (leftdataset, rightdataset), label
            
            # import pdb;pdb.set_trace()
            all_left_idxs = list(map(lambda x: x[0],Pairs.all_pairs))
            all_right_idxs = list(map(lambda x: x[1],Pairs.all_pairs))
            # import pdb; pdb.set_trace()
            all_labels = Pairs.all_labels
            all_left_dataset = sample_data.select(all_left_idxs)
            all_right_dataset = sample_data.select(all_right_idxs)
            # import pdb; pdb.set_trace()
            left_renamed = all_left_dataset.rename_columns({col: f'left_{col}' for col in all_left_dataset.column_names})
            right_renamed = all_right_dataset.rename_columns({col: f'right_{col}' for col in all_right_dataset.column_names})
            # import pdb; pdb.set_trace
            # Concatenate the two datasets
            combined_dataset = concatenate_datasets([left_renamed, right_renamed], axis=1)
            combined_dataset = combined_dataset.add_column("Labels", all_labels)
            # combined_datasets.append(combined_dataset)
            combined_dataset.save_to_disk(f"/scratch/project_465001027/Spatialformer/cache/xenium_{sample_name}_pair", num_proc = 32)
            # import pdb; pdb.set_trace()
    

class CustomDataModule(LightningDataModule):
    def __init__(self, datapath, 
                        sample_dataset = None,
                        pair_index = None,
                        labels = None,
                        batch_size=32, 
                        num_workers = 8, 
                        directionality = True,
                        context_length = 500,
                        padding_idx = 0,
                        special_token_num = 4,
                        n_bins = 50,
                        sep_token = 1949,
                        cls_token = 1
                        ):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.dataset = None
        self.sample_dataset = sample_dataset
        self.pair_index = pair_index
        self.labels = labels
        # self.resume_index = 0
        self.collator = CustomDataCollator(directionality, context_length, padding_idx, special_token_num, n_bins, sep_token, cls_token)
        self.fast_collator = CustomFastDataCollator(directionality, context_length, padding_idx, special_token_num, n_bins, sep_token, cls_token, load_save=False)
        self.num_workers = num_workers
    # def setup(self, stage=None):
    #     print("stage:", stage)
    #     self.dataset = DynamicHuggingFaceDataset(self.datapath, 'train')
    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = DynamicHuggingFaceDataset(self.datapath, 'train')
            self.val_dataset = DynamicHuggingFaceDataset(self.datapath, 'val')
        elif stage == 'test':
            self.test_dataset = DynamicHuggingFaceDatasetFast(self.sample_dataset, self.pair_index, self.labels)
    def train_dataloader(self):
        # self.dataset = DynamicHuggingFaceDataset(self.datapath, 'train')
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=self.num_workers)

    def val_dataloader(self):
        # self.dataset = DynamicHuggingFaceDataset(self.datapath, 'val')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, collate_fn=self.fast_collator, num_workers=self.num_workers)

def create_dataloader_eval(datapath, num_workers, batch_size, directionality, 
                           context_length, padding_idx, special_token_num, 
                           n_bins, sep_token, cls_token, kfold = False, cur_fold = False, split = False, cell_id = False, shuffle=False):

    iter_dataset = DynamicHuggingFaceDatasetEval(datapath, kfold = kfold, cur_fold = cur_fold, split = split)
    collator = CustomDataCollator(directionality, context_length, padding_idx, special_token_num, n_bins, sep_token, cls_token, cell_id)
    dataloader = DataLoader(iter_dataset, batch_size = batch_size,collate_fn=collator, num_workers=num_workers, shuffle=shuffle)
    return dataloader

def create_dataloader_fast(sample_dataset, pair_index, labels, split, num_workers, batch_size, directionality, 
                           context_length, padding_idx, special_token_num, 
                           n_bins, sep_token, cls_token):

    iter_dataset = DynamicHuggingFaceDatasetFast(sample_dataset, pair_index, labels, split)
    collator = CustomFastDataCollator(directionality, context_length, padding_idx, special_token_num, n_bins, sep_token, cls_token, load_save=False)
    dataloader = DataLoader(iter_dataset,batch_size = batch_size,collate_fn=collator, num_workers=num_workers,pin_memory=True)
    return dataloader

def create_dataloader(datapath, num_workers, batch_size, directionality, context_length, padding_idx, special_token_num, n_bins, sep_token, cls_token):
    # Instantiate the training dataset
    # import pdb; pdb.set_trace()
    # train_dataset, val_dataset = CustomIterableDataset(datapath).get_all() #the interleave_datasets already support shuffle
    gpus = torch.cuda.device_count()
    num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
    total_gpus = gpus*num_nodes
    train_iter_dataset = DynamicHuggingFaceDataset2(datapath, split = "train", gpu_num = total_gpus)
    test_iter_dataset = DynamicHuggingFaceDataset2(datapath, split = "val", gpu_num = total_gpus)
    collator = CustomDataCollator(directionality, context_length, padding_idx, special_token_num, n_bins, sep_token, cls_token)
    train_dataloader = DataLoader(train_iter_dataset, batch_size = batch_size, collate_fn=collator, num_workers=num_workers, pin_memory=True)

    val_dataloader = DataLoader(test_iter_dataset, batch_size = batch_size, collate_fn=collator, num_workers=num_workers, pin_memory=True)


    return train_dataloader, val_dataloader

def create_dataloader2(datapath, num_workers, batch_size, directionality, context_length, padding_idx, special_token_num, n_bins, sep_token, cls_token):
    combined_dataset = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset4")  
    combined_dataset_all = concatenate_datasets([combined_dataset["train"], combined_dataset["test"], combined_dataset["validation"]])
    index_path = "/scratch/project_465001027/Spatialformer/data/sample_cell_index.pkl"
    sample_cell_index = get_index(combined_dataset, save_file = index_path)
    print("building all the datasets from all the samples!!!")
    num_sample = len(list(sample_cell_index.keys()))
    train_datasets = []
    test_datasets = []
    for sample_name in tqdm(list(sample_cell_index.keys())[:2]):
        # Filter the dataset for rows corresponding to the current sample_name
        sample_index = list(sample_cell_index[sample_name].values())
        sample_data = combined_dataset_all.select(sample_index)
        sparse_adjmtx,cell_ids = get_adj(sample_data, radius = 30, plot = False)
        # import pdb; pdb.set_trace()
        sample_data = sample_data.select_columns(["Full_Tokens","Gene_Gene_Matrix","Expression"])
        # import pdb;pdb.set_trace()
        Pairs = GetPairs(sparse_adjmtx) #sample_index, (leftdataset, rightdataset), label
        all_pairs = Pairs.all_pairs
        all_labels = Pairs.all_labels
        train_pairs, test_pairs, train_labels, test_labels = train_test_split(all_pairs, all_labels, test_size=0.2, random_state=42)

        train_dataset = BalanceDataset(sample_data, train_pairs, train_labels)
        test_dataset = BalanceDataset(sample_data, test_pairs, test_labels)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    all_train_datasets = ConcatDataset(train_datasets)
    all_test_datasets = ConcatDataset(test_datasets)
    # BalanceDataset
    collator = CustomDataCollator2(directionality, context_length, padding_idx, special_token_num, n_bins, sep_token, cls_token)
    train_dataloader = DataLoader(all_train_datasets, batch_size=batch_size, shuffle=True, collate_fn=collator)#, num_workers=num_workers, pin_memory = pin_memory)
    test_dataloader = DataLoader(all_test_datasets, batch_size=batch_size, shuffle=False, collate_fn=collator)#, num_workers=num_workers, pin_memory = pin_memory)

    # train_dataset, val_dataset = CustomIterableDataset(datapath).get_all() #the interleave_datasets already support shuffle
    return train_dataloader, test_dataloader
 
def create_single_data_loaders(tokenized_datasets, cls_token = 1, padding_idx = 0, sep_token = 1949, batch_size=1, context_length=500, special_token_num = 4, split_num = 2, num_workers = 1):


    data_collator = CustomSingleDataCollator(context_length, cls_token, sep_token, special_token_num, padding_idx)
    if split_num == 2:
        # Create DataLoaders
        
        


        # selected_indices = np.random.choice(tokenized_datasets.shape[0], size=tokenized_datasets.shape[0], replace=False)
        # tokenized_datasets = tokenized_datasets.select(selected_indices) 
        split_dataset = tokenized_datasets.train_test_split(test_size=0.001, seed=42)

        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
        # train_dataset = SingleHuggingFaceDataset(train_dataset)
        # val_dataset = SingleHuggingFaceDataset(val_dataset)
        train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        val_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return train_dataloader, val_dataloader
    elif split_num == 1:
        
        combined_dataloader = DataLoader(tokenized_datasets, collate_fn=data_collator, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return combined_dataloader





if __name__ == "__main__":

    combined_dataset = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset4")  
    index_path = "/scratch/project_465001027/Spatialformer/data/sample_cell_index.pkl"
    sample_cell_index = get_index(combined_dataset, save_file = index_path)
    train_dataloader, test_dataloader = save_pair_dataset(
                                                            combined_dataset, 
                                                            sample_cell_index, 
                                                            radius = 30, 
                                                            num_workers = 32,
                                                            chunk = 9,
                                                            res = 1
                                                            )
    # datapath = "/scratch/project_465001027/Spatialformer/cache"
    # train_dataloader, val_dataloader = create_dataloader(datapath, 
    #                                                          num_workers = 4, 
    #                                                          batch_size = 8,
    #                                                         directionality = True,
    #                                                         context_length = 500, 
    #                                                         padding_idx = 0, 
    #                                                         special_token_num = 4, 
    #                                                         n_bins = 51, 
    #                                                         sep_token = 1949, 
    #                                                         cls_token = 1)
    # train_iter_dataset = DynamicHuggingFaceDataset(datapath)

    # for batch in train_iter_dataset:
    #     import pdb; pdb.set_trace()

    # Iterate through each unique sample name






#Constructing the dataloader by samples, and 

# from torch.utils.data import Dataset, DataLoader
# class BalancedPairsDataset(Dataset):
#     def __init__(self, positive_pairs, negative_pairs):
#         self.positive_pairs = positive_pairs
#         self.negative_pairs = negative_pairs
#         self.total_pairs = min(len(positive_pairs), len(negative_pairs))

#     def __len__(self):
#         return self.total_pairs * 2  # Equal number of positive and negative pairs

#     def __getitem__(self, idx):
#         if idx % 2 == 0:
#             # Return a positive pair
#             i = idx // 2
#             pair = self.positive_pairs[i % len(self.positive_pairs)]
#             label = 1
#         else:
#             # Return a negative pair
#             i = idx // 2
#             pair = self.negative_pairs[i % len(self.negative_pairs)]
#             label = 0
#         return torch.tensor(pair), 
    



# def create_data_loaders(tokenized_datasets, batch_size=1, context_length=1500, special_token_num = 4, split_num = 2, directionality = True, n_bins = 51):
#     '''
    
#     directionality: whether the pair-wise matrix should have the directionality. On the other word, the whether the token that is defined as co-localized
#                     can have attention with all the other tokens. If so, this could be a fully attention matrix. If not, this should be a sparse binary matrix.
#                     default: True

#     '''
#     # Create a Data Collator for batching
#     class CustomDataCollator(object):
#         def __init__(self, context_length, padding_idx=0):
#             self.context_length = context_length
#             self.padding_idx = padding_idx
#             self.special_token_num = special_token_num
#             # self.selection = selection

#         def __call__(self, batch):
#             # Extract sequences and matrices
#             # import pdb; pdb.set_trace()
#             # if self.selection != None:
#             #     batch = [torch.tensor(item['Gene_Gene_Matrix']) for item in batch if item["Full_Tokens"]]
            
#             gg_mtx = [torch.tensor(item['Gene_Gene_Matrix']) for item in batch]
#             Full_Tokens = [torch.tensor(item['Full_Tokens']) for item in batch]
#             raw_exp = [torch.tensor(item['Expression'][0]) for item in batch]
#             # annotation = [item['Annotations'] for item in batch]
#             # niche_annotation = [item['Niche_Annotations'] for item in batch]
#             # Norm_Exp = [torch.tensor(item['Normalized_Exp']) for item in batch]
#             raw_genes = [item["Gene"] for item in batch]
#             raw_exps = [item["Expression"] for item in batch]
#             ranked_genes = [item["Ranked_Gene_Names"] for item in batch]
#             # nuc_pct = [item["pct_nucleus"] for item in batch]
#             # rank_nuc_pct = [torch.tensor([nuc_pct[i][raw_genes[i].index(gene)] for gene in gene_list]) for i,gene_list in enumerate(ranked_genes)] #getting the nucleus expression percentage level


#             # import pdb; pdb.set_trace()
#             # ranked_exp = get_rank_exp(raw_genes, raw_exps, ranked_genes)
#             ranked_exp = [torch.tensor([raw_exps[i][0][raw_genes[i].index(gene)] for gene in gene_list]) for i,gene_list in enumerate(ranked_genes)] #getting the ranked expression level
#             # import pdb; pdb.set_trace()


#             # dis_mtx = [torch.tensor(item['Distance_Matrix']) for item in batch]
            
#             # import pdb; pdb.set_trace()
#             full_tokens = torch.full((len(Full_Tokens), self.context_length), self.padding_idx, dtype=torch.int)
#             for i, v in enumerate(Full_Tokens):
#                 full_tokens[i,:v.size(0)-(4-special_token_num)] = v[4-special_token_num:]
            
#             # import pdb; pdb.set_trace()
#             full_exp = torch.full((len(Full_Tokens), self.context_length), self.padding_idx, dtype=torch.int)
#             for i, v in enumerate(raw_exp):
#                 full_exp[i,: v.size(0)] = v
            


#             norm_exp = torch.full((len(ranked_exp), self.context_length), self.padding_idx, dtype=torch.float)
#             try:
#                 for i, e in enumerate(ranked_exp):
#                     # import pdb; pdb.set_trace()
#                     e = binning(
#                         row=e,
#                         n_bins=n_bins,
#                     )

#                     norm_exp[i,self.special_token_num:self.special_token_num+e.size(0)] = e
#             except:
#                 import pdb; pdb.set_trace()
#                 pass

#             # Pad sequences
#             attention_masks = (full_tokens != self.padding_idx).bool()


#             # Pad 2D matrices
#             gg_mtx_p = torch.full((len(gg_mtx), self.context_length, self.context_length), self.padding_idx, dtype=torch.float)
#             for i, mat in enumerate(gg_mtx):
#                 current_size = mat.shape[0]
#                 gg_mtx_p[i, self.special_token_num:(current_size+self.special_token_num), self.special_token_num:(self.special_token_num+current_size)] = mat

#             if not directionality:
#                 rows_with_ones = torch.any(gg_mtx_p == 1, dim=2)
#                 cols_with_ones = torch.any(gg_mtx_p == 1, dim=1)

#                 # Expand dimensions to match the shape of the original matrix for broadcasting
#                 rows_with_ones = rows_with_ones.unsqueeze(2)
#                 cols_with_ones = cols_with_ones.unsqueeze(1)

#                 # Set the entire row and column to 1 if there is at least one 1
#                 gg_mtx_p = torch.logical_or(rows_with_ones, cols_with_ones)
#                 gg_mtx_p = gg_mtx_p.int()

#             return {
#                 'adjmtx': gg_mtx_p,
#                 'indices': full_tokens,
#                 'attention_mask': attention_masks,
#                 'normalized_exp': norm_exp,
#                 "Expression": full_exp

#             }
    

#     data_collator = CustomDataCollator(context_length, padding_idx=0)
#     if split_num == 2:
#         # Create DataLoaders
#         # train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, collate_fn=data_collator, shuffle=True)
#         val_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size)
#         # test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=batch_size)
#         combined_dataset = concatenate_datasets([tokenized_datasets["train"], tokenized_datasets["test"]])
#         train_dataloader = DataLoader(combined_dataset, collate_fn=data_collator, batch_size=batch_size)
#         return train_dataloader, val_dataloader
#     elif split_num == 1:
        
#         combined_dataloader = DataLoader(tokenized_datasets, collate_fn=data_collator, batch_size=batch_size)
#         return combined_dataloader

