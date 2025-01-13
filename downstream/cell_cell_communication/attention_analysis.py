#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("/scratch/project_465001027/Spatialformer/scripts")
sys.path.append("/scratch/project_465001027/Spatialformer/utils")
from train import manual_train_fm
import os
import json
import torch
from data_loader import create_dataloader_eval
from datasets import load_from_disk
import pickle
from utils import *
import json
from tqdm import tqdm
import numpy as np
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_ann(this_batch, sample_name, sample_cell_index, anns):
    # import pdb; pdb.set_trace()
    left_index = sample_cell_index[sample_name][this_batch["left_cell_ids"][0]]
    right_index = sample_cell_index[sample_name][this_batch["right_cell_ids"][0]]
    left_ct = anns[left_index]["Annotations"]
    right_ct = anns[right_index]["Annotations"]
    return (left_ct, right_ct) 


def get_vocab():
    with open("/scratch/project_465001027/Spatialformer/tokenizer/tokenv4.json", "r") as f:
        vocab = json.load(f)
        vocab = {j:i for i,j in vocab.items()}
    return vocab



def delete_pair(batch, index, batch_size):
    
    new_batch_ptb = {}
    batches = {}
    combination_tokens = []
    combination_indexs = []
    indices = batch["indices"][index]
    token_type_id = batch["token_type_ids"][index]
    mask_attentions = batch['attention_mask'][index]
    left_real_sequence = indices[token_type_id == 1][5:]#include the cls token
    right_real_sequence = indices[token_type_id == 2][4:]#no cls token, only meta
    pad_sequence = indices[token_type_id == 0]
    counter = 0
    cb_counter = 0
    batch_idx = 0
    bc_indices = []
    bc_token_ids = []
    bc_mask_attentions = []
    bc_ptb_pairs = []
    for i in range(len(left_real_sequence)):
        for j in range(len(right_real_sequence)):   
            del_left_token = left_real_sequence[i]
            del_right_token = right_real_sequence[j]
            # Create masks for each range
            mask_left = ~torch.isin(indices[token_type_id == 1], torch.tensor(del_left_token))  # Mask for indices 0-10
            mask_right = ~torch.isin(indices[token_type_id == 2], torch.tensor(del_right_token))  # Mask for indices 11-20
            # Combine masks combine left right and pad mask
            mask = torch.cat((mask_left, mask_right, pad_sequence +1)) #pad mask is int0, we plus 1 here
            mask = mask.type(torch.bool) #convert to bool for filtering
            short_indices = indices[mask]
            short_token_type_id = token_type_id[mask]
            short_mask_attentions = mask_attentions[mask]
            #padding for indice
            num_zeros_to_add = indices.size(0) - short_indices.size(0)
            zeros = torch.zeros(num_zeros_to_add, dtype=indices.dtype)
            new_indices = torch.cat((short_indices, zeros)).unsqueeze(0)
            new_token_type_id = torch.cat((short_token_type_id, zeros)).unsqueeze(0)
            new_mask_attentions = torch.cat((short_mask_attentions, zeros))
            new_mask_attentions = new_mask_attentions.type(torch.bool).unsqueeze(0)
            ptb_pairs = torch.tensor([i,j]).unsqueeze(0)
            
            
            bc_indices.append(new_indices)
            bc_token_ids.append(new_token_type_id)
            bc_mask_attentions.append(new_mask_attentions)
            bc_ptb_pairs.append(ptb_pairs)
            counter = len(bc_ptb_pairs)
            if counter == batch_size:
                batch_idx += 1
                new_batch_ptb["indices"] = torch.cat(bc_indices)
                new_batch_ptb["token_type_ids"] = torch.cat(bc_token_ids)
                new_batch_ptb["attention_mask"] = torch.cat(bc_mask_attentions)
                new_batch_ptb["perturbation_gene_pairs"] = torch.cat(bc_ptb_pairs)
                #append to all batch
                batches[batch_idx] = new_batch_ptb
                bc_indices = []
                bc_token_ids = []
                bc_mask_attentions = []
                bc_ptb_pairs = []
                new_batch_ptb = {}
                counter = 0
                
            token_pairs = torch.tensor([[del_left_token.item(),del_right_token.item()]])
            index_pairs = torch.tensor([[i,j]])
            combination_tokens.append(token_pairs)
            combination_indexs.append(index_pairs)
            
    
    #append to all batch
    combination_tokens = torch.cat(combination_tokens)
    combination_indexs = torch.cat(combination_indexs)
    #for the last batch
    if combination_tokens.shape[0] != batch_idx*batch_size:
        batch_idx += 1
        new_batch_ptb["indices"] = torch.cat(bc_indices)
        new_batch_ptb["token_type_ids"] = torch.cat(bc_token_ids)
        new_batch_ptb["attention_mask"] = torch.cat(bc_mask_attentions)
        new_batch_ptb["perturbation_gene_pairs"] = torch.cat(bc_ptb_pairs)
        batches[batch_idx] = new_batch_ptb
#     combinations = np.array(combinations)
    return batches, combination_tokens, combination_indexs


def get_pairs(dataloader, sample_name, sample_cell_index, anns, ct1="T-cells", ct2="Macrophages", pair_num=100, crop_cell_ids = None):
    all_infos = {}
    counter = 0
    for i, batch in tqdm(enumerate(dataloader)):
        all_diff = []
        left_cell_id = batch["left_cell_ids"][0]
        right_cell_id = batch["right_cell_ids"][0]
        #whether defined the cropped cell ids
        if crop_cell_ids is not None:
            if left_cell_id not in crop_cell_ids:
                continue  # Skip this batch if the left cell ID is not in cropped cell IDs
        # import pdb; pdb.set_trace()
        ct_pair = get_ann(batch, sample_name, sample_cell_index, anns = anns)
        label = batch["pair_label"].item()
        if ct1 in ct_pair and ct2 in ct_pair and label == 1:
            counter += 1
            if counter <= pair_num:
                _, ref_pair_prob = model.get_embeddings(batch, [-1], True, False) #normal prob
                batches,combination_tokens,combination_indexs  = delete_pair(batch = batch, index = 0, batch_size = 8)
                with torch.no_grad():  # Disable gradient tracking
                    # Convert the dictionary items to a list
                    # Process the batch
                    for batch_idx, ptb_batch in tqdm(batches.items(), total = len(batches)):
                        _, ptb_pair_prob = model.get_embeddings(ptb_batch, [-1], True, False) #perturbated prob

                        diff = ptb_pair_prob[:, 1] - ref_pair_prob[:, 1]
                        #to cpu
                        diff = diff.cpu().numpy()
                        all_diff.append(diff)
                #concat all the differences
                all_diffs = np.hstack(all_diff)
                combination_tokens = combination_tokens.cpu().numpy()
                combination_indexs = combination_indexs.cpu().numpy()
                all_infos.setdefault((left_cell_id, right_cell_id),{}).setdefault("diff",all_diffs)
                all_infos.setdefault((left_cell_id, right_cell_id),{}).setdefault("combination_tokens",combination_tokens)
                all_infos.setdefault((left_cell_id, right_cell_id),{}).setdefault("combination_indexs",combination_indexs)
                all_infos.setdefault((left_cell_id, right_cell_id),{}).setdefault("s_r",ct_pair)
            else:
                break
        else:
            continue
    return all_infos

get_file_path = lambda path, filename: os.path.join("/scratch/project_465001027/Spatialformer", path, filename)

def load_model(model_ckp_path):
    config_path = get_file_path("config", "_config_train_large_pair.json")
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    # model_ckp_path = "/scratch/project_465001027/Spatialformer/output/checkpoints/step=0100000-train_total_loss=-2.2727-val_total_loss=0.0000.ckpt"
    model = manual_train_fm(config = config)
    ckp = torch.load(model_ckp_path, map_location=torch.device(device))
    params = ckp["state_dict"]
    model.load_state_dict(params)
    model.eval()
    model.to(device)
    return model


if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="Get the pair genes for specific cell types")
    # Adding arguments
    parser.add_argument('-ct1', '--cell_type1', type=str, required=True, help='The first cell types that you want to investigare gene-gene interaction.')
    parser.add_argument('-ct2', '--cell_type2', type=str, required=True, help='The second cell types that you want to investigare gene-gene interaction.')
    parser.add_argument('-pair_num', '--pair_number', type=int, required=True, default =100, help='number of cell pairs belong to both cell type1 or cell type2. be careful, the larger the value you set, the longer time it will run. suggest 100 as the beginning')
    parser.add_argument('-cci', '--crop_cell_id_path', type=str, required=False, default = None, help='a file path of the array like cell ids')
    # Parse the arguments
    args = parser.parse_args()

    if args.crop_cell_id_path is not None:
        crop_cell_ids = np.load(args.crop_cell_id_path, allow_pickle=True)
    #Configure the dataloader
    datapath = "/scratch/project_465001027/Spatialformer/cache/xenium_VUILD96MF_pair"
    # model_ckp_path = "/scratch/project_465001027/Spatialformer/output/checkpoints/step=0100000-train_total_loss=-2.2727-val_total_loss=0.0000.ckpt" #one slide
    model_ckp_path = "/scratch/project_465001027/Spatialformer/output/checkpoints/step=0044000-train_total_loss=-1.3226-val_total_loss=0.2488.ckpt"
    train_sample = "VUILD96MF"
    dataloader = create_dataloader_eval(datapath, 
                                        num_workers = 0, 
                                        batch_size = 1, #this must be 1
                                        directionality = True,
                                        context_length = 500, 
                                        padding_idx = 0, 
                                        special_token_num = 4, 
                                        n_bins = 51, 
                                        sep_token = 1949, 
                                        cls_token = 1,
                                        cell_id = True)
    model = load_model(model_ckp_path)
    
    #loading the sample dataset and paired dataset
    xenium_VUILD102LF_pair = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_VUILD102LF_pair")
    combined_dataset = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset4")  
    index_path = "/scratch/project_465001027/Spatialformer/data/sample_cell_index.pkl"
    sample_cell_index = get_index(combined_dataset, save_file = index_path)
    combined_dataset_all = concatenate_datasets([combined_dataset["train"], combined_dataset["test"], combined_dataset["validation"]])
    sample_dataset = combined_dataset_all.select(list(sample_cell_index[train_sample].values()))
    anns = combined_dataset_all.select_columns(["Annotations"])
    #getting all the paired information
    # all_infos = get_pairs(dataloader, train_sample, sample_cell_index, anns, ct1="T-cells", ct2="Macrophages", cell_num=100)
    all_infos = get_pairs(dataloader, train_sample, sample_cell_index, anns, ct1=args.cell_type1, ct2=args.cell_type2, pair_num=args.pair_number, crop_cell_ids=crop_cell_ids)
    pickle.dump(all_infos, open("/scratch/project_465001027/Spatialformer/downstream/cell_cell_communication/data/all_infos_alllung_crop_id.pkl","wb"))



    #script:
    # python attention_analysis.py -ct1 T-cells -ct2 Macrophages -pair_num 100 -cci /scratch/project_465001027/Spatialformer/downstream/cell_cell_communication/data/VUILD96MF_croped_cell_id.npy
    