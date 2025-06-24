#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("/home/sxr280/Spatialformer_6_10/scripts")
sys.path.append("/home/sxr280/Spatialformer_6_10/utils")
from train import manual_train_fm
import os
import json
import torch
from data_loader import create_dataloader_eval
from datasets import load_from_disk, load_dataset
import pickle
from utils import *
# from utils.utils import GetPairs, get_adj, split_dataset
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




def delete_pair(batch, index, batch_size):
    # Extract relevant tensors only once
    indices = batch["indices"][index]
    token_type_id = batch["token_type_ids"][index]
    mask_attentions = batch['attention_mask'][index]

    left_real_sequence = indices[token_type_id == 1][5:]
    right_real_sequence = indices[token_type_id == 2][4:]
    pad_sequence = indices[token_type_id == 0]

    left_len = left_real_sequence.size(0)
    right_len = right_real_sequence.size(0)
    total_pairs = left_len * right_len

    # Precompute all token pairs and index pairs
    left_tokens = left_real_sequence.repeat_interleave(right_len)
    right_tokens = right_real_sequence.repeat(left_len)
    token_pairs = torch.stack([left_tokens, right_tokens], dim=1)
    index_i = torch.arange(left_len).repeat_interleave(right_len)
    index_j = torch.arange(right_len).repeat(left_len)
    index_pairs = torch.stack([index_i, index_j], dim=1)

    # Prepare for batching
    batches = {}
    combination_tokens = []
    combination_indexs = []

    # Precompute masks for all pairs
    mask_left_all = ~(left_real_sequence.unsqueeze(1) == left_tokens.unsqueeze(0)).any(dim=0)
    mask_right_all = ~(right_real_sequence.unsqueeze(1) == right_tokens.unsqueeze(0)).any(dim=0)

    for start in range(0, total_pairs, batch_size):
        end = min(start + batch_size, total_pairs)
        bc_indices = []
        bc_token_ids = []
        bc_mask_attentions = []
        bc_ptb_pairs = []
        for idx in range(start, end):
            del_left_token = left_tokens[idx]
            del_right_token = right_tokens[idx]

            # Create masks for each range
            mask_left = ~(left_real_sequence == del_left_token)
            mask_right = ~(right_real_sequence == del_right_token)
            mask = torch.cat((mask_left, mask_right, pad_sequence + 1)).type(torch.bool)
            short_indices = indices[mask]
            short_token_type_id = token_type_id[mask]
            short_mask_attentions = mask_attentions[mask]

            num_zeros_to_add = indices.size(0) - short_indices.size(0)
            zeros = torch.zeros(num_zeros_to_add, dtype=indices.dtype)
            new_indices = torch.cat((short_indices, zeros)).unsqueeze(0)
            new_token_type_id = torch.cat((short_token_type_id, zeros)).unsqueeze(0)
            new_mask_attentions = torch.cat((short_mask_attentions, zeros)).type(torch.bool).unsqueeze(0)
            ptb_pairs = torch.tensor([index_i[idx], index_j[idx]]).unsqueeze(0)

            bc_indices.append(new_indices)
            bc_token_ids.append(new_token_type_id)
            bc_mask_attentions.append(new_mask_attentions)
            bc_ptb_pairs.append(ptb_pairs)

            combination_tokens.append(token_pairs[idx].unsqueeze(0))
            combination_indexs.append(index_pairs[idx].unsqueeze(0))

        batch_idx = start // batch_size + 1
        new_batch_ptb = {
            "indices": torch.cat(bc_indices),
            "token_type_ids": torch.cat(bc_token_ids),
            "attention_mask": torch.cat(bc_mask_attentions),
            "perturbation_gene_pairs": torch.cat(bc_ptb_pairs),
        }
        batches[batch_idx] = new_batch_ptb

    combination_tokens = torch.cat(combination_tokens)
    combination_indexs = torch.cat(combination_indexs)
    return batches, combination_tokens, combination_indexs


def get_pairs(dataloader, sample_name, sample_cell_index, anns, ct1="T-cells", ct2="Macrophages", pair_num=100, crop_cell_ids=None):
    all_infos = {}
    counter = 0
    # Precompute set for fast lookup if crop_cell_ids is provided
    crop_cell_ids_set = set(crop_cell_ids) if crop_cell_ids is not None else None

    total = len(dataloader)
    for i, batch in tqdm(enumerate(dataloader), total=total):
        left_cell_id = batch["left_cell_ids"][0]
        right_cell_id = batch["right_cell_ids"][0]

        # Fast skip if not in crop_cell_ids
        if crop_cell_ids_set is not None and left_cell_id not in crop_cell_ids_set:
            continue

        ct_pair = get_ann(batch, sample_name, sample_cell_index, anns=anns)
        label = batch["pair_label"].item()
        if ct1 in ct_pair and ct2 in ct_pair and label == 1:
            counter += 1
            if counter > pair_num:
                break

            with torch.no_grad():
                _, ref_pair_prob = model.get_embeddings(batch, [-1], True, False)
                batches, combination_tokens, combination_indexs = delete_pair(batch=batch, index=0, batch_size=8)
                all_diff = []

                # Preallocate for efficiency
                ptb_probs = []
                for ptb_batch in batches.values():
                    _, ptb_pair_prob = model.get_embeddings(ptb_batch, [-1], True, False)
                    ptb_probs.append(ptb_pair_prob[:, 1].cpu().numpy())
                if ptb_probs:
                    all_diffs = np.hstack(ptb_probs) - ref_pair_prob[:, 1].cpu().numpy()[:, None]
                    all_diffs = all_diffs.flatten()
                else:
                    all_diffs = np.array([])

            all_infos[(left_cell_id, right_cell_id)] = {
                "diff": all_diffs,
                "combination_tokens": combination_tokens.cpu().numpy(),
                "combination_indexs": combination_indexs.cpu().numpy(),
                "s_r": ct_pair
            }
    return all_infos



def load_model(model_ckp_path, base_dir):
    get_file_path = lambda path, filename: os.path.join(base_dir, path, filename)
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
    parser.add_argument('-adata', '--anndata_path', type=str, required=False, default = None, help='Optional, in case your input is anndata from scanpy')
    parser.add_argument('-dataset', '--huggingface_dataset', type=str, required=False, default = None, help='Optional, in case you want to reproduce the result of the paper')
    parser.add_argument('-cci', '--crop_cell_id_path', type=str, required=False, default = "/home/sxr280/Spatialformer_6_10/downstream/cell_cell_communication/data/VUILD96MF_croped_cell_id.npy", help='a file path of the array like cell ids')
    parser.add_argument('-num_procs', '--num_process', type=int, required=True, default = 8, help='number of cpu to load/download the dataset from huggingface')
    parser.add_argument('-index', '--index_path', type=str, required=True, default = "/home/sxr280/Spatialformer/cache/data/sample_cell_index.pkl", help='a file path of the array like cell ids')
    parser.add_argument('-sample', '--sample_name', type=str, required=True, default = "VUILD96MF", help='The name of the sample if your input is the huggingface dataset')
    parser.add_argument('-ckp', '--checkpoint', type=str, required=True, default = "/home/sxr280/Spatialformer_lumi/output/checkpoints/61slides.ckpt", help='The checkpoint of the pair-wise cell dataset')
    parser.add_argument('-cache_dir', '--cache_dir', type=str, required=True, default = "/home/sxr280/Spatialformer/cache/", help='The path to store the cache')
    parser.add_argument('-base_dir', '--base_dir', type=str, required=True, default = "/home/sxr280/Spatialformer_6_10", help='The base path of the running code')
    # Parse the arguments
    args = parser.parse_args()
    
    adata = args.anndata_path
    datapath = args.huggingface_dataset
    index_path = args.index_path
    num_procs = args.num_process
    sample_name = args.sample_name
    cci = args.crop_cell_id_path
    pair_num=args.pair_number
    model_ckp_path = args.checkpoint
    cache_dir = args.cache_dir
    base_dir = args.base_dir

    if args.crop_cell_id_path is not None:
        crop_cell_ids = np.load(cci, allow_pickle=True)
    else:
        crop_cell_ids = None
    #Configure the dataloader

    
    
    
    #get the dataloader for pair-wise cells
    # dataset = load_from_disk(datapath)  
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
    
    
    
    model = load_model(model_ckp_path, base_dir)
    #loading the sample dataset and paired dataset
    combined_dataset = load_dataset("TerminatorJ/xenium_pandavid_dataset4", cache_dir = cache_dir, num_proc = num_procs)
    combined_dataset_all = concatenate_datasets([combined_dataset["train"], combined_dataset["test"], combined_dataset["validation"]])
    
    #get sample dataset
    sample_cell_index = get_index(combined_dataset_all, save_file = index_path)
    sample_dataset = combined_dataset_all.select(list(sample_cell_index[sample_name].values()))

    anns = sample_dataset.select_columns(["Annotations"])
    #getting all the paired information
    
    # all_infos = get_pairs(dataloader, train_sample, sample_cell_index, anns, ct1="T-cells", ct2="Macrophages", cell_num=100)
    import pdb; pdb.set_trace()
    all_infos = get_pairs(dataloader, sample_name, sample_cell_index, anns, ct1=args.cell_type1, ct2=args.cell_type2, pair_num=pair_num, crop_cell_ids=crop_cell_ids)
    pickle.dump(all_infos, open("/home/sxr280/Spatialformer/downstream/cell_cell_communication/data/all_infos_alllung_crop_id.pkl","wb"))



    #script:
    # python attention_analysis.py -ct1 T-cells -ct2 Macrophages -pair_num 100 -cci /scratch/project_465001027/Spatialformer/downstream/cell_cell_communication/data/VUILD96MF_croped_cell_id.npy
    