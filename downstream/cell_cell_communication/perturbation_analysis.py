#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("/home/sxr280/Spatialformer_6_10/scripts")
sys.path.append("/home/sxr280/Spatialformer_6_10/utils")
sys.path.append("/home/sxr280/Spatialformer_6_10/spatialformer")
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
import scanpy as sc
from tools import embed_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_ann(this_batch, sample_name, sample_cell_index, anns):
    # import pdb; pdb.set_trace()
    left_index = sample_cell_index[sample_name][this_batch["left_cell_ids"][0]]
    right_index = sample_cell_index[sample_name][this_batch["right_cell_ids"][0]]
    left_ct = anns[left_index]["Annotations"]
    right_ct = anns[right_index]["Annotations"]
    return (left_ct, right_ct) 




def delete_pair(batch, index, batch_size):
    '''
    
    Delete the pair of genes in the batch, and return the new batch with perturbation gene pairs.
    '''
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
    for i in tqdm(range(len(left_real_sequence))):
        for j in tqdm(range(len(right_real_sequence))):   
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
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        all_diff = []
        
        left_cell_id = batch["left_cell_ids"][0]
        right_cell_id = batch["right_cell_ids"][0]
        #whether defined the cropped cell ids
        if crop_cell_ids is not None:
            if left_cell_id not in crop_cell_ids:
                continue  # Skip this batch if the left cell ID is not in cropped cell IDs
        # import pdb; pdb.set_trace()
        try:
            ct_pair = get_ann(batch, sample_name, sample_cell_index, anns = anns)
        except Exception as e:
            # import pdb; pdb.set_trace()
            ct_pair = (anns[anns.index == left_cell_id].values[0], anns[anns.index == right_cell_id].values[0])
            
        label = batch["pair_label"].item()
        if ct1 in ct_pair and ct2 in ct_pair and label == 1:
            counter += 1
            print(f"Calculating the NO.{counter} pair")
            if counter <= pair_num:
                _, ref_pair_prob = model.get_embeddings(batch, [-1], True, False) #normal prob
                batches,combination_tokens,combination_indexs  = delete_pair(batch = batch, index = 0, batch_size = 8)
                with torch.no_grad():  # Disable gradient tracking
                    # Convert the dictionary items to a list
                    # Process the batch
                    for batch_idx, ptb_batch in tqdm(batches.items(), total = len(batches)):
                        # import pdb; pdb.set_trace()
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
    parser.add_argument('-ts', '--tissue', type=str, required=True, help='The tissue type of the input data.')
    parser.add_argument('-cd', '--condition', type=str, required=True, help='The tissue condition of the input data, e.g. healthy or diseased.')
    parser.add_argument('-rd', '--radius', type=int, required=True, default =10, help='The radius of defining the cell-cell colocalization.')
    parser.add_argument('-pair_num', '--pair_number', type=int, required=True, default =100, help='number of cell pairs belong to both cell type1 or cell type2. be careful, the larger the value you set, the longer time it will run. suggest 100 as the beginning')
    parser.add_argument('-adata', '--anndata_path', type=str, required=False, default = None, help='Optional, in case your input is anndata from scanpy. The anndata should contain the cell centroid in the obsm index that has the ["spatial"] key')
    parser.add_argument('-dataset', '--huggingface_dataset', type=str, required=False, default = None, help='Optional, in case you want to reproduce the result of the paper')
    parser.add_argument('-cci', '--crop_cell_id_path', type=str, required=False, default = "/home/sxr280/Spatialformer_6_10/downstream/cell_cell_communication/data/VUILD96MF_croped_cell_id.npy", help='a file path of the array like cell ids')
    parser.add_argument('-num_procs', '--num_process', type=int, required=False, default = 8, help='number of cpu to load/download the dataset from huggingface')
    parser.add_argument('-index', '--index_path', type=str, required=False, default = "/home/sxr280/Spatialformer/cache/data/sample_cell_index.pkl", help='a file path of the array like cell ids. set this if using the huggingface dataset')
    parser.add_argument('-sample', '--sample_name', type=str, required=False, default = "VUILD96MF", help='The name of the sample ')
    parser.add_argument('-ckp', '--checkpoint', type=str, required=False, default = "/home/sxr280/Spatialformer_lumi/output/checkpoints/61slides.ckpt", help='The checkpoint of the pair-wise cell dataset')
    parser.add_argument('-cache_dir', '--cache_dir', type=str, required=True, default = "/home/sxr280/Spatialformer/cache/", help='The path to store the cache')
    parser.add_argument('-base_dir', '--base_dir', type=str, required=True, default = "/home/sxr280/Spatialformer_6_10", help='The base path of the running code')
    parser.add_argument('-out_dir', '--output_dir', type=str, required=True, default = "/home/sxr280/Spatialformer/downstream/cell_cell_communication/data", help='The output path of the running code')
    # Parse the arguments
    args = parser.parse_args()
    
    adatapath = args.anndata_path
    datapath = args.huggingface_dataset
    index_path = args.index_path
    num_procs = args.num_process
    sample_name = args.sample_name
    cci = args.crop_cell_id_path
    pair_num=args.pair_number
    model_ckp_path = args.checkpoint
    cache_dir = args.cache_dir
    base_dir = args.base_dir
    out_dir = args.output_dir
    tissue = args.tissue
    condition = args.condition
    radius = args.radius
    sample_name = args.sample_name

    if args.crop_cell_id_path is not None:
        crop_cell_ids = np.load(cci, allow_pickle=True)
    else:
        crop_cell_ids = None
    #Configure the dataloader

    
    
    
    #get the dataloader for pair-wise cells
    # dataset = load_from_disk(datapath)  
    if datapath is not None:
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
        #loading the sample dataset and paired dataset
        combined_dataset = load_dataset("TerminatorJ/xenium_pandavid_dataset4", cache_dir = cache_dir, num_proc = num_procs)
        combined_dataset_all = concatenate_datasets([combined_dataset["train"], combined_dataset["test"], combined_dataset["validation"]])
        
        #get sample dataset
        sample_cell_index = get_index(combined_dataset_all, save_file = index_path)
        sample_dataset = combined_dataset_all.select(list(sample_cell_index[sample_name].values()))

        anns = combined_dataset_all.select_columns(["Annotations"])
    else:
        adata = sc.read_h5ad(adatapath)
        sparse_adj, cell_ids = get_adj(sample_dataset = None, anndata = adata, radius = radius, plot = False, sym = True)
        ### Getting the cell pairs according to the distance
        Pairs = GetPairs(sparse_adj, num_workers = 8) #assign the 1:1 negative to the positive

        all_pairs = Pairs.all_pairs
        all_labels = Pairs.all_labels

        print("all pairs:", all_pairs.shape)

        # subset the pairs according to the user requested number of pairs
        # selected_pairs, selected_labels = split_dataset(all_pairs, all_labels, n_splits = 0, test_size = None, zero_shot_cell_size = zero_shot_cell_size)
        # print("selected_pairs shape:", selected_pairs.shape)
        left_cells = adata.obs.index[all_pairs[:,0]]
        right_cells = adata.obs.index[all_pairs[:,1]]
        dataloader = embed_data(adata,
                tissue = tissue, 
                condition = condition,
                method = "gene",
                model_ckp_path = model_ckp_path, 
                batch_size = 1,
                mode = "pair",
                only_loader = True,
                left_cell = left_cells,
                right_cell = right_cells,
                pair_label = all_labels,
                num_workers = 8,
                reveal_name = False
                )

        anns = adata.obs["cell_type"]
        sample_cell_index = None
    
    model = load_model(model_ckp_path, base_dir)
    all_infos = get_pairs(dataloader, sample_name, sample_cell_index, anns = anns, ct1=args.cell_type1, ct2=args.cell_type2, pair_num=pair_num, crop_cell_ids=crop_cell_ids)
    pickle.dump(all_infos, open(os.path.join(out_dir, f"all_infos_crop_id_{args.cell_type1}_{args.cell_type2}_{sample_name}.pkl"),"wb"))



    # script for huggingface dataset as input:
    # python perturbation_analysis.py -ct1 T-cells -ct2 Macrophages -pair_num 100 -cci /scratch/project_465001027/Spatialformer/downstream/cell_cell_communication/data/VUILD96MF_croped_cell_id.npy
    
    
    # script for the anndata as input:
    # python perturbation_analysis.py -ct1 Epithelial -ct2 Myoepithelial -adata /home/sxr280/Spatialformer_6_10/downstream/cell_cell_communication/data/croped_sdata_Xenium_breast.h5ad -pair_num 100 -cci /home/sxr280/Spatialformer_6_10/downstream/cell_cell_communication/data/VUILD96MF_croped_cell_id.npy -ts Breast -cd Healthy -cache_dir /home/sxr280/Spatialformer/cache/ -base_dir /home/sxr280/Spatialformer_6_10 -rd 10
    