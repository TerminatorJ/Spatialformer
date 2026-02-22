#!/usr/bin/env python
# coding: utf-8

# # Cell-cell colocalization for Xenium breast sample

# We cannot use the script/fine-tune.py directly because the pipeline is built for the huggingface dataset as the input.     Any external new test dataset should be loaded via "tools.get_embeddings.py" function  
# - The breast cancer tumor microenvironment dataset can be downloaded via:  
# https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast.   
# - Publication DOI:   
# https://doi.org/10.1038/s41467-023-43458-x

# In[1]:


import sys
sys.path.append("/home/sxr280/Spatialformer")
sys.path.append("/home/sxr280/Spatialformer/train")
sys.path.append("/home/sxr280/Spatialformer/spatialformer/")
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.utils import GetPairs, get_adj, split_dataset
from fine_tune import FineTune
import json
from tools import embed_data
import argparse
import wandb


# Loading the Xenium count matrix, and cell metadata

#load the Xenium breast cancer dataset



def main(cell_by_gene_path = "/scratch/project_465001820/Spatialformer/data/Xenium/outs/cell_feature_matrix.h5",
         cell_meta_path = "/scratch/project_465001820/Spatialformer/data/Xenium/outs/cells.parquet",
         radius = 6,
         zero_shot_cell_size = 500,
         sample_name = "breast_cancer",
         fine_tune_mode = "zero_shot",
         config_path = "None",
         tissue = "Breast",
         condition = "Disease",
         max_cells = 100000,
         batch_size = 4,
         rank = 8,
         lora_alpha = 16):
    # Define split ratios and random seed
    
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    random_seed = 42
    np.random.seed(random_seed)

    if cell_by_gene_path.split(".")[1] == "h5":
        print(f"Loading the {tissue} dataset from h5 file")
        adata = sc.read_10x_h5(cell_by_gene_path)
        if max_cells:
            random_indices = np.random.choice(adata.n_obs, size=max_cells, replace=False)
            adata = adata[random_indices, ]
    elif cell_by_gene_path.split(".")[-1] == "csv":
        print(f"Loading the {tissue} dataset from csv file")
        adata = sc.read_csv(cell_by_gene_path, first_column_names=True)
        if max_cells:
            random_indices = np.random.choice(adata.n_obs, size=max_cells, replace=False)
            adata = adata[random_indices, ]

    # Get total number of cells and create shuffled indices
    n_cells = adata.n_obs
    
    indices = np.random.permutation(n_cells)

    # Calculate split points
    n_test = int(n_cells * test_ratio)
    n_val = int(n_cells * val_ratio)

    # Split indices randomly
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]

    # Split adata
    adata_test = adata[test_indices, :].copy()
    adata_val = adata[val_indices, :].copy()
    adata_train = adata[train_indices, :].copy()

    if cell_meta_path.split(".")[-1] == "parquet":
        print(f"Loading the {tissue} dataset metadata from parquet file")
        cell_meta = pd.read_parquet(cell_meta_path)
        spatial_cols = ["x_centroid", "y_centroid"]
        if max_cells:
            cell_meta = cell_meta.loc[random_indices]
    elif cell_meta_path.split(".")[-1] == "csv":
        print(f"Loading the {tissue} dataset metadata from csv file")
        cell_meta = pd.read_csv(cell_meta_path, index_col=0)
        spatial_cols = ["center_x", "center_y"]
        if max_cells:
            cell_meta = cell_meta.loc[random_indices, :]
    # Split cell_meta using the same indices
    cell_meta_test = cell_meta.iloc[test_indices]
    cell_meta_val = cell_meta.iloc[val_indices]
    cell_meta_train = cell_meta.iloc[train_indices]

    # Assign spatial coordinates
    adata_test.obsm["spatial"] = cell_meta_test[spatial_cols].values
    adata_val.obsm["spatial"] = cell_meta_val[spatial_cols].values
    adata_train.obsm["spatial"] = cell_meta_train[spatial_cols].values

    print(f"Train: {adata_train.n_obs}, Val: {adata_val.n_obs}, Test: {adata_test.n_obs}")
    #convert the data type of index into interger
    try:
        adata.obs.index = adata.obs.index.astype(int)
    except ValueError:
        print("Index is already in integer format or cannot be converted.")
    print(adata_test)
    #add the spatial coordinates to the anndata
    # import pdb; pdb.set_trace()

    with open(config_path, 'r') as json_file:
        config = json.load(json_file)


    
    if fine_tune_mode == "zero_shot":
        # Getting the dataloader

        adata_test.var["gene_name"] = adata_test.var.index

        # Getting the matrix of the cell neighbors
        # Getting the asymmetry matrix
        sparse_adj, cell_ids = get_adj(sample_dataset = None, anndata = adata_test, radius = radius, plot = False, sym = True)
        ### Getting the cell pairs according to the distance
        Pairs = GetPairs(sparse_adj, num_workers = 8) #assign the 1:1 negative to the positive

        all_pairs_test = Pairs.all_pairs
        all_labels_test = Pairs.all_labels

        print("all pairs:", all_pairs_test.shape)


        # selected_pairs_test, selected_labels_test = split_dataset(all_pairs, all_labels, n_splits = 0, test_size = None, zero_shot_cell_size = zero_shot_cell_size)
        # test_dataloader = data_prepare(sample_name, kfold, num_workers, batch_size, radius=r, test_size = None, zero_shot_cell_size = zero_shot_cell_size, split_mode = "random")

        # print("selected_pairs test shape:", selected_pairs_test.shape)

        left_cells_test = adata_test.obs.index[all_pairs_test[:,0]]
        right_cells_test = adata_test.obs.index[all_pairs_test[:,1]]
        test_dataloader = embed_data(adata_test,
                    tissue = tissue, 
                    condition = condition,
                    assay = assay,
                    method = "gene",
                    batch_size = batch_size,
                    mode = "pair",
                    only_loader = True,
                    left_cell = left_cells_test,
                    right_cell = right_cells_test,
                    pair_label = all_labels_test,
                    num_workers = 8,
                    reveal_name = False
                    )
        # Run the model to get the predictions
        
        Finetune = FineTune(config = config, 
                            sample_name = sample_name, 
                            radius = radius, 
                            fine_tune_mode = fine_tune_mode, 
                            wandb = True, 
                            strategy = "ddp",
                            rank = rank,
                            lora_alpha = lora_alpha)
        results = Finetune.test(test_dataloader)
    elif fine_tune_mode == "lora" or fine_tune_mode == "full_tune" or fine_tune_mode == "probe":
        # Getting the dataloader
        adata_train.var["gene_name"] = adata_train.var.index

        # Getting the matrix of the cell neighbors
        # Getting the asymmetry matrix
        sparse_adj, cell_ids = get_adj(sample_dataset = None, anndata = adata_train, radius = radius, plot = False, sym = True)
        ### Getting the cell pairs according to the distance
        Pairs = GetPairs(sparse_adj, num_workers = 8) #assign the 1:1 negative to the positive
        all_pairs_train = Pairs.all_pairs
        all_labels_train = Pairs.all_labels
        print("all pairs train:", all_pairs_train.shape)
        # test_dataloader = data_prepare(sample_name, kfold, num_workers, batch_size, radius=r, test_size = None, zero_shot_cell_size = zero_shot_cell_size, split_mode = "random")
        left_cells_train = adata_train.obs.index[all_pairs_train[:,0]]
        right_cells_train = adata_train.obs.index[all_pairs_train[:,1]]
        #get only dataloader
        train_dataloader = embed_data(adata_train,
                    tissue = tissue, 
                    condition = condition,
                    assay = assay,
                    method = "gene",
                    batch_size = batch_size,
                    mode = "pair",
                    only_loader = True,
                    left_cell = left_cells_train,
                    right_cell = right_cells_train,
                    pair_label = all_labels_train,
                    num_workers = 16,
                    reveal_name = False
                    )

        #for validation
        # Getting the dataloader
        adata_val.var["gene_name"] = adata_val.var.index

        # Getting the matrix of the cell neighbors
        # Getting the asymmetry matrix
        sparse_adj, cell_ids = get_adj(sample_dataset = None, anndata = adata_val, radius = radius, plot = False, sym = True)
        ### Getting the cell pairs according to the distance
        Pairs = GetPairs(sparse_adj, num_workers = 8) #assign the 1:1 negative to the positive
        all_pairs_val = Pairs.all_pairs
        all_labels_val = Pairs.all_labels
        print("all pairs val:", all_pairs_val.shape)
        # test_dataloader = data_prepare(sample_name, kfold, num_workers, batch_size, radius=r, test_size = None, zero_shot_cell_size = zero_shot_cell_size, split_mode = "random")
        left_cells_val = adata_val.obs.index[all_pairs_val[:,0]]
        right_cells_val = adata_val.obs.index[all_pairs_val[:,1]]
        val_dataloader = embed_data(adata_val,
                    tissue = tissue, 
                    condition = condition,
                    assay = assay,
                    method = "gene",
                    batch_size = batch_size,
                    mode = "pair",
                    only_loader = True,
                    left_cell = left_cells_val,
                    right_cell = right_cells_val,
                    pair_label = all_labels_val,
                    num_workers = 16,
                    reveal_name = False
                    )
        # Run the model to get the predictions
        
        Finetune = FineTune(config = config, 
                            sample_name = sample_name, 
                            radius = radius, 
                            fine_tune_mode = fine_tune_mode, 
                            wandb = True, 
                            strategy = "ddp",
                            rank = rank,
                            lora_alpha = lora_alpha)
        results = Finetune.train(None, train_dataloader, val_dataloader)
    wandb.finish()
    return results
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tuning parameters.')
    parser.add_argument('--fine_tune_mode', type=str, default='zero_shot',
                        help='The mode for fine-tuning the model. Default is "zero_shot". Optional "lora /"probe"/“zero_shot”/full_tune/')
    parser.add_argument('--rank', type=int, default=8,
                        help="The rank parameter of the lora, if using lora, please provide valid rank, default as 8")          
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help="The lora alpha parameter of the lora, if using lora, please provide valid alpha, default as 16. This indicate how much contribution of lora should be in the final weighting.")             
    parser.add_argument('--radius', type=int, nargs='+', default=[30],
                    help='A list of radius values between query and key cells')
    parser.add_argument('--cell_by_gene_path', type=str, default="/scratch/project_465001820/Spatialformer/data/Xenium_Breast/outs/cell_feature_matrix.h5",
                        help='The output path of the h5 file from Xenium')
    parser.add_argument('--cell_meta_path', type=str, default="/scratch/project_465001820/Spatialformer/data/Xenium_Breast/outs/cells.parquet",
                        help='The .parquet file path of the cell metadata from Xenium')
    parser.add_argument('--zero_shot_cell_size', type=int, default=500,
                        help='The number of centers to select for zero-shot fine-tuning. Default is 500.')
    parser.add_argument('--sample_name', type=str, default='breast_cancer',
                        help='The name of the sample. Default is "breast_cancer".')
    parser.add_argument('--checkpoint', type=str, default="/scratch/project_465001820/Spatialformer/output/checkpoints/step=0096000-train_total_loss=-2.9351-val_total_loss=0.0000.ckpt",
                        help='The path to the model checkpoint')
    parser.add_argument('--tissue', type=str, default="Breast",
                        help='The tissue type of the input')
    parser.add_argument('--condition', type=str, default="Disease",
                        help='The condition of the input samples')
    parser.add_argument('--assay', type=str, default="Xenium",
                        help='The assay of the input samples')
    parser.add_argument('--max_cells', type=int, default="100000",
                        help='The max number of cells to use. This is different from the zero-shot cell size. It is used to limit the number of cells in the dataset. Default is 100000.')
    parser.add_argument('--config_path', type=str, default=None,
                        help="The configuration path of the model")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="The batch size of the input samples")

    
    
    args = parser.parse_args()
    
    cell_by_gene_path = args.cell_by_gene_path
    fine_tune_mode = args.fine_tune_mode
    radius = args.radius
    zero_shot_cell_size = args.zero_shot_cell_size
    sample_name = args.sample_name
    cell_meta_path = args.cell_meta_path 
    max_cells = args.max_cells
    batch_size = args.batch_size
    rank = args.rank
    lora_alpha = args.lora_alpha
    #running the fine-tuning script
    
    all_results = {}
    #ckp from small panel
    # model_ckp_path = "/scratch/project_465001820/Spatialformer/output/checkpoints/step=0096000-train_total_loss=-2.9351-val_total_loss=0.0000.ckpt"
    #ckp from +5k panel
    # model_ckp_path = "/scratch/project_465001820/Spatialformer/output/checkpoints/stepstep=0176000-traintrain_total_loss=-2.8414-valval_total_loss=0.0000.ckpt"

    for r in radius:
        print(f"Running fine-tuning for radius: {r}")
        results = main(cell_by_gene_path = cell_by_gene_path,
            cell_meta_path = cell_meta_path,
            radius = r,
            zero_shot_cell_size = zero_shot_cell_size,
            sample_name = sample_name,
            fine_tune_mode = fine_tune_mode,
            config_path = args.config_path,
            output_dir = args.output_dir,
            tissue = args.tissue,
            condition = args.condition,
            max_cells = max_cells,
            batch_size = batch_size,
            rank = rank,
            lora_alpha = lora_alpha)
    
        #tesing the model
        all_results[r] = results[0]
    
    # os.killpg(os.getpid(), signal.SIGTERM)  # Kill all child processes

    #Save all the results
    # Convert the results to a DataFrame
    data = []
    for r, metrics in all_results.items():
        metrics['radius'] = r  # Optional: If you want to keep track of which model produced the results
        data.append(metrics)  # Append to the data list
    # Create the DataFrame
    df_results = pd.DataFrame(data)
    mean_values = df_results.groupby('radius').mean()
    print(df_results)
    print(mean_values)

    # Save to CSV file
    mean_values.to_csv(f'/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/results/{sample_name}_{fine_tune_mode}_mean_values_per_radius_{str(max_cells)}.csv', index=True)  # Include index if you want to keep fold numbers
    df_results.to_csv(f'/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/results/{sample_name}_{fine_tune_mode}_all_values_{str(max_cells)}.csv', index=True)

         
#default scripts
#for the Xenium breast cancer dataset, we can run the script with the following command:
# python cell_cell_communication_zero_shot_multi_platform.py --radius 10 20 30 50 80 100 120

#for Xenium colon dataset, we can run the script with the following command:
# python cell_cell_communication_zero_shot_multi_platform.py --radius 10 20 30 50 80 100 120 --fine_tune_mode zero_shot --cell_by_gene_path /scratch/project_465001820/Spatialformer/data/Xenium_CRC/cell_feature_matrix.h5 --cell_meta_path /scratch/project_465001820/Spatialformer/data/Xenium_CRC/cells.parquet --sample_name Xenium_CRC --zero_shot_cell_size 500 --tissue Colon --condition Disease


#For Xenium
#Lora fine-tuning
# python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode lora --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_metadata.csv --sample_name MERFISH_Lung --zero_shot_cell_size 500 --tissue Lung --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 10000

# python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode lora --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_metadata.csv --sample_name MERFISH_Lung --zero_shot_cell_size 500 --tissue Lung --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 100 --zero_shot_cell_size 50



#for MERFISH lung dataset, we can run the script with the following command:
#For MERFISH, fine-tune by rola is highly recommended
# python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode zero_shot --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_metadata.csv --sample_name MERFISH_Lung --zero_shot_cell_size 500 --tissue Lung --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json

#for MERFISH colon dataset, we can run the script with the following command:
# python cell_cell_communication_zero_shot_multi_platform.py --radius 10 20 30 50 80 100 120 --fine_tune_mode zero_shot --cell_by_gene_path /scratch/project_465001820/Spatialformer/data/MERFISH_Colon/HumanColonCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer/data/MERFISH_Colon/HumanColonCancerPatient1_cell_metadata.csv --sample_name MERFISH_Colon --zero_shot_cell_size 500 --tissue Colon --condition Disease

#for MERFISH breast dataset, we can run the script with the following command:
# python cell_cell_communication_zero_shot_multi_platform.py --radius 10 20 30 50 80 100 120 --fine_tune_mode zero_shot --cell_by_gene_path /scratch/project_465001820/Spatialformer/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_metadata.csv --sample_name MERFISH_Breast --zero_shot_cell_size 500 --tissue Breast --condition Disease


#fine tune mode -> test
#few-shot fine-tuning test (10k cells)
# python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode zero_shot --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_metadata.csv --sample_name MERFISH_Lung --tissue Lung --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 10000
# python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode zero_shot --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_metadata.csv --sample_name MERFISH_Breast --tissue Breast --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 10000
# python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode zero_shot --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_metadata.csv --sample_name MERFISH_Colon --tissue Colon --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 10000

#full fine-tuning test (100k cells)
# python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode zero_shot --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_metadata.csv --sample_name MERFISH_Lung --tissue Lung --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 100000
# python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode zero_shot --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_metadata.csv --sample_name MERFISH_Breast --tissue Breast --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 100000
# python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode zero_shot --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_metadata.csv --sample_name MERFISH_Colon --tissue Colon --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 100000


#For Full fine-tune experiments
#For Lung
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode full_tune --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_metadata.csv --sample_name MERFISH_Lung --zero_shot_cell_size 500 --tissue Lung --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --batch_size 32
#For Breast
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode full_tune --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_metadata.csv --sample_name MERFISH_Breast --zero_shot_cell_size 500 --tissue Breast --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --batch_size 32
#For Column
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode full_tune --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_metadata.csv --sample_name MERFISH_Colon --zero_shot_cell_size 500 --tissue Colon --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --batch_size 32

#For LoRA experiments(100k)
#For Lung
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode lora --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_metadata.csv --sample_name MERFISH_Lung --zero_shot_cell_size 500 --tissue Lung --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --batch_size 32
#For Breast
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode lora --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_metadata.csv --sample_name MERFISH_Breast --zero_shot_cell_size 500 --tissue Breast --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --batch_size 32
#For Column
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode lora --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_metadata.csv --sample_name MERFISH_Colon --zero_shot_cell_size 500 --tissue Colon --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --batch_size 32


#For LoRA experiments(10k)
#For Lung
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode lora --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_metadata.csv --sample_name MERFISH_Lung --zero_shot_cell_size 500 --tissue Lung --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --batch_size 32 --max_cells 10000
#For Breast
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode lora --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_metadata.csv --sample_name MERFISH_Breast --zero_shot_cell_size 500 --tissue Breast --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --batch_size 32 --max_cells 10000
#For Colon
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode lora --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_metadata.csv --sample_name MERFISH_Colon --zero_shot_cell_size 500 --tissue Colon --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --batch_size 32 --max_cells 10000


#full fine-tuning test (100k cells)
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode zero_shot --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Lung/HumanLungCancerPatient1_cell_metadata.csv --sample_name MERFISH_Lung --tissue Lung --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 100000
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode zero_shot --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Breast/HumanBreastCancerPatient1_cell_metadata.csv --sample_name MERFISH_Breast --tissue Breast --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 100000
# srun /scratch/project_465001820/miniconda3/envs/spatialformer_flash_attn/bin/python cell_cell_communication_zero_shot_multi_platform.py --radius 30 --fine_tune_mode zero_shot --rank 64 --lora_alpha 128 --cell_by_gene_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_by_gene.csv --cell_meta_path /scratch/project_465001820/Spatialformer_main_practice/data/MERFISH_Colon/HumanColonCancerPatient1_cell_metadata.csv --sample_name MERFISH_Colon --tissue Colon --condition Disease --config_path /scratch/project_465001820/Spatialformer/spatialformer/config/_config_fine_tune_probe.json --max_cells 100000
