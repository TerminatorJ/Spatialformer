from train import *
import os
import pytorch_lightning as pl
import torch
import json
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import pandas as pd
from datasets import DatasetDict, load_dataset, concatenate_datasets, Dataset
from utils import uniform_quantile_global, binning
from datetime import datetime
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_model(config_path, ckp_path, bpp, ag_loss, nlayers, n_tasks, spatial_embedding):
    #loading the model
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    config["bpp"] = bpp
    config['ag_loss'] = ag_loss
    config['nlayers'] = nlayers
    config['n_tasks'] = n_tasks
    config['spatial_embedding'] = spatial_embedding

    Trainer = MyTrainer(config = config)
    model = Trainer.plmodel
    ckp = torch.load(ckp_path)
    params = ckp["state_dict"]
    model.load_state_dict(params)
    #set to eval mode
    model.eval()
    #transfer to cuda
    model.to(device)
    return config, model
def norm_mean(hidden_repr, attention_mask):
    # import pdb; pdb.set_trace()
    attention_mask_sq = attention_mask.unsqueeze(-1).cuda()  # Shape: [80, 400, 1]
    embed_norm = torch.sum(hidden_repr * attention_mask_sq, dim=1) / torch.sum(attention_mask_sq, dim=1)
    return embed_norm

def get_repr(dataloader, objective):
    embeds = []
    anns = []
    niche_anns = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            # import pdb; pdb.set_trace()
            if not objective == "baseline":
                attention_mask = batch["attention_mask"]
                annotation = batch["annotation"]
                niche_annotation = batch["niche_annotation"]
                #getting the last layer
                hidden_repr = model.get_embeddings(batch, [-1])[0]
                embed_norm = norm_mean(hidden_repr, attention_mask)
                embed_norm = embed_norm.detach().cpu().numpy()
                embeds.append(embed_norm)
                anns += annotation
                niche_anns += niche_annotation
            else:
                annotation = batch["annotation"]
                niche_annotation = batch["niche_annotation"]
                anns += annotation
                niche_anns += niche_annotation
                exp = batch["Expression"].detach().cpu().numpy()

                embeds.append(exp)


    embeddings = np.concatenate(embeds, axis=0)
    return embeddings, anns, niche_anns

def sample_dataset(my_dataset, fraction, label_type):
    #group by the cell types
    #Select 5% of the samples randomly
    my_df = my_dataset.to_pandas()
    if label_type == "cell_types":
        #filtering out the not available cell type
        my_df = my_df[my_df["Annotations"] != "not available"]
        cell_types = my_df['Annotations'].unique()
        print(f"There are {len(cell_types)} cell types exist: \n {cell_types}")
        df_sampled = my_df.groupby('Annotations').apply(lambda x: x.sample(frac=fraction)).reset_index(drop=True)
        sampled_types = df_sampled['Annotations'].unique()
        # import pdb; pdb.set_trace()
        print(f"The total cells left: {df_sampled.shape[0]}")
        print(f"There are {len(sampled_types)} cell types after filtering: \n {cell_types}")
        print(f"Their counts are:")
        print(df_sampled["Annotations"].value_counts())
        print(f"The number of samples left:")
        print(len(df_sampled["Sample_Names"].unique()), "\n", df_sampled["Sample_Names"].unique())
        #transfer the dataframe back to the dataset
        sampled_dataset = Dataset.from_pandas(df_sampled)
    elif label_type == "cell_niches":
        # import pdb; pdb.set_trace()
        #filtering out the not available cell type
        # my_df = my_df[my_df["Niche_Annotations"] != "NAN"]
        my_df = my_df.dropna(subset=['Niche_Annotations'])
        cell_niches = my_df['Niche_Annotations'].unique()
        print(f"There are {len(cell_niches)} cell niches exist: \n {cell_niches}")
        df_sampled = my_df.groupby('Niche_Annotations').apply(lambda x: x.sample(frac=fraction)).reset_index(drop=True)
        sampled_types = df_sampled['Niche_Annotations'].unique()
        print(f"The total cells left: {df_sampled.shape[0]}")
        print(f"There are {len(sampled_types)} cell types after filtering: \n {sampled_types}")
        print(f"Their counts are:")
        print(df_sampled["Niche_Annotations"].value_counts())
        print(f"The number of samples left:")
        print(len(df_sampled["Sample_Names"].unique()), "\n", df_sampled["Sample_Names"].unique())
        #transfer the dataframe back to the dataset
        sampled_dataset = Dataset.from_pandas(df_sampled)

    return sampled_dataset

def getexpcount(dataset, label_type):
    # Assume 'train_dataset' is a pandas DataFrame or a similar object like a list of dictionaries
    # First, extract the relevant columns
    expression_data = [i[0] for i in dataset['Expression']]  # This should be a list of expression values per cell
    gene_names = dataset[0]['Gene']  # This should be a list of corresponding gene names
    cell_ids = dataset["Cell_id"]
    df = pd.DataFrame(expression_data, index=cell_ids, columns=gene_names)
    if label_type == "cell_types":
        anns = dataset["Annotations"]
    elif label_type == "cell_niches":
        anns = dataset["Niche_Annotations"]
    return df, np.array(anns)





def workflow(my_dataset, fraction, batch_size, bpp, ag_loss, nlayers, n_tasks, objective, save_fra, label_type):
   
    
    if save_fra:
        print("saving the dataset to erda...")
        #save the dataset to the memory
        train_dataset = sample_dataset(my_dataset["train"], fraction, label_type)
        test_dataset = sample_dataset(my_dataset["test"], fraction, label_type)
        val_dataset = sample_dataset(my_dataset["validation"], fraction, label_type)

        train_dataset.save_to_disk(f"/tmp/erda/Spatialformer/xenium_25_lung_dataset_frac10_train")
        test_dataset.save_to_disk(f"/tmp/erda/Spatialformer/xenium_25_lung_dataset_frac10_test")
        val_dataset.save_to_disk(f"/tmp/erda/Spatialformer/xenium_25_lung_dataset_frac10_val")

    else:
        train_dataset = load_from_disk(f"/tmp/erda/Spatialformer/xenium_25_lung_dataset_frac10_train")
        test_dataset = load_from_disk(f"/tmp/erda/Spatialformer/xenium_25_lung_dataset_frac10_test")
        val_dataset = load_from_disk(f"/tmp/erda/Spatialformer/xenium_25_lung_dataset_frac10_val")
    

    #get the exp count dataframe
    # Creating a DataFrame with each cell's expression data
    print("Generating the expression matrix and annotations")
    train_X, train_ann = getexpcount(train_dataset, label_type)
    test_X, test_ann = getexpcount(test_dataset, label_type)
    val_X, val_ann = getexpcount(val_dataset, label_type)
    # import pdb; pdb.set_trace()
    
    if label_type == "cell_types":
        np.save(f"/home/sxr280/Spatialformer/data/train_frac10_ann.npy", train_ann)
        np.save(f"/home/sxr280/Spatialformer/data/test_frac10_ann.npy", test_ann)
        np.save(f"/home/sxr280/Spatialformer/data/val_frac10_ann.npy", val_ann)
        train_X.to_csv(f"/home/sxr280/Spatialformer/data/train_frac10.csv")
        test_X.to_csv(f"/home/sxr280/Spatialformer/data/test_frac10.csv")
        val_X.to_csv(f"/home/sxr280/Spatialformer/data/val_frac10.csv")
    elif label_type == "cell_niches":
        np.save(f"/home/sxr280/Spatialformer/data/train_frac10_nicheann.npy", train_ann)
        np.save(f"/home/sxr280/Spatialformer/data/test_frac10_nicheann.npy", test_ann)
        np.save(f"/home/sxr280/Spatialformer/data/val_frac10_nicheann.npy", val_ann)
        train_X.to_csv(f"/home/sxr280/Spatialformer/data/train_frac10nicheX.csv")
        test_X.to_csv(f"/home/sxr280/Spatialformer/data/test_frac10nicheX.csv")
        val_X.to_csv(f"/home/sxr280/Spatialformer/data/val_frac10nicheX.csv")



    #save the benchmark dataset into the memory


    #get dataloader
    train_dataloader = create_data_loaders(train_dataset, batch_size=batch_size, context_length=400, special_token_num = 0, directionality = True, split_num = 1)
    test_dataloader = create_data_loaders(test_dataset, batch_size=batch_size, context_length=400, special_token_num = 0, directionality = True, split_num = 1)
    val_dataloader = create_data_loaders(val_dataset, batch_size=batch_size, context_length=400, special_token_num = 0, directionality = True, split_num = 1)
    print("running train")
    train_embeddings, train_anns, train_niche_anns = get_repr(train_dataloader, objective)
    print("running test")
    test_embeddings, test_anns, test_niche_anns = get_repr(test_dataloader, objective)
    print("running val")
    val_embeddings, val_anns, val_niche_anns = get_repr(val_dataloader, objective)

    np.save(f"/home/sxr280/Spatialformer/data/train_embedding_{bpp}_{ag_loss}_{nlayers}_{fraction}_{n_tasks}_{objective}_{current_time}.npy", train_embeddings)
    np.save(f"/home/sxr280/Spatialformer/data/train_labels_{bpp}_{ag_loss}_{nlayers}_{fraction}_{n_tasks}_{objective}_{current_time}.npy", train_anns)
    np.save(f"/home/sxr280/Spatialformer/data/train_nichelabels_{bpp}_{ag_loss}_{nlayers}_{fraction}_{n_tasks}_{objective}_{current_time}.npy", train_niche_anns)



    np.save(f"/home/sxr280/Spatialformer/data/test_embedding_{bpp}_{ag_loss}_{nlayers}_{fraction}_{n_tasks}_{objective}_{current_time}.npy", test_embeddings)
    np.save(f"/home/sxr280/Spatialformer/data/test_labels_{bpp}_{ag_loss}_{nlayers}_{fraction}_{n_tasks}_{objective}_{current_time}.npy", test_anns)
    np.save(f"/home/sxr280/Spatialformer/data/test_nichelabels_{bpp}_{ag_loss}_{nlayers}_{fraction}_{n_tasks}_{objective}_{current_time}.npy", test_niche_anns)


    np.save(f"/home/sxr280/Spatialformer/data/val_embedding_{bpp}_{ag_loss}_{nlayers}_{fraction}_{n_tasks}_{objective}_{current_time}.npy", val_embeddings)
    np.save(f"/home/sxr280/Spatialformer/data/val_labels_{bpp}_{ag_loss}_{nlayers}_{fraction}_{n_tasks}_{objective}_{current_time}.npy", val_anns)
    np.save(f"/home/sxr280/Spatialformer/data/val_nichelabels_{bpp}_{ag_loss}_{nlayers}_{fraction}_{n_tasks}_{objective}_{current_time}.npy", val_niche_anns)








    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='getting the hidden representation of the model')
    parser.add_argument('--special_token_num', type = int, default=4, help='In the input data, whether to use the special auxiliary tokens. Set the number as 4 when doing the normal training, and set 0 when doing ablation. default 4')
    parser.add_argument('--dataset_path', type = str, help='The dataset directory for the online and inhouse path.')
    parser.add_argument('--ckp_path', type = str, help = 'The model checkpoint that is used to generate the embedding for testing')
    parser.add_argument('--bpp', action = 'store_true', help = 'Whther to use bpp matrix, which will introduce the bpp_weight in the model')
    parser.add_argument('--ag_loss', action = 'store_true',  help = 'Whther to use the guiding loss the drive the model learn the specific patterns')
    parser.add_argument('--nlayers', type = int, help = 'The number of layers for the model to use, which should match the exact model scale in the model checkpoints')
    parser.add_argument('--batch_size', type = int, default=15, help = 'The number of batch size that is used to the map function to get the representation')
    parser.add_argument('--fraction', type = float, default=0.01, help = 'The fraction of cells that can be used to test the model performance and getting the embeddings out')
    parser.add_argument('--n_tasks', type = int, default=2, help = 'The number of tasks for training the model')
    parser.add_argument('--objective', type = str, default=None, help = 'The training objective')
    parser.add_argument('--save_fra', action = 'store_true', help = 'Whether to save the split dataset')
    parser.add_argument('--spatial_embedding', action = 'store_true', help = 'Whether to use the spatial embedding')
    parser.add_argument('--label_type', type = str, default="cell_types", help = 'The label of the cells, which can be cell niches and cell types')
    # parser.add_argument('--mini_batch', action = 'store_true', help = 'whether use minibatch')

    args = parser.parse_args()
    #loading the model
    global model
    hf_cache = "/home/sxr280/Spatialformer/cache"
    config, model = load_model("/home/sxr280/Spatialformer/config/_config_train.json", args.ckp_path, args.bpp, args.ag_loss, args.nlayers, args.n_tasks, args.spatial_embedding)
    #loading the dataset
    my_dataset = load_from_disk(os.path.join(hf_cache, args.dataset_path))
    workflow(my_dataset, args.fraction, args.batch_size, args.bpp, args.ag_loss, args.nlayers, args.n_tasks, args.objective, args.save_fra, args.label_type)


#To get the 2 tasks embeddings 
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0009000-train_total_loss=0.3292-val_total_loss=0.3359.ckpt \
# --bpp  --nlayers 8 --batch_size 30 --fraction 0.1 --n_tasks 2 
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0004000-train_total_loss=5.0878-val_total_loss=5.0833.ckpt \
# --bpp  --nlayers 8 --batch_size 32 --fraction 0.1 --n_tasks 2
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0016000-train_total_loss=2.7622-val_total_loss=2.6395.ckpt \
# --bpp  --nlayers 8 --batch_size 32 --fraction 0.1 --n_tasks 2
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0020000-train_total_loss=2.5735-val_total_loss=2.5471.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 2
#predict cell types without tunableembeddings
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update4 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0020000-train_total_loss=2.5735-val_total_loss=2.5471.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 2 --label_type cell_niches 

#To get the 2 tasks with runable spatial embeddings
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0010000-train_total_loss=3.8043-val_total_loss=3.7695.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 2 --spatial_embedding

# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0028000-train_total_loss=3.1092-val_total_loss=2.9783.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 2 --spatial_embedding

#  python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update4 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0032000-train_total_loss=2.9954-val_total_loss=2.8468.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 2 --spatial_embedding --label_type cell_niches 


# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update4 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0032000-train_total_loss=2.9954-val_total_loss=2.8468.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 2 --spatial_embedding --save_fra

#for linear schedule trained with spatial embeddings
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update4 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0022000-train_total_loss=0.7509-val_total_loss=0.7559.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 2 --spatial_embedding
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update4 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0028000-train_total_loss=0.3467-val_total_loss=0.3072.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 2 --spatial_embedding

#getting the cell nich embeddings with the spatial embeddings
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update4 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0022000-train_total_loss=0.7509-val_total_loss=0.7559.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 2 --spatial_embedding --label_type cell_niches 
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update4 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0028000-train_total_loss=0.3467-val_total_loss=0.3072.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 2 --spatial_embedding --label_type cell_niches 





#single task spatial
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0009000-train_total_loss=0.4713-val_total_loss=0.4776.ckpt \
# --bpp  --nlayers 8 --batch_size 32 --fraction 0.1 --n_tasks 1 --objective spatial


# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0016000-train_total_loss=0.4538-val_total_loss=0.4657.ckpt \
# --bpp  --nlayers 8 --batch_size 32 --fraction 0.1 --n_tasks 1 --objective spatial









#single task exp
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0009000-train_total_loss=0.0053-val_total_loss=0.0046.ckpt \
# --bpp  --nlayers 8 --batch_size 30 --fraction 0.1 --n_tasks 1 --objective exp

# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0016000-train_total_loss=0.0025-val_total_loss=0.0023.ckpt \
# --bpp  --nlayers 8 --batch_size 16 --fraction 0.1 --n_tasks 1 --objective exp


#single task MT
# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0002000-train_total_loss=5.3131-val_total_loss=5.2911.ckpt \
# --bpp  --nlayers 8 --batch_size 30 --fraction 0.1 --n_tasks 1 --objective mt

# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0008000-train_total_loss=4.9323-val_total_loss=4.7789.ckpt \
# --bpp  --nlayers 8 --batch_size 30 --fraction 0.1 --n_tasks 1 --objective mt

# python get_embeddings.py --special_token_num 0 --dataset_path xenium_25_lung_dataset_update3 --ckp_path /home/sxr280/Spatialformer/output/checkpoints/step=0018000-train_total_loss=4.5273-val_total_loss=4.4960.ckpt \
# --bpp  --nlayers 8 --batch_size 30 --fraction 0.1 --n_tasks 1 --objective mt

# #single task baseline
# python get_embeddings.py --special_token_num 4 --dataset_path xenium_25_lung_dataset_update3 --ckp_path  /home/sxr280/Spatialformer/output/checkpoints/step=0016000-train_total_loss=0.0025-val_total_loss=0.0023.ckpt \
# --bpp  --nlayers 8 --batch_size 30 --fraction 0.1 --n_tasks 1 --objective baseline


#save the dataset
# python get_embeddings.py --special_token_num 4 --dataset_path xenium_25_lung_dataset_update3 --ckp_path  /home/sxr280/Spatialformer/output/checkpoints/step=0016000-train_total_loss=0.0025-val_total_loss=0.0023.ckpt \
# --bpp  --nlayers 8 --batch_size 30 --fraction 0.1 --n_tasks 1 --objective baseline
