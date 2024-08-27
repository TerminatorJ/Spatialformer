from train import *
import os
import pytorch_lightning as pl
import torch
import json
import numpy as np
from tqdm import tqdm
import pickle
import argparse
from datasets import DatasetDict, load_dataset, concatenate_datasets, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataCollator(object):
    def __init__(self, context_length=400, padding_idx=0, special_token_num=4):
        self.context_length = context_length
        self.padding_idx = padding_idx
        self.special_token_num = special_token_num
        # self.selection = selection

    def __call__(self, batch):
            # Extract sequences and matrices

            gg_mtx = [torch.tensor(item['Gene_Gene_Matrix']) for item in batch]
            Full_Tokens = [torch.tensor(item['Full_Tokens']) for item in batch]
            Norm_Exp = [torch.tensor(item['Normalized_Exp']) for item in batch]
            
            # import pdb; pdb.set_trace()
            full_tokens = torch.full((len(Full_Tokens), self.context_length), self.padding_idx, dtype=torch.int)
            for i, v in enumerate(Full_Tokens):
                full_tokens[i,:v.size(0)-(4-self.special_token_num)] = v[4-self.special_token_num:]
            



            norm_exp = torch.full((len(Norm_Exp), self.context_length), self.padding_idx, dtype=torch.float)
            for i, e in enumerate(Norm_Exp):
                norm_exp[i,self.special_token_num:self.special_token_num+e.size(0)] = e
            # import pdb; pdb.set_trace()
            # Pad sequences
            attention_masks = (full_tokens != self.padding_idx).bool()
            # import pdb; pdb.set_trace()

            # Pad 2D matrices
            gg_mtx_p = torch.full((len(gg_mtx), self.context_length, self.context_length), self.padding_idx, dtype=torch.float)
            for i, mat in enumerate(gg_mtx):
                current_size = mat.shape[0]
                gg_mtx_p[i, self.special_token_num:(current_size+self.special_token_num), self.special_token_num:(self.special_token_num+current_size)] = mat
            # import pdb; pdb.set_trace()
            return {
                'adjmtx': gg_mtx_p,
                'indices': full_tokens,
                'attention_mask': attention_masks,
                'normalized_exp': norm_exp
            }


def load_model(config_path, bpp, ag_loss, nlayers):
    #loading the model
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    config["bpp"] = bpp
    config['ag_loss'] = ag_loss
    config['nlayers'] = nlayers
    Trainer = MyTrainer(config = config)
    model = Trainer.plmodel
    return config, model
def norm_mean(hidden_repr, attention_mask):
    # import pdb; pdb.set_trace()
    attention_mask_sq = attention_mask.unsqueeze(-1).cuda()  # Shape: [80, 400, 1]
    embed_norm = torch.sum(hidden_repr * attention_mask_sq, dim=1) / torch.sum(attention_mask_sq, dim=1)
    return embed_norm
def get_repr(example):
    mem_before = torch.cuda.memory_allocated()
    import pdb; pdb.set_trace()
    with torch.no_grad():
        data_collator = CustomDataCollator(special_token_num = 0)
        batch = data_collator(example)
        attention_mask = batch["attention_mask"]

        # import pdb; pdb.set_trace()
        #getting the last layer
        hidden_repr = model.get_embeddings(batch, [-1])[0]
        embed_norm = norm_mean(hidden_repr, attention_mask)
        import pdb; pdb.set_trace()
        # embed_norm = torch.mean(hidden_repr, dim = 1)
    mem_after = torch.cuda.memory_allocated()
    print(f"GPU memory usage: {mem_before/1e9} GB -> {mem_after/1e9} GB")  
    torch.cuda.empty_cache()
    return {"Embeddings_Norm": embed_norm}
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='getting the hidden representation of the model')
    parser.add_argument('--special_token_num', type = "int", default=4, help='In the input data, whether to use the special auxiliary tokens. Set the number as 4 when doing the normal training, and set 0 when doing ablation. default 4')
    parser.add_argument('--dataset_path', type = 'str', help='The dataset directory for the online and inhouse path.')
    parser.add_argument('--ckp_path', type = "str", help = 'The model checkpoint that is used to generate the embedding for testing')
    parser.add_argument('--bpp', action = 'true', help = 'Whther to use bpp matrix, which will introduce the bpp_weight in the model')
    parser.add_argument('--ag_loss', action = 'true',  help = 'Whther to use the guiding loss the drive the model learn the specific patterns')
    parser.add_argument('--nlayers', type = 'str', help = 'The number of layers for the model to use, which should match the exact model scale in the model checkpoints')
    parser.add_argument('--batch_size', type = 'int', help = 'The number of batch size that is used to the map function to get the representation')
    args = parser.parse_args()
    #loading the model
    global model
    hf_cache = "/home/sxr280/Spatialformer/cache"

    config, model = load_model("/home/sxr280/Spatialformer/config/_config_train.json", args.bpp, args.ag_loss, args.nlayers)
    #loading the dataset
    my_dataset = load_dataset(args.dataset_path, cache_dir = hf_cache, num_proc = 1)
    #split the dataset according to the cell type
    #this run slowly
    combined_dataset = concatenate_datasets([my_dataset["train"], my_dataset["test"]])
    #group by the cell types
    #Select 5% of the samples randomly
    combined_df = combined_dataset.to_pandas()
    
    #filtering out the not available cell type
    combined_df = combined_df[combined_df["Annotations"] != "not available"]
    cell_types = combined_df['Annotations'].unique()
    print(f"There are {len(cell_types)} cell types exist: \n {cell_types}")
    df_sampled = combined_df.groupby('Annotations').apply(lambda x: x.sample(frac=0.10)).reset_index(drop=True)
    sampled_types = df_sampled['Annotations'].unique()
    print(f"There are {len(sampled_types)} cell types after filtering: \n {cell_types}")
    print(f"Their counts are:")
    print(df_sampled["Annotations"].value_counts())
    print(f"The number of samples left:")
    print(len(df_sampled["Sample_Names"].unique()), "\n", df_sampled["Sample_Names"].unique())
    #transfer the dataframe back to the dataset
    sampled_dataset = Dataset.from_pandas(df_sampled)

    #loading the parameters
    ckp = torch.load(args.ckp_paths)
    params = ckp["state_dict"]
    model.load_state_dict(params)
    #set to eval mode
    model.eval()
    #transfer to cuda
    model.to(device)
    # import pdb; pdb.set_trace()

    tokenized_datasets = sampled_dataset.map(get_repr, batched = True, batch_size = 15)


    tokenized_datasets.save_to_disk(f"/home/sxr280/Spatialformer/data/embedding_visualization_arrow_{args.bpp}_{args.ag_loss}_{args.nlayers}")


#for the embeddings without bias                
#python test.py 
#for the embeddings with bias                
#python test.py --bias True
#set the power 50 to do the simulation, without bias
#python test.py --simulation --power 50
#setting the power as 1 to test the score effects
#python test.py --simulation --power 1



#python test.py --special_token_num 0 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --ckp_path {}
