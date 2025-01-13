
import os
from datasets import DatasetDict, load_dataset, concatenate_datasets, Dataset, load_from_disk
import json
hf_cache = "/home/sxr280/Spatialformer/cache"
root = "/tmp/erda/Spatialformer/downloaded_data/processed/"
concat_name = "xenium_pan_dataset"


def single_process(rank_gene_names, full_tokens):
    # import pdb; pdb.set_trace()
    new_full_tokens = []
    for i,rank_gene_name in enumerate(rank_gene_names):
        full_token = full_tokens[i]
        meta_infos = [meta_dict[meta_token] for meta_token in full_token[:4]]
        full_token = [new_tokens[name] for name in rank_gene_name]
        new_meta_tokens = [new_tokens[meta_info] for meta_info in meta_infos]
        # import pdb; pdb.set_trace()
        new_full_tokens.append(new_meta_tokens + full_token)
    # import pdb; pdb.set_trace()
    return new_full_tokens
def pansingle_process(rank_gene_names, full_tokens):
    # import pdb; pdb.set_trace()
    new_full_tokens = []
    for i,rank_gene_name in enumerate(rank_gene_names):
        full_token = full_tokens[i]
        # import pdb; pdb.set_trace()
        meta_infos = [panmeta_dict[meta_token.item()] for meta_token in full_token[:4]]
        full_token = [new_tokens[name] for name in rank_gene_name]
        new_meta_tokens = [new_tokens[meta_info] for meta_info in meta_infos]
        # import pdb; pdb.set_trace()
        new_full_tokens.append(new_meta_tokens + full_token)
    # import pdb; pdb.set_trace()
    return new_full_tokens
def multi_process(examples):
    # import pdb; pdb.set_trace()
    out = {"Expression":[], "Split":[], "Cell_Ids": [], "Gene": [], "Ranked_Gene_Names":[], "Full_Tokens":[], "Gene_Gene_Matrix":[]}

    # Process each example
    for key, values in examples.items():
        if key == "Full_Tokens":
            full_tokens = single_process(examples["Ranked_Gene_Names"], examples["Full_Tokens"])
            # import pdb; pdb.set_trace()
            out[key].extend(full_tokens)
        elif key in out.keys():
            out[key].extend(values)
        else:
            pass
    
    # import pdb; pdb.set_trace()
    return out

def panmulti_process(examples):
    # import pdb; pdb.set_trace()
    out = {"Expression":[], "Split":[], "Cell_Ids": [], "Gene": [], "Ranked_Gene_Names":[], "Full_Tokens":[], "Gene_Gene_Matrix":[]}

    # Process each example
    for key, values in examples.items():
        if key == "Full_Tokens":
            full_tokens = pansingle_process(examples["Ranked_Gene_Names"], examples["Full_Tokens"])
            # import pdb; pdb.set_trace()
            out[key].extend(full_tokens)
        elif key in out.keys():
            out[key].extend(values)
        else:
            pass
    
    # import pdb; pdb.set_trace()
    return out



#loading the old tokens
# with open("/home/sxr280/Spatialformer/tokenizer/token.json", "rb") as f1:
#     old_tokens = json.load(f1)
# switched_tokens = {value: key for key, value in old_tokens.items()}
#loading the new tokens
global new_tokens
global meta_dict
global panmeta_dict
with open("/home/sxr280/Spatialformer/tokenizer/tokenv3.json", "rb") as f1:
    new_tokens = json.load(f1)
with open("/home/sxr280/Spatialformer/tokenizer/token.json", "rb") as f2:
    meta_dict = json.load(f2)
with open("/home/sxr280/Spatialformer/tokenizer/tokenv2.json", "rb") as f3:
    panmeta_dict = json.load(f3)
meta_dict = {v:k for k,v in meta_dict.items()}
panmeta_dict = {v:k for k,v in panmeta_dict.items()}


out_path = os.path.join(root, concat_name)
#loading David dataset
david_dataset = load_from_disk(os.path.join(hf_cache,"xenium_25_lung_dataset_update4"))


#loading the pandataset
pan_dataset = load_from_disk(out_path)
new_pan_dataset = pan_dataset.map(
    panmulti_process,
    batched=True,
    num_proc=32,
    batch_size=1000)
# import pdb; pdb.set_trace()

# Apply the function using map
new_david_dataset = david_dataset.map(
    multi_process,
    batched=True,
    num_proc=32,
    batch_size=1000)
# import pdb; pdb.set_trace()
#concate them

combined_dataset = DatasetDict({
            'train': concatenate_datasets([new_david_dataset["train"], new_pan_dataset["train"]]),
            'test': concatenate_datasets([new_david_dataset["test"], new_pan_dataset["test"]]),
            'validation': concatenate_datasets([new_david_dataset["validation"], new_pan_dataset["validation"]])
            })

combined_dataset.save_to_disk("/tmp/erda/Spatialformer/downloaded_data/processed/xenium_pandavid_dataset")
combined_dataset.push_to_hub("xenium_pandavid_dataset")