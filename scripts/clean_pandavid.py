from datasets import load_from_disk, load_dataset
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# hf_cache = "/scratch/project_465001027/spatialformer/cache"
# # david_dataset = load_dataset("TerminatorJ/xenium_25_lung_dataset_update4",cache_dir = hf_cache) 
# pandavid_dataset = load_dataset("TerminatorJ/xenium_pandavid_dataset",cache_dir = hf_cache) 

# #after filtering
# def is_non_empty(example):
#     """Check if 'Ranked_Gene_Names' is non-empty."""
#     if example['Ranked_Gene_Names'] is not None and len(example['Ranked_Gene_Names']) > 0 and len(example['Ranked_Gene_Names']) > 10:
        
#         return True
#     else:
#         print(example['Cell_id'])
#         return False


# # Filter the dataset
# cleaned_pandavid_dataset = pandavid_dataset.filter(is_non_empty,num_proc=32)

# # Optionally, print out the size of the new dataset
# print("Number of examples in cleaned dataset:", len(cleaned_pandavid_dataset))

# # cleaned_pandavid_dataset.save_to_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset2")
# # combined_dataset.push_to_hub(concat_name)

# cleaned_pandavid_dataset.save_to_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset2")
# cleaned_pandavid_dataset.push_to_hub("xenium_pandavid_dataset2")



#round two of filtering
# hf_cache = "/scratch/project_465001027/spatialformer/cache"
# david_dataset = load_dataset("TerminatorJ/xenium_25_lung_dataset_update4",cache_dir = hf_cache) 
pandavid_dataset = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset3")
# pandavid_dataset = load_dataset("TerminatorJ/xenium_pandavid_dataset2",cache_dir = hf_cache) 
def filterfunc(example):
    """Check if 'Ranked_Gene_Names' is non-empty."""
    if example['Ranked_Gene_Names'] is not None and len(example['Ranked_Gene_Names']) > 10 and np.array(example["Gene_Gene_Matrix"]).sum() > 0:
        
        return True
    else:
        # print(example['Cell_id'])
        return False
    


cleaned_pandavid_dataset = pandavid_dataset.filter(filterfunc,num_proc=48)
cleaned_pandavid_dataset.save_to_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset3")
cleaned_pandavid_dataset.push_to_hub("xenium_pandavid_dataset3")
import pdb; pdb.set_trace()