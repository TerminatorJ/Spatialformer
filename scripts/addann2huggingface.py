from datasets import DatasetDict, load_dataset, concatenate_datasets, Dataset, load_from_disk
import json
import os

hf_cache = "/home/sxr280/Spatialformer/cache"
root_dir = "/scratch/project_465001027/spatialformer/david_data"
# import pdb; pdb.set_trace()
with open("/home/sxr280/Spatialformer/david_data/allsample_nich_annotation.json", 'r') as file:
    all_ann_dict = json.load(file) 
# import pdb; pdb.set_trace()
# Map annotations to the DataFrame, creating a new column 'Niche_Annotations'

dataset = load_from_disk(os.path.join(hf_cache,"xenium_25_lung_dataset_update3"))
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()
val_df = dataset['validation'].to_pandas()
#drop unnecessary columns
train_df = train_df.drop(["Distance_Matrix", "__index_level_0__"], axis=1)
test_df = test_df.drop(["Distance_Matrix", "__index_level_0__"], axis=1)
val_df = val_df.drop(["Distance_Matrix", "__index_level_0__"], axis=1)
# import pdb; pdb.set_trace()
train_df['compound_key'] = train_df['Sample_Names'] + '_' + train_df['Cell_id']
test_df['compound_key'] = test_df['Sample_Names'] + '_' + test_df['Cell_id']
val_df['compound_key'] = val_df['Sample_Names'] + '_' + val_df['Cell_id']

# import pdb; pdb.set_trace()
train_df['Niche_Annotations'] = train_df['compound_key'].map(all_ann_dict)
test_df['Niche_Annotations'] = test_df['compound_key'].map(all_ann_dict)
val_df['Niche_Annotations'] = val_df['compound_key'].map(all_ann_dict)
print("After filtering")
# import pdb; pdb.set_trace()
# Assuming train_df, test_df, and val_df are your filtered DataFrames
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_dataset = Dataset.from_pandas(val_df)

# Optional: Create a DatasetDict if you want to package these splits
new_dataset_dict = DatasetDict({'train': train_dataset,'test': test_dataset,'validation': val_dataset})

new_dataset_dict.save_to_disk("/home/sxr280/Spatialformer/cache/xenium_25_lung_dataset_update4")
new_dataset_dict.push_to_hub("TerminatorJ/xenium_25_lung_dataset_update4")

