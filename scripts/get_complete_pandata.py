from datasets import load_from_disk
import os
from datasets import DatasetDict, load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm
# def add_sample_name(example):
#     # You can define a naming strategy here; for example, using an index or some other logic
#     example['Sample_Names'] = name  # Replace with your desired logic
#     return example

def ds_alignment(ds, name):
    

    # Determine the size of the dataset
    num_rows = ds.shape[0]

    # Create a new column 'triple' filled with the value "triple"
    column = [name] * num_rows
    # import pdb; pdb.set_trace()
    # Add the new column to the dataset
    dataset = ds.add_column("Sample_Names", column)

    return dataset

def counter(ds):
    c = 0
    for split in ["train","test","validation"]:
        c += ds[split].shape[0]
    return c


def split_ds(file):

    name = file.split("/")[-1].split("_arrow")[0]
    sta_datasets = load_from_disk(file)
    count = counter(sta_datasets)
    # import pdb; pdb.set_trace()
    train_dataset = ds_alignment(sta_datasets["train"], name)
    test_dataset = ds_alignment(sta_datasets["test"], name)
    val_dataset = ds_alignment(sta_datasets["validation"], name)
    print(f"the number of cells in {file} is {str(count)}")
    return train_dataset, test_dataset, val_dataset
def concat_ds(files):
    train_list = []
    test_list = []
    val_list = []
    for file in tqdm(files):
        # import pdb; pdb.set_trace()
        train_dataset, test_dataset, val_dataset = split_ds(file)
        # import pdb; pdb.set_trace()
        train_list.append(train_dataset)
        # import pdb; pdb.set_trace()
        test_list.append(test_dataset)
        val_list.append(val_dataset)
        # import pdb; pdb.set_trace()
    print("concating the dataset")
    combined_dataset = DatasetDict({
            'train': concatenate_datasets(train_list),
            'test': concatenate_datasets(test_list),
            'validation': concatenate_datasets(val_list)
            })
    return combined_dataset



root = "/tmp/erda/Spatialformer/downloaded_data/processed/"
concat_name = "xenium_pan_dataset"
out_path = os.path.join(root, concat_name)
arrow_files = [os.path.join(root, file) for file in os.listdir(root) if "outs_arrow" in file]
# import pdb; pdb.set_trace()

# run_counter = [ds_alignment(arrow_file) for arrow_file in arrow_files[0]]


combined_dataset = concat_ds(arrow_files)

#saving locally
combined_dataset.save_to_disk(out_path)
#split the dataset into train test validation
# import pdb; pdb.set_trace()

combined_dataset.push_to_hub(concat_name)



