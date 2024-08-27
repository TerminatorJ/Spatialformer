
from datasets import DatasetDict, load_dataset, concatenate_datasets, load_from_disk
import numpy as np
divides = 12
hf_cache = "/tmp/erda/Spatialformer/"
train_datasets = []
test_datasets = []
val_datasets = []
for i in range(1, divides+1):
    tokenized_datasets = load_from_disk(f"/tmp/erda/Spatialformer/xenium_25_lung_dataset_update2_{i}")
    train_datasets.append(tokenized_datasets["train"])
    test_datasets.append(tokenized_datasets["test"])
    val_datasets.append(tokenized_datasets["validation"])
    # dataset = load_dataset(f"/tmp/erda/Spatialformer/xenium_25_lung_dataset_update2_{i}", cache_dir = hf_cache, num_proc = 10)
    # datasets_list.append(dataset)
# import pdb; pdb.set_trace()
# Function to format the Distance_Matrix values
def format_distance_matrix(example):
    if 'Distance_Matrix' in example:
        # Assuming Distance_Matrix is a list or array


        example['Distance_Matrix'] = [
            np.round(np.array(row), 2).tolist() for row in example['Distance_Matrix']
        ]
        example['pct_nucleus'] = [
            [round(value, 2) for value in row] for row in example['pct_nucleus']
        ]

        # import pdb;pdb.set_trace()
    return example

all_train_dataset = concatenate_datasets(train_datasets)
all_test_dataset = concatenate_datasets(test_datasets)
all_val_dataset = concatenate_datasets(val_datasets)

new_dataset = DatasetDict({
                'train': all_train_dataset,
                'test': all_test_dataset,
                'validation': all_val_dataset
            })

# Apply the formatting function to each dataset
for split in new_dataset.keys():
    new_dataset[split] = new_dataset[split].map(format_distance_matrix, batched = True, batch_size = 2000)

# import pdb; pdb.set_trace()
# new_dataset.save_disk
new_dataset.save_to_disk("/tmp/erda/Spatialformer/xenium_25_lung_dataset_update3")
new_dataset.push_to_hub("TerminatorJ/xenium_25_lung_dataset_update3")