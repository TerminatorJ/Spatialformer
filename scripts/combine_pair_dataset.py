from datasets import concatenate_datasets, load_from_disk
import os
datapath = "/scratch/project_465001027/Spatialformer/cache"
all_files = os.listdir(datapath)
datasets_paths = [os.path.join(datapath, file) for file in all_files if (("TIL" in file) & ("pair" in file)) or (("THD" in file) & ("pair" in file)) or (("VU" in file) & ("pair" in file))]


all_datasets = []

for path in datasets_paths:
    dataset = load_from_disk(path)
    all_datasets.append(dataset)
conct_dataset = concatenate_datasets(all_datasets)
conct_dataset.save_to_disk("/scratch/project_465001027/Spatialformer/cache/lung_pairs_dataset")
