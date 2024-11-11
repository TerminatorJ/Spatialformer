from datasets import load_dataset, load_from_disk
import os
import json
from tqdm import tqdm
hf_cache = "/scratch/project_465001027/Spatialformer/cache"


#loading David dataset
david_dataset = load_dataset("TerminatorJ/xenium_25_lung_dataset_update4", cache_dir = hf_cache, num_proc = 8)

def run(split):
    # Dictionary to hold annotations grouped by data_name
    annotations_by_id = {}
    niche_by_id = {}
    # Group annotations by data_name
    for record in tqdm(david_dataset[split]):
        cellid = record['Cell_Ids']
        annotation = record['Annotations']
        dataname = record["Sample_Names"]
        niche = record["Niche_Annotations"]

        annotations_by_id.setdefault(dataname, {}).setdefault(cellid, annotation)
        niche_by_id.setdefault(dataname, {}).setdefault(cellid, niche)
    return annotations_by_id, niche_by_id

def save_ann(all_ann, all_niche):
    # Save each annotation dictionary to its own file
    output_directory = '/scratch/project_465001027/Spatialformer/data'
    for data_name in tqdm(all_ann.keys()):
        file_path = os.path.join(output_directory, f"{data_name}_annotations.json")
        with open(file_path, 'w') as file:
            # Convert the list of annotations to JSON and save
            json.dump(all_ann[data_name], file, indent=4)
    for data_name in tqdm(all_niche.keys()):
        file_path = os.path.join(output_directory, f"{data_name}_niches.json")
        with open(file_path, 'w') as file:
            # Convert the list of annotations to JSON and save
            json.dump(all_niche[data_name], file, indent=4)

all_ann = {}
all_niche = {}
annotations_by_id_train, niche_by_id_train = run("train")
annotations_by_id_test, niche_by_id_test = run("test")
annotations_by_id_val, niche_by_id_val = run("validation")
all_ann.update(annotations_by_id_train)
all_ann.update(annotations_by_id_test)
all_ann.update(annotations_by_id_val)
all_niche.update(niche_by_id_train)
all_niche.update(niche_by_id_test)
all_niche.update(niche_by_id_val)
save_ann(all_ann, all_niche)

