#testing the david data, this script is used to get the cell center coordinates for calculating the adjacency matrix
from datasets import load_from_disk
import os
import pandas as pd
from tqdm import tqdm
import json
#loading the data
# combined_dataset = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset3")

def get_centroid():
    '''
    output is a dict that contain {sample_name: {cell_id: (x,y)}}
    '''
    #getting the cell files in batch
    # List all files in the specified directory

    all_files = os.listdir("/scratch/project_465001027/spatialformer/david_data")

    centroid_dict = {}
    counter = 0
    for file_path in tqdm(all_files):

        if "relabel" in file_path:
            sample_name = file_path.split("__")[-3]
            counter += 1
            # import pdb; pdb.set_trace()
            # Construct the path to the "outs" directory
            content = os.path.join("/scratch/project_465001027/spatialformer/david_data", file_path, "outs")
            
            # List all filenames in the "outs" directory
            
            
            try:
                target_file = os.path.join(content, "cells.csv.gz")
                df = pd.read_csv(target_file, compression='gzip')
            except:
                target_file = os.path.join(content, "cells.csv")
                print("Not a gzipped file")
                df = pd.read_csv(target_file)
            
            cell_c = dict(zip(df["cell_id"], zip(df["x_centroid"], df["y_centroid"])))
            centroid_dict[sample_name] = cell_c
            assert len(cell_c) == df.shape[0], "The cell number is not match the dict length"
    print(f"{counter} files finished")
    return centroid_dict

def add_centroid_column(example):
    # Lookup the centroid using values from the sample
    centroid = all_centroid_dict[example["Sample_Names"]][example["Cell_Ids"]]
    # Add the 'centroid' information to the example
    example["centroid_x"] = round(centroid[0], 2)
    example["centroid_y"] = round(centroid[1], 2)
    return example


if __name__ == "__main__":
    # centroid_dict = get_centroid()
    # print("Job done, saving to json")
    # with open("/scratch/project_465001027/Spatialformer/data/david_centroid_dict.json", "w") as f1:
    #     json.dump(centroid_dict, f1, indent = 4)

    #Getting all the downloaded dict #the downloaded file is ran on the DIKU cluster
    with open("/scratch/project_465001027/Spatialformer/data/downloaded_centroid_dict.json", "r") as f1:
        downloaded_centroid_dict = json.load(f1)

    #Getting all the david dict
    with open("/scratch/project_465001027/Spatialformer/data/david_centroid_dict.json", "r") as f2:
        david_centroid_dict = json.load(f2)
    all_centroid_dict = {}
    all_centroid_dict.update(downloaded_centroid_dict)
    all_centroid_dict.update(david_centroid_dict)
    print(f"The number of samples: {len(all_centroid_dict)}")
    
    #loading the data
    combined_dataset = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset3")  
    #adding the centroid information by mapping
    
    combined_dataset_adj = combined_dataset.map(add_centroid_column, num_proc=40)
    #saving the new data
    combined_dataset_adj.save_to_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset4", num_proc=40)
    





        
