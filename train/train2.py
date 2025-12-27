import os
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
os.environ["TRITON_CACHE_DIR"] = "/scratch/project_465001820/spatialformer/spatialformer/cache/triton_cache"

from spatialformer.utils import *
import torch
import json
from train.train_config import preparation


if __name__ == "__main__":
        
    set_seed(43)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
    else:
        print("No GPUs available.") 
    # Load the JSON configuration file
    config_path = "/scratch/project_465001820/Spatialformer/spatialformer/config/_config_train_large_pair.json"
    with open(config_path, 'r') as json_file:
        model_config = json.load(json_file)
    #################Training input: Options (single; pair)#################    
    input_type = model_config["input_type"]  # "single" or "pair"
    # datapath = "/scratch/project_465001820/Spatialformer/cache/xenium_5k_combined_example_dataset"
    datapath = "/scratch/project_465001820/Spatialformer/cache/xenium_5k_pandavid_dataset_v2"
    ############################################################################
    #prepare the data and model
    ############################################################################
    if model_config["num_hard_negatives_per_query"] == 0:
        run_name = "add_xenium_5k_data_withouthardneg"
    else:
        run_name = "add_xenium_5k_data_withhardneg"

    if input_type == "single":
        run_name = "add_xenium_5k_data_single"
    ############################################################################
    pre = preparation(model_config=model_config,
                    datapath=datapath,
                    output_dir="/scratch/project_465001820/Spatialformer/output",
                    wandb_mode="online",
                    input_type=input_type,
                    run_name=run_name
                    )
    #train the model
    pre.train_model()