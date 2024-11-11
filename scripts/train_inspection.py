#testing the training dataloader
import os
import json
from datasets import load_dataset
from train import *
from tqdm import tqdm
hf_cache = "/scratch/project_465001027/spatialformer/cache"
with open(os.path.join("/scratch/project_465001027/Spatialformer/config/_config_train_small.json"), 'r') as json_file:
    config = json.load(json_file)

# combined_dataset = load_dataset("TerminatorJ/xenium_pandavid_dataset", cache_dir = hf_cache, num_proc = 4)
combined_dataset = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset2")
meta_counter = int(config["organ"]) + int(config["specie"]) + int(config["assay"]) + int(config["condition"])
# import pdb; pdb.set_trace()
train_dataloader, val_dataloader = create_data_loaders(combined_dataset, batch_size=config["batch_size"], context_length=config["context_length"], special_token_num = meta_counter, directionality = config["directionality"])

Trainer = MyTrainer(config = config)
model = Trainer.plmodel

for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Testing Progress"):
    # print(batch)
    hidden_repr = model.get_embeddings(batch, [-1])[0]
    
# Trainer = MyTrainer(config = config)
# if config["retake_training"]:
#     Trainer.resume_train(config["pretrained_path"], train_dataloader, val_dataloader)
# else:
#     Trainer.train(train_dataloader, val_dataloader)

   
