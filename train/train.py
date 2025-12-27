from pathlib import Path
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]
model_path = os.path.join(p_path, "model")
util_path = os.path.join(p_path, "utils")
loader_path = os.path.join(p_path, "spatialformer")
sys.path.append(model_path)
sys.path.append(util_path)
sys.path.append(loader_path)
from utils import *
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal
import json
import numpy as np
import logging
from dataloader import PairwiseSpatialDataModule
from Spaformer_pair import Spaformer
from datasets import DatasetDict, load_dataset, concatenate_datasets, load_from_disk
import argparse

os.environ["WANDB_CACHE_DIR"] = "/scratch/project_465001820/Spatialformer/cache"
os.environ["WANDB_DIR"] = "/scratch/project_465001820/Spatialformer/cache"
os.environ["WANDB_CONFIG_DIR"] = "/scratch/project_465001820/Spatialformer/cache"
os.environ["WANDB_CACHE_DIR"] = "/scratch/project_465001820/Spatialformer/cache"
hf_cache = "/scratch/project_465001820/spatialformer/cache"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#track the NAN loss
torch.autograd.set_detect_anomaly(True)

#max length: 2579
# Define the model
def manual_train_fm(config=None):
    """
    Loading the Spaformer model with specified configurations.
    """
    pl.seed_everything(42)
    model = Spaformer(dim_model=config['dim_model'], 
                        nheads=config['nheads'], 
                        nlayers=config['nlayers'],
                        dropout=config['dropout'],
                        masking_p=config['masking_p'], 
                        n_tokens=config['n_tokens'],
                        n_atokens=config['n_atokens'],
                        context_length=config['context_length'],
                        warmup=config['warmup'],
                        lr=config['lr'],
                        max_epochs=config['max_epochs'],
                        pool=config['pool'],
                        bpp=config['bpp'],
                        bpp_scale = config['bpp_scale'],
                        ag_loss = config['ag_loss'],
                        mask_way = config['mask_way'], 
                        outer_config = config)
                      

    return model
 






class MyTrainer:
    def __init__(self, config):
        self.config = config
        self.plmodel = manual_train_fm(config=config)
        self.output_dir = "/scratch/project_465001820/Spatialformer/output"
        self.gpus = torch.cuda.device_count()
        self.num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        logging.info(f"The number of GPUS: {self.gpus}")
        logging.info(f"The number of nodes: {self.num_nodes}")
        self.trainer = None

    def make_callback(self):
        # Callbacks
        callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(self.output_dir, "checkpoints"),
            filename=f"{{step:07d}}-{{train_total_loss:.4f}}-{{val_total_loss:.4f}}",
            every_n_train_steps=4000,
            save_top_k=-1,
            monitor='train_total_loss',
            save_on_train_epoch_end=False
        ), LearningRateMonitor(logging_interval="step"),
        ]

        return callbacks
    def set_trainer(self):
        self.logger = WandbLogger(project = "Spaformer", 
                                  name = "pilot", 
                                  log_model = "all", 
                                  save_dir = self.output_dir)
        # self.logger = CSVLogger("/home/sxr280/Spatialformer/output", name="my_experiment")
        
        self.trainer = pl.Trainer(
            # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
            accelerator="auto",
            devices=self.gpus,
            # devices=1,
            max_steps=self.config["total_step"],
            val_check_interval = 1.0,
            default_root_dir=self.output_dir,
            num_sanity_val_steps=0,
            callbacks=self.make_callback(),
            log_every_n_steps=50,
            logger=self.logger,
            precision='bf16',
            strategy = self.config['strategy'],
            num_nodes = self.num_nodes,
            gradient_clip_val = 1,
            accumulate_grad_batches = self.config['accumulate_grad_batches']
        )
    def resume_train(self, ckp, train_dataloader=None, val_dataloader=None, datamodule=None):
        self.logger = WandbLogger(project = "Spaformer", 
                                  name = "pilot", 
                                  log_model = "all", 
                                  save_dir = self.output_dir)
        logging.info("resuming the training ...")
        self.trainer = pl.Trainer(
            accelerator="auto",
            devices=self.gpus,
            strategy = self.config['strategy'],
            num_nodes = self.num_nodes,
            val_check_interval = 1.0,
            num_sanity_val_steps=0,
            gradient_clip_val = 1,
            logger=self.logger,
            default_root_dir=self.output_dir,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            precision='bf16',
            callbacks=self.make_callback(),
            max_steps=self.config["total_step"], 
            resume_from_checkpoint=ckp,
            accumulate_grad_batches = self.config['accumulate_grad_batches'])
        if datamodule==None:
            # self.trainer.fit(self.plmodel, self.data_module)
            self.trainer.fit(self.plmodel, train_dataloader, val_dataloader)
        else:
            import pdb; pdb.set_trace()
            self.trainer.fit(self.plmodel, datamodule)


    def train(self, ckp_path, train_dataloader=None, val_dataloader=None, datamodule=None):
        self.set_trainer()
        if ckp_path is not None:
            print("loading the pre-trained weights")
            self.plmodel = self.load_pretrained_lm_weights(ckp_path)
        if datamodule==None:
            # self.trainer.fit(self.plmodel, self.data_module)
            self.trainer.fit(self.plmodel, train_dataloader, val_dataloader)
        else:
            self.trainer.fit(self.plmodel, datamodule)
        # self.trainer.fit(self.plmodel, train_dataloader, val_dataloader)
    def load_pretrained_lm_weights(self, ckp_path):
        ckp = torch.load(ckp_path, map_location=device)
        params = ckp["state_dict"]
        self.plmodel.load_state_dict(params)
        # base_model.eval()
        self.plmodel.to(device)
        return self.plmodel
    def test(self, test_loader):
        if self.config['pretrained_weights_path'] is not None:
            self.plmodel.load_pretrained_lm_weights()
        self.trainer.test(model=self.plmodel, dataloaders = test_loader)

        




    

def mean_length_of_full_tokens(dataset_split):
    lengths = [len(tokens) for tokens in dataset_split['Full_Tokens']]
    return np.mean(lengths)


def get_all_dataset(file_names):
    train_datasets = []
    test_datasets = []
    val_datasets = []
    all_mean = []
    for i, name in enumerate(file_names):
        remote_name = "TerminatorJ/"+name
        sta_datasets = load_dataset(remote_name, cache_dir = hf_cache, num_proc = 1)
        train_datasets.append(sta_datasets["train"])
        test_datasets.append(sta_datasets["test"])
        val_datasets.append(sta_datasets["validation"])
        mean_length_train = mean_length_of_full_tokens(sta_datasets['train'])
        mean_length_test = mean_length_of_full_tokens(sta_datasets['test'])
        mean_length_validation = mean_length_of_full_tokens(sta_datasets['validation'])
        mean_length = np.mean([mean_length_train, mean_length_test, mean_length_validation])
        all_mean.append(mean_length)
        print("mean length of %s is " % name.split("/")[-1], mean_length)
    logging.info(f"overall mean lenght of these dataset is {np.mean(all_mean)}")
    return train_datasets, test_datasets, val_datasets



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training the model.")
    # Add arguments
    parser.add_argument('--config', type=str, required=True, help='The configuration file path for training the model.')
    # Parse the arguments
    args = parser.parse_args()
    
    config_path = args.config
    
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)

    input_mode = config["input_mode"]
    meta_counter = int(config["organ"]) + int(config["specie"]) + int(config["assay"]) + int(config["condition"])
    
    #training the model with the single input mode
    if input_mode == "single":
        from data_loader import create_single_data_loaders
        combined_dataset = load_dataset("TerminatorJ/xenium_pandavid_dataset4",cache_dir = "/scratch/project_465001820/Spatialformer/cache/", num_proc=8)
        combined_dataset = concatenate_datasets([combined_dataset["train"], combined_dataset["test"], combined_dataset["validation"]])
        train_dataloader, val_dataloader = create_single_data_loaders(combined_dataset, cls_token = 1, padding_idx = 0, 
                           sep_token = 1949, batch_size=config["batch_size"], context_length=config["context_length"], 
                           special_token_num = 4, split_num = 2, num_workers = 8)
        Trainer = MyTrainer(config = config)
        if config["retake_training"]:
            Trainer.resume_train(config["pretrained_path"], train_dataloader, val_dataloader)
        else:
            Trainer.train(config["pretrained_path"], train_dataloader, val_dataloader)
    elif input_mode == "pair":
        from Spaformer_pair import Spaformer
        # from data_loader import CustomDataModule
        from data_loader import create_dataloader
        from tqdm import tqdm
        
        datapath = "/scratch/project_465001820/Spatialformer/cache"
        # train_dataloader, val_dataloader = create_dataloader(datapath, 
        #                                                     num_workers = 16, 
        #                                                     batch_size = config["batch_size"],
        #                                                     # batch_size = 16,
        #                                                     directionality = config["directionality"],
        #                                                     context_length = config["context_length"], 
        #                                                     padding_idx = 0, 
        #                                                     special_token_num = meta_counter, 
        #                                                     n_bins = 51, 
        #                                                     sep_token = 1949, 
        #                                                     cls_token = 1)
        datamodule = PairwiseSpatialDataModule(path = "/scratch/project_465001820/Spatialformer/cache/xenium_5k_pandavid_dataset_v2", 
                        suffix="arrow",
                        train_frac=0.99,
                        batch_size=2,
                        num_workers=16,
                        positive_threshold=30,
                        hard_negative_min=50,
                        hard_negative_max=150,
                        num_positives_per_query=1,  # Sample N positives per query
                        num_hard_negatives_per_query=0,
                        num_easy_negatives_per_query=1,
                        use_gpu_faiss=False,
                        pin_memory=False,
                        persistent_workers=False,
                        input_type="pair",
                        use_cuda_in_collator=False,
                        num_precompute_workers=80,  
                        slide_name="Xenium_Prime_Human_Prostate_FFPE_xe_outs",
                        no_sparse=True
                        )

     
        Trainer = MyTrainer(config = config)                                

        if config["retake_training"]:
             Trainer.resume_train(config["pretrained_path"], datamodule = datamodule)
            # Trainer.resume_train(config["pretrained_path"], train_dataloader, val_dataloader)
            # Trainer.resume_train(config["pretrained_path"])
        else:
            Trainer.train(config["pretrained_path"], datamodule = datamodule)
            # Trainer.train(config["pretrained_path"], train_dataloader, val_dataloader)#if pretrained_path exist, we are fine-tuning the model
            #  Trainer.train()

   

#example command to run the training script
# python train.py --config /scratch/project_465001820/Spatialformer/config/_config_train_large_single.json
    

    

    
    