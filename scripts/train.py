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
sys.path.append(model_path)
sys.path.append(util_path)
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
from datasets import load_from_disk
from Spaformer_pair import Spaformer
from datasets import DatasetDict, load_dataset, concatenate_datasets

os.environ["WANDB_CACHE_DIR"] = "/scratch/project_465001027/Spatialformer/cache"
os.environ["WANDB_DIR"] = "/scratch/project_465001027/Spatialformer/cache"
os.environ["WANDB_CONFIG_DIR"] = "/scratch/project_465001027/Spatialformer/cache"
os.environ["WANDB_CACHE_DIR"] = "/scratch/project_465001027/Spatialformer/cache"
hf_cache = "/scratch/project_465001027/spatialformer/cache"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#track the NAN loss
torch.autograd.set_detect_anomaly(True)

def manual_train_fm(config=None):
    
    pl.seed_everything(42)
    # import pdb; pdb.set_trace()
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
        self.output_dir = "/scratch/project_465001027/Spatialformer/output"
        # self.data_module = data_module
        # Load the resume index from a file if needed
        # self.resume_index_path = os.path.join(self.output_dir , 'resume_index.pt')
        # if os.path.isfile(self.resume_index_path):
        #     self.data_module.load_state(self.resume_index_path)
        # self.data_module.setup()
        self.gpus = torch.cuda.device_count()
        # import pdb; pdb.set_trace()
        
        self.num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        logging.info(f"The number of GPUS: {self.gpus}")
        logging.info(f"The number of nodes: {self.num_nodes}")
        self.trainer = None
        # self.set_trainer()

    def make_callback(self):
        # Callbacks
        callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(self.output_dir, "checkpoints"),
            filename=f"{{step:07d}}-{{train_total_loss:.4f}}-{{val_total_loss:.4f}}",
            every_n_train_steps=4000,
            save_top_k=-1,
            # every_n_epochs=1,
            monitor='train_total_loss',
            save_on_train_epoch_end=False
        ), LearningRateMonitor(logging_interval="step"),
        # EarlyStopping(monitor = "val_loss", min_delta = 0.00, verbose = True, mode = "min")
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
    def resume_train(self, ckp, train_dataloader, val_dataloader):
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

        # self.trainer.fit(self.plmodel, self.data_module)
        self.trainer.fit(self.plmodel, train_dataloader, val_dataloader)


    def train(self, ckp_path, train_dataloader, val_dataloader):
        self.set_trainer()
        if ckp_path is not None:
            print("loading the pre-trained weights")
            self.plmodel = self.load_pretrained_lm_weights(ckp_path)
        self.trainer.fit(self.plmodel, train_dataloader, val_dataloader)
        # self.trainer.fit(self.plmodel, self.data_module)
        # self.data_module.save_state(self.resume_index_path)
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

class MultiDataLoader:
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        # Create iterators for each DataLoader
        iterators = [iter(loader) for loader in self.loaders]
        
        while True:
            for iterator in iterators:
                try:
                    yield next(iterator)
                except StopIteration:
                    continue  # Move to the next DataLoader if one is exhausted
            break  # Exit once all are exhausted

if __name__ == "__main__":
    
    # with open(os.path.join("/scratch/project_465001027/Spatialformer/config/_config_train_large_single.json"), 'r') as json_file:
    #     config = json.load(json_file)
    with open(os.path.join("/scratch/project_465001027/Spatialformer/config/_config_train_large_pair.json"), 'r') as json_file:
        config = json.load(json_file)

    # combined_dataset = load_dataset("TerminatorJ/xenium_pandavid_dataset", cache_dir = hf_cache, num_proc = 8)
        # import pdb; pdb.set_trace()
    input_mode = config["input_mode"]
    meta_counter = int(config["organ"]) + int(config["specie"]) + int(config["assay"]) + int(config["condition"])
    if input_mode == "single":
        from data_loader import create_single_data_loaders
        combined_dataset = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset4")
        # import pdb; pdb.set_trace()
        combined_dataset = concatenate_datasets([combined_dataset["train"], combined_dataset["test"], combined_dataset["validation"]])

        # combined_iter_dataset = combined_dataset.to_iterable_dataset(num_shards=64)
        # train_dataloader, val_dataloader = create_data_loaders(combined_dataset, batch_size=config["batch_size"], context_length=config["context_length"], special_token_num = meta_counter, directionality = config["directionality"])
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
        
        # combined_dataset = load_from_disk("/scratch/project_465001027/Spatialformer/cache/xenium_pandavid_dataset4") 
        #the index for fast retriving 
        # index_path = "/scratch/project_465001027/Spatialformer/data/sample_cell_index.pkl"
        #getting all the sample names
        # sample_cell_index = get_index(combined_dataset, save_file = index_path)

        # for sample_name in tqdm(list(sample_cell_index.keys())[:2]):
        # train_dataloader, val_dataloader = create_sample_data_loaders(
        #                                                             combined_dataset, 
        #                                                             sample_cell_index, 
        #                                                             radius = 30, 
        #                                                             batch_size = config["batch_size"], 
        #                                                             directionality = config["directionality"],
        #                                                             context_length = config["context_length"], 
        #                                                             padding_idx = 0,
        #                                                             special_token_num = meta_counter,
        #                                                             n_bins = 51,
        #                                                             sep_token = 1949,
        #                                                             cls_token = 1,
        #                                                             num_workers = 1,
        #                                                             pin_memory = False)
        
        datapath = "/scratch/project_465001027/Spatialformer/cache"
        train_dataloader, val_dataloader = create_dataloader(datapath, 
                                                            num_workers = 16, 
                                                            batch_size = config["batch_size"],
                                                            # batch_size = 16,
                                                            directionality = config["directionality"],
                                                            context_length = config["context_length"], 
                                                            padding_idx = 0, 
                                                            special_token_num = meta_counter, 
                                                            n_bins = 51, 
                                                            sep_token = 1949, 
                                                            cls_token = 1)

        # datamodule = CustomDataModule(datapath, 
        #                 batch_size=config["batch_size"], 
        #                 num_workers = 4, 
        #                 directionality = config["directionality"],
        #                 context_length = config["context_length"],
        #                 padding_idx = 0,
        #                 special_token_num = meta_counter,
        #                 n_bins = 51,
        #                 sep_token = 1949,
        #                 cls_token = 1)
        # Trainer = MyTrainer(config = config, data_module = datamodule)      
        Trainer = MyTrainer(config = config)                                
        # train_dataloader, val_dataloader = create_dataloader2(datapath, 
        #                                                     num_workers = 8, 
        #                                                     batch_size = config["batch_size"],
        #                                                     # batch_size = 16,
        #                                                     directionality = config["directionality"],
        #                                                     context_length = config["context_length"], 
        #                                                     padding_idx = 0, 
        #                                                     special_token_num = meta_counter, 
        #                                                     n_bins = 51, 
        #                                                     sep_token = 1949, 
        #                                                     cls_token = 1)
        if config["retake_training"]:
            Trainer.resume_train(config["pretrained_path"], train_dataloader, val_dataloader)
            # Trainer.resume_train(config["pretrained_path"])
        else:
            Trainer.train(config["pretrained_path"], train_dataloader, val_dataloader)#if pretrained_path exist, we are fine-tuning the model
            #  Trainer.train()

   

    

    

    
    