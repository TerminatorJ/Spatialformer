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
# import pdb; pdb.set_trace()
from Spaformer import Spaformer 
# import pdb; pdb.set_trace() 
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import json
import numpy as np
import logging
from datasets import load_from_disk
from datasets import concatenate_datasets, DatasetDict
from torch.utils.data import ConcatDataset
from h5toloader import get_dataset,create_data_loaders
os.environ["WANDB_CACHE_DIR"] = "/scratch/project_465001027/spatialformer/cache"
os.environ["AMD_SERIALIZE_KERNEL"] = "3"
torch.set_float32_matmul_precision("medium")
data_path = "/scratch/project_465001027/spatialformer/david_data"
data_name = [file.split("/")[-1] for file in os.listdir(data_path) if "relabel" in file]
adata_paths = [os.path.join(data_path, name, "processed", name + "." + "h5ad") for name in data_name]
other_array_path = ["/scratch/project_465001027/spatialformer/data/processed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs_arrow"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
                        learnable_pe=config['learnable_pe'],
                        specie=config['specie'],
                        assay=config['assay'],
                        modality=config['modality'])
                      

    return model
 
class MyTrainer:
    def __init__(self, config):
        self.config = config
        self.plmodel = manual_train_fm(config=config)
        self.output_dir = "/scratch/project_465001027/spatialformer/output"
        
        self.gpus = torch.cuda.device_count()
        self.trainer = None

    def make_callback(self):
        # Callbacks
        callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(self.output_dir, "checkpoints"),
            filename=f"{{step:07d}}-{{train_loss:.4f}}-{{val_loss:.4f}}",
            every_n_train_steps=1000,
            save_top_k=-1,
            # every_n_epochs=1,
            monitor='train_loss',
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
        
        self.trainer = pl.Trainer(
            accelerator="auto",
            devices=self.gpus,
            max_steps=60000,
            default_root_dir=self.output_dir,
            callbacks=self.make_callback(),
            log_every_n_steps=1000,
            check_val_every_n_epoch=50,
            logger=self.logger,
            precision='32',
            strategy = self.config['strategy'],
            num_nodes = 1,
            gradient_clip_val = 1,
            accumulate_grad_batches = self.config['accumulate_grad_batches']
        )
    def resume_train(self, ckp, train_loader, val_loader):
        self.logger = WandbLogger(project = "Spaformer", 
                                  name = "pilot", 
                                  log_model = "all", 
                                  save_dir = self.output_dir)
        # import pdb; pdb.set_trace()
        logging.info("resuming the training ...")
        self.trainer = pl.Trainer(
            accelerator="auto",
            devices=self.gpus,
            strategy = self.config['strategy'],
            num_nodes = 1,
            gradient_clip_val = 1,
            logger=self.logger,
            default_root_dir=self.output_dir,
            log_every_n_steps=1000,
            check_val_every_n_epoch=50,
            precision='bf16',
            callbacks=self.make_callback(),
            max_steps=60000, 
            resume_from_checkpoint=ckp,
            accumulate_grad_batches = self.config['accumulate_grad_batches'])
        self.trainer.fit(self.plmodel, train_loader, val_loader)


    def train(self, train_loader, val_loader):
        # import pdb; pdb.set_trace()
        self.set_trainer()
        self.trainer.fit(self.plmodel, train_loader, val_loader)

    def test(self, test_loader):
        if self.config['pretrained_weights_path'] is not None:
            self.plmodel.load_pretrained_lm_weights()
        self.trainer.test(model=self.plmodel, dataloaders = test_loader)
        
    

def mean_length_of_full_tokens(dataset_split):
    lengths = [len(tokens) for tokens in dataset_split['Full_Tokens']]
    return np.mean(lengths)


def get_all_adata(adata_paths):
    train_datasets = []
    test_datasets = []
    val_datasets = []
    all_mean = []
    for i, path in enumerate(adata_paths + other_array_path):
        # import pdb; pdb.set_trace()
        if i < len(adata_paths):
            sta_datasets = get_dataset(path)
        else:
            sta_datasets = load_from_disk(path)
        train_datasets.append(sta_datasets["train"])
        test_datasets.append(sta_datasets["test"])
        val_datasets.append(sta_datasets["validation"])

        mean_length_train = mean_length_of_full_tokens(sta_datasets['train'])
        mean_length_test = mean_length_of_full_tokens(sta_datasets['test'])
        mean_length_validation = mean_length_of_full_tokens(sta_datasets['validation'])
        mean_length = np.mean([mean_length_train, mean_length_test, mean_length_validation])
        all_mean.append(mean_length)
        print("mean length of %s is " % path.split("/")[-1], mean_length)



    logging.info(f"overall mean lenght of these dataset is {np.mean(all_mean)}")
    return train_datasets, test_datasets, val_datasets



if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    
    with open(os.path.join("/scratch/project_465001027/spatialformer/config/_config_train.json"), 'r') as json_file:
        config = json.load(json_file)

    # data_path = "/scratch/project_465001027/spatialformer/data/processed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.h5ad"
    # data_path = "/scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/processed/relabel_output-XETG00048__0003392__THD0008__20230313__191400.h5ad"
    # import pdb; pdb.set_trace()
    # tokenized_datasets = get_dataset(data_path)
    train_datasets, test_datasets, val_datasets = get_all_adata(adata_paths)
    
    #concating all the dataset
    combined_dataset = DatasetDict({
    'train': concatenate_datasets(train_datasets),
    'test': concatenate_datasets(test_datasets),
    'validation': concatenate_datasets(val_datasets)
    })



    
    # import pdb; pdb.set_trace()
    # train_dataloader, val_dataloader, test_dataloader = create_data_loaders(tokenized_datasets, batch_size=config["batch_size"], context_length=config["context_length"])
    train_dataloader, val_dataloader = create_data_loaders(combined_dataset, batch_size=config["batch_size"], context_length=config["context_length"])
    # for batch in train_dataloader:
    #     pass
    Trainer = MyTrainer(config = config)

    Trainer.train(train_dataloader, val_dataloader)
    # Trainer.resume_train("/scratch/project_465001027/spatialformer/output/checkpoints/step=0000002-val_loss=0.0000.ckpt", train_dataloader, val_dataloader)
    # print("skip the training, only getting the test performance###")
    # test 
    # Trainer.test(test_dataloader)
    # for i,batch in enumerate(test_dataloader):
    #     if i < 1:
    #         get = batch
    # import pdb; pdb.set_trace()
    # plmodel = Trainer.plmodel
    # plmodel.get_embeddings(get, [0])
    
    '''
    #testing the nan issue
    import pickle
    from torch.nn import init
    # os.environ["visible_cuda"] = "0"
    import pdb; pdb.set_trace()
    data = pickle.load(open("/scratch/project_465001027/spatialformer/scripts/save_input.pkl","rb"))
    # data = {i: [data[i][j].to("cpu") for j in range(len(data[i]))] for i in data}

    # pickle.dump(data, open("/scratch/project_465001027/spatialformer/scripts/save_input.pkl", "wb"))
    #loading the model
    with open(os.path.join("/scratch/project_465001027/spatialformer/config/_config_train.json"), 'r') as json_file:
        config = json.load(json_file)
    Trainer = MyTrainer(config = config)
    model = Trainer.plmodel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    first_param_before = next(iter(model.parameters()))
    masked_indices, adjmtx, attention_mask = data[12]
    masked_indices.to(model.device)
    adjmtx.to(model.device)
    attention_mask.to(model.device)
    
    
    predictions = model(masked_indices, adjmtx, attention_mask)

    import pdb; pdb.set_trace()
    # Assuming 'model' is your neural network model
    # for param in model.parameters():
    #     if param.requires_grad:
    #         print("checking whether the parameters are nan:",torch.isnan(param).any())
    #         init.normal_(param.data, mean=0, std=0.01)  # Initialize with Gaussian noise
    first_param_after = next(iter(model.parameters()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #loading the parameters
    import pdb; pdb.set_trace()
    ckp = torch.load("/scratch/project_465001027/spatialformer/output/checkpoints/step=0000003-train_loss=6.4212-val_loss=0.0000.ckpt")
    params = ckp["state_dict"]
    model.load_state_dict(params)

    masked_indices, adjmtx, attention_mask = data[12]
    #change to the same device
    masked_indices.to(model.device)
    adjmtx.to(model.device)
    attention_mask.to(model.device)
    #swich the torch as the size
    A = torch.ones_like(masked_indices)
    B = torch.ones_like(adjmtx)
    C = torch.one_like(attention_mask)
    predictions = model(A, B, attention_mask)


    predictions = model(masked_indices, adjmtx, attention_mask)
    '''





    

    

    
    