from pathlib import Path
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from datasets import load_from_disk
from datasets import DatasetDict, load_dataset, concatenate_datasets
import signal
import torch
from pytorch_lightning.strategies import DDPStrategy
import json
import numpy as np
import logging
from typing import Dict, Any
#setting the python environment
import sys
import pandas as pd
import sys
from spatialformer.utils import *
from spatialformer.model import Spaformer
from spatialformer.dataloader import PairwiseSpatialDataModule
import pwd

#setting the wandb environment
os.environ["WANDB_CACHE_DIR"] = "/scratch/project_465001820/Spatialformer/cache"
os.environ["WANDB_DIR"] = "/scratch/project_465001820/Spatialformer/cache"
os.environ["WANDB_CONFIG_DIR"] = "/scratch/project_465001820/Spatialformer/cache"
os.environ["WANDB_CACHE_DIR"] = "/scratch/project_465001820/Spatialformer/cache"
os.environ['HOME'] = pwd.getpwuid(os.getuid()).pw_dir
hf_cache = "/scratch/project_465001820/Spatialformer/cache"
os.environ["HF_HOME"] = hf_cache
cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(f"The number of cpu for each task: {cpus_per_task}")

if torch.cuda.is_available():
    os.environ["flash_attn_2_cuda"] = "True"
else:
    os.environ["flash_attn_2_cuda"] = "False"
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
                        warmup=config['warmup'],
                        use_flash_attn=config["use_flash_attn"],
                        lr=config['lr'],
                        max_epochs=config['max_epochs'],
                        mask_way = config['mask_way'], 
                        outer_config = config)
                      
    return model



#set up the model preparation
class preparation:
    """
    preparation
    ----------
    Description:
        Encapsulates configuration and setup logic required to train or run predictions with the SpatialFormer model.
        The class exposes properties that lazily construct and return the configured model, trainer, and data
        modules, and provides helper methods to run training and prediction flows.
  
    Properties:
        pre_model -> SpatialFormer instance
            - Constructs an SpatialFormer model using self.model_config.
            - If resume_from_wandb_model_name or resume_from_local_checkpoint is provided in model_config:
                - If wandb artifact name is provided, initializes a wandb run, downloads the artifact, and expects
                  a checkpoint file at "model.ckpt" inside the artifact directory.
                - If resume_from_local_checkpoint is provided, uses that path directly.
                - Loads the checkpoint with torch.load(map_location="cpu") and extracts ['state_dict'].
                - Calls model.load_state_dict(new_state_dict, strict=False) to load weights and logs missing/unexpected keys.
            - If no checkpoint is provided, returns a freshly initialized model.
            - Side effects: may contact wandb (network), perform file I/O, and log via logging.
        pre_trainer -> pytorch_lightning.Trainer
            - Configures a WandbLogger with project "SpatialFormer", name "pilot", log_model="all", and save_dir=self.output_dir.
            - Defines callbacks:
                - ModelCheckpoint that saves every_n_train_steps=2000 to output_dir/checkpoints with a filename template
                  including step, train_total_loss and val_total_loss; saves all checkpoints (save_top_k=-1).
                - LearningRateMonitor with logging_interval="step".
            - Depending on model_config["EET_analysis"]["gene_motif_fold"]:
                - If not None: creates a Trainer configured for distributed usage (precision='bf16', accelerator='auto',
                  devices=-1, num_nodes self.num_nodes, max_steps from config, no explicit strategy).
                - Otherwise: creates a Trainer with similar settings plus strategy from model_config["strategy"],
                  gradient clipping (value, 0.5), and other parameters.
            - Key Trainer args used: logger, accelerator, max_epochs, devices=-1, strategy (optional), num_nodes,
              max_steps, log_every_n_steps=50, check_val_every_n_epoch=1, default_root_dir=self.output_dir,
              callbacks, precision='bf16', accumulate_grad_batches from config.
            - The Trainer's world_size attribute is later used when creating data modules.
        pre_dataset -> MerlinDataModuleDistributed
            - Loads motif_tokens.json from a fixed path: "/scratch/project_465001820/spatialformer/spatialformer/tokenizer/motif_tokens.json".
            - Extracts sorted motif keys as motif columns.
            - Constructs MerlinDataModuleDistributed with:
                - path=self.datapath
                - columns=motif_columns
                - train_frac from model_config
                - batch_size from model_config
                - world_size set to self.pre_trainer.world_size
                - splits=False
            - Returns the configured data module.
        pre_pred_dataset -> MerlinDataModuleDistributed
            - Same as pre_dataset but passes drop_last=True to the data module to ensure deterministic batch sizes during prediction.
    Methods:
        train_model() -> spatialformer
            - Retrieves model (pre_model), trainer (pre_trainer), and datamodule (pre_dataset).
            - Calls trainer.fit(model, datamodule).
            - Returns the model instance after training (or as-is if fit encounters an exception).
        pred_model() -> Any
            - Retrieves model (pre_model), trainer (pre_trainer), and datamodule (pre_pred_dataset).
            - Calls trainer.predict(model, datamodule) and returns the list/collection of predictions produced by Lightning.
            - Note: post-processing of predictions (aggregation, decoding) is not performed here and is left to the caller.
    Notes, caveats, and recommendations:
        - The class relies on many global imports and project-specific implementations (MerlinDataModuleDistributed, spatialformer).
          Ensure these are available in the runtime environment.
        - The motif_tokens.json path is hardcoded. If running in a different environment, update the path or provide a
          mechanism to override it.
        - The constructor modifies environment variables and attempts to initialize torch.distributed when a specific
          configuration flag is set. This can have global effects for the process; instantiate carefully.
        - Loading checkpoints is done with strict=False; unexpected or missing keys are logged but not enforced. This may
          be desirable for fine-tuning but could mask mismatches.
        - Wandb usage: resuming from a Wandb artifact initiates a wandb.run; make sure wandb is configured (API key,
          network) when resuming artifacts.
        - The Trainer uses precision='bf16' which requires appropriate hardware (AMP-capable or bfloat16 support).
        - The class currently constructs the Wandb logger with a fixed run name "pilot"; consider making this configurable.
    Example usage:
        (Illustrative — actual imports and configuration keys must match project code)
            cfg = {...}  # model and training config
            prep = preparation(model_config=cfg, datapath="/data", output_dir="/output")
            trainer = prep.pre_trainer
            model = prep.pre_model
            dm = prep.pre_dataset
            trainer.fit(model, dm)
    """
    
    def __init__(self, 
                 model_config:Dict[str, Any],
                 datapath:str = None,
                 output_dir:str = None,
                 wandb_mode:str = "online",
                 input_type:str = "single",
                 run_name: str = "pilot",
                 ):
        self.model_config = model_config
        self.datapath = datapath
        self.run_name = run_name
        # get the pair parquet path from the single parquet path
        self.wandb_mode = wandb_mode
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        self.input_type = input_type
        #customized for the enlarge embedding size and spatial embedding size
        self.extend_keys = {
            'embeddings.weight',
            'spatialembeds.emb.weight',
            'classifier_head.weight',
            'classifier_head.bias',
        }
    @property
    def pre_model(self):
        
        # loading the model
        model = manual_train_fm(self.model_config)
        return model
    
    @property
    def pre_trainer(self):
        import wandb
        if self.wandb_mode == "offline":
            logger = WandbLogger(project="Spaformer", 
                        name=self.run_name, 
                        offline=(self.wandb_mode == "offline"),
                        save_dir=self.output_dir,
                        settings=wandb.Settings(
                        start_method="thread",   # avoid multiprocessing issues
                        _disable_stats=True      # disable background system metric tracking
                        ))
        else:
            logger = WandbLogger(project="Spaformer", 
                        name=self.run_name, 
                        log_model="best", 
                        save_dir=self.output_dir,
                        settings=wandb.Settings(
                        start_method="thread",   # avoid multiprocessing issues
                        _disable_stats=True      # disable background system metric tracking
                        ))
        # logger = CSVLogger(self.output_dir, name="just_test")

        
        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(self.output_dir, "checkpoints"),
                filename="step{step:07d}-train{train_total_loss:.4f}-val{val_total_loss:.4f}",
                every_n_train_steps=2000,
                save_top_k=-1,
                monitor='train_total_loss',
                save_on_train_epoch_end=False
            ), LearningRateMonitor(logging_interval="step")] 
        if self.model_config["val_check_interval"] is None:
            trainer = pl.Trainer(
                        logger=logger,
                        accelerator="auto",
                        max_epochs=self.model_config["max_epochs"],
                        devices=torch.cuda.device_count(), 
                        strategy=self.model_config["strategy"],
                        num_nodes = self.num_nodes,
                        max_steps=self.model_config["total_step"],
                        log_every_n_steps=50,
                        limit_val_batches=0,  # Disable validation
                        num_sanity_val_steps=0,  # Disable sanity check before training
                        default_root_dir=self.output_dir,
                        callbacks=callbacks,
                        precision='bf16', #bf16
                        gradient_clip_val=1,
                        gradient_clip_algorithm="norm",
                        accumulate_grad_batches = self.model_config['accumulate_grad_batches'])
        else:
            trainer = pl.Trainer(
                            logger=logger,
                            accelerator="auto",
                            max_epochs=self.model_config["max_epochs"],
                            devices=torch.cuda.device_count(), 
                            strategy=self.model_config["strategy"],
                            num_nodes = self.num_nodes,
                            max_steps=self.model_config["total_step"],
                            log_every_n_steps=50,
                            val_check_interval=self.model_config["val_check_interval"],
                            default_root_dir=self.output_dir,
                            callbacks=callbacks,
                            precision='bf16', #bf16
                            gradient_clip_val=1,
                            gradient_clip_algorithm="norm",
                            accumulate_grad_batches = self.model_config['accumulate_grad_batches'])

        return trainer

    def prepare_extended_checkpoint(self, model, ckpt_path, old_size=1950, new_size=6065):
        """
        Extend tensors and rebuild optimizer param_groups for model architecture changes.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        model.cpu()
        model_state = model.state_dict()
        
        # Count actual parameters (not buffers)
        total_params = sum(1 for _ in model.parameters())
        
        logging.info(f"[INFO] Model has {total_params} parameters, {len(model_state)} state_dict keys")
        
        # ============================================
        # 1. Fix state_dict
        # ============================================
        logging.info("[STATE_DICT] Fixing...")
        new_state_dict = {}
        
        for key, new_param in model_state.items():
            new_param = new_param.cpu().clone()
            
            if key in ckpt['state_dict']:
                old_param = ckpt['state_dict'][key].cpu().clone()
                
                if old_param.shape == new_param.shape:
                    new_state_dict[key] = old_param
                elif old_param.shape[0] == old_size and new_param.shape[0] == new_size:
                    logging.info(f"  [EXTEND] {key}: {old_param.shape} -> {new_param.shape}")
                    if old_param.dim() == 2:
                        new_param[:old_size, :] = old_param
                    elif old_param.dim() == 1:
                        new_param[:old_size] = old_param
                    new_state_dict[key] = new_param
                else:
                    logging.info(f"  [SHAPE MISMATCH] {key}: {old_param.shape} -> {new_param.shape}")
                    new_state_dict[key] = new_param
            else:
                logging.info(f"  [NEW] {key}")
                new_state_dict[key] = new_param
        
        # Log deleted keys
        for key in ckpt['state_dict']:
            if key not in model_state:
                logging.info(f"  [DELETED] {key}")
        
        ckpt['state_dict'] = new_state_dict
        
        # ============================================
        # 2. Rebuild optimizer_states completely
        # ============================================
        logging.info(f"\n[OPTIMIZER] Rebuilding for {total_params} parameters...")
        
        # Extract old settings if available
        old_settings = {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0,
            'amsgrad': False,
            'initial_lr': 0.001,
        }
        
        if 'optimizer_states' in ckpt and ckpt['optimizer_states']:
            opt_state = ckpt['optimizer_states'][0]
            if 'param_groups' in opt_state and opt_state['param_groups']:
                for k, v in opt_state['param_groups'][0].items():
                    if k != 'params':
                        old_settings[k] = v
        
        # Rebuild param_groups with correct count
        new_param_group = old_settings.copy()
        new_param_group['params'] = list(range(total_params))
        
        ckpt['optimizer_states'] = [
            {
                'state': {},
                'param_groups': [new_param_group]
            }
        ]
        logging.info(f"  [OK] Rebuilt with {total_params} params")
        
        # ============================================
        # 3. Reset lr_schedulers
        # ============================================
        if 'lr_schedulers' in ckpt:
            logging.info("[SCHEDULER] Resetting")
            ckpt['lr_schedulers'] = []
        
        # ============================================
        # 4. Save
        # ============================================
        logging.info(f"\n[INFO] Epoch: {ckpt.get('epoch', 0)}, Step: {ckpt.get('global_step', 0)}")
        
        new_ckpt_path = ckpt_path.replace('.ckpt', '_extended.ckpt')
        torch.save(ckpt, new_ckpt_path)
        logging.info(f"[SAVED] {new_ckpt_path}")
        
        return new_ckpt_path
    
    @property
    def pre_dataset(self):
        #as the same format to fetch columns
        if self.input_type not in ["single", "pair"]:
            logging.error(f"Invalid input_type: {self.input_type}. Must be 'single' or 'pairwise'.")

            

        datamodule = PairwiseSpatialDataModule(path=self.datapath, 
                        suffix=self.model_config['suffix'],
                        train_frac=self.model_config['train_frac'],
                        batch_size=self.model_config['batch_size'],
                        num_workers=min(cpus_per_task,64),
                        positive_threshold=self.model_config["positive_threshold"],
                        hard_negative_min=self.model_config['hard_negative_min'],
                        hard_negative_max=self.model_config["hard_negative_max"],
                        num_positives_per_query=self.model_config["num_positives_per_query"],  # Sample N positives per query
                        num_hard_negatives_per_query=self.model_config["num_hard_negatives_per_query"],
                        num_easy_negatives_per_query=self.model_config["num_easy_negatives_per_query"],
                        use_gpu_faiss=False,
                        pin_memory=False,
                        persistent_workers=False,
                        input_type=self.input_type,
                        use_cuda_in_collator=False,
                        num_precompute_workers=80,  
                        slide_name=self.model_config["slide_name"],
                        no_sparse=self.model_config["no_sparse"]
                        )

 
        # import pdb; pdb.set_trace()
        return datamodule
    
    @property
    def pre_pred_dataset(self):
        #as the same format to fetch columns
        if self.input_type not in ["single", "pair"]:
            logging.error(f"Invalid input_type: {self.input_type}. Must be 'single' or 'pairwise'.")


        datamodule = PairwiseSpatialDataModule(path=self.datapath, 
                        suffix=self.model_config['suffix'],
                        train_frac=self.model_config['train_frac'],
                        batch_size=self.model_config['batch_size'],
                        num_workers=min(cpus_per_task,64),
                        positive_threshold=self.model_config["positive_threshold"],
                        hard_negative_min=self.model_config['hard_negative_min'],
                        hard_negative_max=self.model_config["hard_negative_max"],
                        num_positives_per_query=self.model_config["num_positives_per_query"],  # Sample N positives per query
                        num_hard_negatives_per_query=self.model_config["num_hard_negatives_per_query"],
                        num_easy_negatives_per_query=self.model_config["num_easy_negatives_per_query"],
                        use_gpu_faiss=False,
                        pin_memory=False,
                        persistent_workers=False,
                        input_type=self.input_type,
                        use_cuda_in_collator=False,
                        num_precompute_workers=80,  
                        slide_name=self.model_config["slide_name"],
                        no_sparse=self.model_config["no_sparse"]
                        )

 
        return datamodule           
    
    def train_model(self):
        model = self.pre_model
        if self.model_config.get("resume_from_local_checkpoint", None):
            if self.model_config["resume_before_5k"]:
                # Prepare checkpoint with extended embeddings
                ckpt_path = self.prepare_extended_checkpoint(model, self.model_config["resume_from_local_checkpoint"])
            else:
                ckpt_path = self.model_config["resume_from_local_checkpoint"]
        else:
            ckpt_path = None
        trainer = self.pre_trainer
        datamodule = self.pre_dataset
        # Resume training with extended checkpoint
        trainer.fit(model, datamodule, ckpt_path=ckpt_path)
        import pdb; pdb.set_trace()
        return model
    
    def pred_model(self):
        import pdb; pdb.set_trace()
        model = self.pre_model
        import pdb; pdb.set_trace()
        trainer = self.pre_trainer
        import pdb; pdb.set_trace()
        datamodule = self.pre_pred_dataset
        import pdb; pdb.set_trace()
        predictions = trainer.predict(model, datamodule)
        import pdb; pdb.set_trace()
        # Process predictions as needed
        return predictions
        
    



if __name__ == "__main__":
    with open(os.path.join("/scratch/project_465001820/Spatialformer/spatialformer/config/_config_train_large_pair.json"), 'r') as json_file:
        config = json.load(json_file)

    
    
