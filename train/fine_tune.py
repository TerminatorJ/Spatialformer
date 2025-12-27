
from data_loader import *
import json
from datasets import load_from_disk, concatenate_datasets, load_dataset
from train import manual_train_fm
import sys
from sklearn.model_selection import train_test_split
sys.path.append("/scratch/project_465001027/Spatialformer/model")
from Spaformer_ft import FTNetwork
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import pandas as pd
import argparse
import pickle
import time
import os
import torch

# Set NCCL environment variables
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def data_prepare(sample_name, kfold, num_workers, batch_size, radius=30, test_size = 2000, zero_shot_cell_size = 2000, split_mode = "leave_cell_out"):
    combined_dataset = load_dataset("TerminatorJ/xenium_pandavid_dataset4", cache_dir = "/scratch/project_465001820/Spatialformer/cache/", num_proc = 8) #loading the combined dataset
    # combined_dataset = load_from_disk("/scratch/project_465001820/Spatialformer/cache/xenium_pandavid_dataset4")  
    index_path = "/scratch/project_465001820/Spatialformer/data/sample_cell_index.pkl"
    sample_cell_index = get_index(combined_dataset, save_file = index_path)

    combined_dataset_all = concatenate_datasets([combined_dataset["train"], combined_dataset["test"], combined_dataset["validation"]])
    sample_dataset = combined_dataset_all.select(list(sample_cell_index[sample_name].values()))
    # sample_dataset = sample_dataset.select(range(200)) #use this for testing the pipeline
    #Getting the asymmetry matrix
    # import pdb; pdb.set_trace()
    sparse_adj, cell_ids = get_adj(sample_dataset, radius = radius, plot = False, sym = True)

    
    # import pdb; pdb.set_trace()
    ### Getting the cell pairs according to the distance
    Pairs = GetPairs(sparse_adj, num_workers = 8) #assign the 1:1 negative to the positive
    
    all_pairs = Pairs.all_pairs
    all_labels = Pairs.all_labels
    # reginal_split(all_pairs, all_labels)
    # import pdb; pdb.set_trace()
    #building the dataloader
    kfold_data_loader = {}
    test_data_loader = None
    if kfold:
        train_test_splits, train_test_labels = split_dataset(all_pairs, all_labels, n_splits = kfold, split_mode = split_mode, test_size = test_size) #all the k-fold splits
        
        



        for fold in range(kfold):
            train_pairs, test_pairs = train_test_splits[fold][0], train_test_splits[fold][1]
            train_labels, test_labels = train_test_labels[fold][0], train_test_labels[fold][1]

            #saving the 5 folds train-test pairs and labels
            pickle.dump(train_pairs, open(f"/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/train_pairs_fold{fold}.pkl", "wb"))
            pickle.dump(test_pairs, open(f"/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/test_pairs_fold{fold}.pkl", "wb"))
            pickle.dump(train_labels, open(f"/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/train_labels_fold{fold}.pkl", "wb"))
            pickle.dump(test_labels, open(f"/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/test_labels_fold{fold}.pkl", "wb"))


            # import pdb; pdb.set_trace()
            val_size = int(0.15 * len(train_pairs)) 
            train_pairs, val_pairs, train_labels, val_labels = train_test_split(
                train_pairs, 
                train_labels, 
                test_size=val_size, 
                random_state=42, 
                shuffle=True  # Shuffle the data before splitting
            )
            train_dataloader = create_dataloader_fast(sample_dataset, 
                                                    train_pairs,
                                                    train_labels,
                                                    split = "train",
                                                    num_workers = num_workers, 
                                                    batch_size = batch_size,
                                                    directionality = True,
                                                    context_length = 500, 
                                                    padding_idx = 0, 
                                                    special_token_num = 4, 
                                                    n_bins = 51, 
                                                    sep_token = 1949, 
                                                    cls_token = 1,
                                                    )
            test_dataloader = create_dataloader_fast(sample_dataset, 
                                                    test_pairs, 
                                                    test_labels,
                                                    split = "test",
                                                    num_workers = num_workers, 
                                                    batch_size = batch_size,
                                                    directionality = True,
                                                    context_length = 500, 
                                                    padding_idx = 0, 
                                                    special_token_num = 4, 
                                                    n_bins = 51, 
                                                    sep_token = 1949, 
                                                    cls_token = 1,
                                                    )
            val_dataloader = create_dataloader_fast(sample_dataset, 
                                                    val_pairs,
                                                    val_labels,
                                                    split = "val",
                                                    num_workers = num_workers, 
                                                    batch_size = batch_size,
                                                    directionality = True,
                                                    context_length = 500, 
                                                    padding_idx = 0, 
                                                    special_token_num = 4, 
                                                    n_bins = 51, 
                                                    sep_token = 1949, 
                                                    cls_token = 1,
                                                    )
            kfold_data_loader[fold] = (train_dataloader, test_dataloader, val_dataloader)
        return kfold_data_loader
    else:
        # import pdb; pdb.set_trace()
        selected_pairs, selected_labels = split_dataset(all_pairs, all_labels, n_splits = kfold, test_size = None, zero_shot_cell_size = zero_shot_cell_size)
        #save the zero-shot benchmarking dataset
        # import pdb; pdb.set_trace()
        pickle.dump(selected_pairs, open(f"/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/selected_pairs_radius_{radius}.pkl", "wb"))
        pickle.dump(selected_labels, open(f"/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/selected_labels_{radius}.pkl", "wb"))
        print("selected_pairs shape:", selected_pairs.shape)
        selected_pairs = np.array([[33921, 33929]])
        import pdb; pdb.set_trace()
        selected_labels = np.array([1])
        test_dataloader = create_dataloader_fast(sample_dataset, 
                                                    selected_pairs, 
                                                    selected_labels,
                                                    split = "test",
                                                    num_workers = num_workers, 
                                                    batch_size = batch_size,
                                                    directionality = True,
                                                    context_length = 500, 
                                                    padding_idx = 0, 
                                                    special_token_num = 4, 
                                                    n_bins = 51, 
                                                    sep_token = 1949, 
                                                    cls_token = 1,
                                                    )
        
        # import pdb; pdb.set_trace()

        
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        return test_dataloader

class FineTune:
    def __init__(self, config, ckp_path, sample_name, radius, fine_tune_mode, wandb, strategy):
        self.config = config
        self.sample_name = sample_name
        self.fold = None
        self.wandb = wandb
        self.radius = radius
        self.strategy = strategy
        self.total_steps = config["total_step"]
        self.fine_tune_mode = fine_tune_mode
        # import pdb; pdb.set_trace()
        self.base_model = self.load_pretrained_lm_weights(config, ckp_path) #loading the parameters of the model
        self.output_dir = "/scratch/project_465001027/Spatialformer/output/fine_tune"
        self.probe_model = FTNetwork(self.base_model, fine_tune_mode = fine_tune_mode, outer_config = config) #here is the proximity or not
        
        self.gpus = torch.cuda.device_count()
        # self.gpus = 1
        # import pdb; pdb.set_trace()
        self.num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        print(f"The number of GPUS: {self.gpus}")
        self.trainer = None

    def make_callback(self):
        # Callbacks
        callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(self.output_dir, "checkpoints"),
            filename=f"{{step:07d}}-{{train_loss:.4f}}_{self.sample_name}_{self.fold}_{self.radius}_{self.fine_tune_mode}_{self.total_steps}",
            every_n_train_steps=2000,
            save_top_k=-1,
            # every_n_epochs=1,
            monitor='val_loss',
            save_on_train_epoch_end=False
        ), LearningRateMonitor(logging_interval="step")]
        # EarlyStopping(monitor = "val_loss", min_delta = 0.00, verbose = True, mode = "min", patience=5)]
        

        return callbacks
    def set_trainer(self):
        if self.wandb:
            self.logger = WandbLogger(project = "Spaformer", 
                                    name = f"fine_tune_{self.radius}_{self.sample_name}_{self.fold}_{self.fine_tune_mode}_{self.total_steps}", 
                                    log_model = "all", 
                                    save_dir = self.output_dir)
        else:
            self.logger = None
        self.trainer = pl.Trainer(
            # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
            accelerator="auto",
            devices=self.gpus,
            check_val_every_n_epoch=1,
            # val_check_interval = 0.1,
            max_steps=self.config["total_step"],
            default_root_dir=self.output_dir,
            num_sanity_val_steps=0,
            callbacks=self.make_callback(),
            log_every_n_steps=10,
            logger=self.logger,
            precision='bf16',
            # precision=16,
            strategy = self.strategy,
            num_nodes = self.num_nodes,
            gradient_clip_val = 1,
            accumulate_grad_batches = self.config['accumulate_grad_batches']
        )

    def train(self, fold, train_dataloader, val_dataloader):
        self.fold = fold
        self.set_trainer()
        # self.probe_model.train()
        
        self.trainer.fit(self.probe_model, train_dataloader, val_dataloader)
        return self.probe_model
        # self.trainer.fit(self.plmodel, train_dataloader, val_dataloader)
        # self.data_module.save_state(self.resume_index_path)
    def test(self, probe_model, test_dataloader):
        self.set_trainer()
        probe_model.eval()
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            print("before testing")
            results = self.trainer.test(probe_model, test_dataloader)
            print("after testing")
        return results
    def load_pretrained_lm_weights(self, config, ckp_path):
        base_model = manual_train_fm(config=config)
        ckp = torch.load(ckp_path, map_location=device)
        params = ckp["state_dict"]
        base_model.load_state_dict(params)
        # base_model.eval()
        base_model.to(device)
        return base_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tuning parameters.')
    parser.add_argument('--fine_tune_mode', type=str, default='zero_shot',
                        help='The mode for fine-tuning the model. Default is "zero_shot". Optional "lora /"probe"/“zero_shot”/full_tune/')
    parser.add_argument('--radius', type=int, default='30',
                        help='The radius between query and key cells')
    
    args = parser.parse_args()
    sample_name = "VUILD110" #"THD0008"
    
    fine_tune_mode = args.fine_tune_mode #optional "fine_tune" /"probe"/ “zero_shot”/
    num_workers = 8
    batch_size = 8
    # radius_list = [10,20,30,50,80,100,120]
    radius_list = [args.radius]
    # radius_list = [30]
    # radius_list = [30] #this is only for cell type JSD evaluation
    all_results = {}
    # import pdb; pdb.set_trace()
    # model_ckp_path = "/scratch/project_465001820/Spatialformer/output/checkpoints/step=0100000-train_total_loss=-2.2727-val_total_loss=0.0000.ckpt"#single slides
    # model_ckp_path = "/scratch/project_465001820/Spatialformer/output/checkpoints/step=0044000-train_total_loss=-1.3226-val_total_loss=0.2488.ckpt"#all lung slides
    model_ckp_path = "/scratch/project_465001820/Spatialformer/output/checkpoints/step=0096000-train_total_loss=-2.9351-val_total_loss=0.0000.ckpt"#all 61 slides
    #loading the trained spatialformer as the base model
    config_path = "/scratch/project_465001820/Spatialformer/config/_config_fine_tune_probe.json"
    with open(config_path, 'r') as json_file:
        config = json.load(json_file) 
    total_steps = config["total_step"]
    
    for r in radius_list:
        
        #running the kfold validation
        if fine_tune_mode == "lora":

            test_size = 2000 #it will take effects only if split_mode = random, 
            kfold = 5
            split_mode = "leave_cell_out"

            Finetune = FineTune(config, config["model_ckp_path"], sample_name, r, fine_tune_mode, wandb = True)
            kfold_data_loader = data_prepare(sample_name, kfold, num_workers, batch_size, radius=r, test_size = test_size, split_mode = split_mode)
            for fold, (train_dataloader, test_dataloader, val_dataloader) in kfold_data_loader.items():
                print(f"running the fold {fold}")
                
                probe_model = Finetune.train(fold, train_dataloader, val_dataloader, config["model_ckp_path"])
                results = Finetune.test(probe_model, test_dataloader)
                all_results.setdefault(r, {}).setdefault(fold, results[0])
                wandb.finish()
        elif fine_tune_mode == "full_tune":
            kfold = 5
            test_size = 2000
            split_mode = "leave_cell_out"
            
            Finetune = FineTune(config, model_ckp_path, sample_name, r, fine_tune_mode, wandb = True)
            kfold_data_loader = data_prepare(sample_name, kfold, num_workers, batch_size, radius=r, test_size = test_size, split_mode = split_mode)
            for fold, (train_dataloader, test_dataloader, val_dataloader) in kfold_data_loader.items():
                print(f"running the fold {fold}")

                probe_model = Finetune.train(fold, train_dataloader, val_dataloader)
                results = Finetune.test(probe_model, test_dataloader)
                all_results.setdefault(r, {}).setdefault(fold, results[0])
                wandb.finish()



        elif fine_tune_mode == "probe":

            test_size = 2000 #it will take effects only if split_mode = random, 
            kfold = 5
            split_mode = "leave_cell_out"

            Finetune = FineTune(config, model_ckp_path, sample_name, r, fine_tune_mode, wandb = True)
            kfold_data_loader = data_prepare(sample_name, kfold, num_workers, batch_size, radius=r, test_size = test_size, split_mode = split_mode)
            for fold, (train_dataloader, test_dataloader, val_dataloader) in kfold_data_loader.items():
                print(f"running the fold {fold}")
                # import pdb; pdb.set_trace()
                probe_model = Finetune.train(fold, train_dataloader, val_dataloader)
                results = Finetune.test(probe_model, test_dataloader)
                all_results.setdefault(r, {}).setdefault(fold, results[0])
                wandb.finish()
                
        elif fine_tune_mode == "zero_shot":
            import signal
            zero_shot_cell_size = 500    
            kfold = 0
            test_dataloader = data_prepare(sample_name, kfold, num_workers, batch_size, radius=r, test_size = None, zero_shot_cell_size = zero_shot_cell_size, split_mode = "random")
            # import pdb; pdb.set_trace()
            Finetune = FineTune(config, model_ckp_path, sample_name, r, fine_tune_mode, strategy = "ddp" , wandb = True)
            # test_dataloader = data_prepare(sample_name, kfold, num_workers, batch_size, radius=r, test_size = None, zero_shot_cell_size = zero_shot_cell_size, split_mode = "random")
            probe_model = Finetune.probe_model
            results = Finetune.test(probe_model, test_dataloader)
            #tesing the model
            all_results[r] = results[0]
            #kill all the subprocess
            os.killpg(os.getpid(), signal.SIGTERM)  # Kill all child processes
    wandb.finish()
            

            
    #Save all the results
    if fine_tune_mode != "zero_shot":
        #getting all results
        data = []
        for r, folds in all_results.items():
            for fold, metrics in folds.items():
                metrics['radius'] = r  # Optional: If you want to keep track of which model produced the results
                metrics['fold'] = fold  # Include the fold number
                data.append(metrics)  # Append to the data list
        # Create the DataFrame
        df_results = pd.DataFrame(data)

        mean_values = df_results.groupby('radius').mean()
        print(df_results)
        print(mean_values)

        # Save to CSV file
        mean_values.to_csv(f'/scratch/project_465001820/Spatialformer/output/metrics/{sample_name}_{fine_tune_mode}_{total_steps}_mean_values_per_radius.csv', index=True)  # Include index if you want to keep fold numbers
        df_results.to_csv(f'/scratch/project_465001820/Spatialformer/output/metrics/{sample_name}_{fine_tune_mode}_{total_steps}_all_values.csv', index=True)

    else:
        #getting all results
        data = []
        for r, metrics in all_results.items():
            metrics['radius'] = r  # Optional: If you want to keep track of which model produced the results
            data.append(metrics)  # Append to the data list
        # Create the DataFrame
        df_results = pd.DataFrame(data)
        print(df_results)
        # Save to CSV file
        formatted_time = time.strftime('%Y%m%d_%H%M%S')
        df_results.to_csv(f'/scratch/project_465001820/Spatialformer/output/metrics/{sample_name}_{fine_tune_mode}_all_values_{formatted_time}.csv', index=True)


