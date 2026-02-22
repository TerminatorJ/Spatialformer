import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Recall, Specificity, AUROC, F1Score, Precision
import torch.nn as nn
import sys
sys.path.append("/scratch/project_465001820/Spatialformer/train")
sys.path.append("/scratch/project_465001820/Spatialformer/utils")
sys.path.append("/scratch/project_465001820/Spatialformer/spatialformer/model")
from data_loader import *
from utils import complete_masking, Lora
from Spaformer_pair import *
import torch.distributed as dist



# The Probe Model that extends the Base Model
class FTNetwork(pl.LightningModule):
    def __init__(self, base_model, fine_tune_mode, outer_config: dict = None, rank: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.save_hyperparameters(ignore=['base_model', 'outer_config'])
        
        # --- Model Setup (kept original logic) ---
        self.base_model = base_model
        if fine_tune_mode == "lora":
            self.lora = Lora(lora_config={"r": rank, "lora_alpha": lora_alpha, 
                                            "target_modules": ["qkv", "proj", "ffn1", "ffn2", "expand_conv", "conv_proj"],
                                            "modules_to_save": ["classifier_head", "pair_head", "MEloss", "Pairloss"]})
            self.base_model = self.lora.wrapper(self.base_model)
            self.base_model.token_type_embeddings.weight.requires_grad = False
            self.base_model.print_trainable_parameters()
        elif fine_tune_mode == "probe":
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("Only pair head trainable!!!")
            for param in self.base_model.pair_head.parameters():
                param.requires_grad = True
            self.base_model.token_type_embeddings.weight.requires_grad = False
        elif fine_tune_mode == "full_tune":
            for param in self.base_model.parameters():
                param.requires_grad = True
            self.base_model.token_type_embeddings.weight.requires_grad = False

        # --- Loss Setup ---
        self.MEloss = MaskedMSELoss(n_token=outer_config["n_tokens"]+outer_config["n_atokens"], 
                                    cls_token=outer_config["cls_token"], sep_token=outer_config["sep_token"])
        self.Pairloss = PairLoss()
        self.lr = outer_config["lr"]

        # --- Metric Setup ---
        # Using ModuleDict ensures Lightning manages devices and resets automatically
        def create_metrics():
            return nn.ModuleDict({
                'accuracy': Accuracy(task='binary'),
                'recall': Recall(task='binary'),
                'specificity': Specificity(task='binary'),
                'auroc': AUROC(task='binary'),
                'f1': F1Score(num_classes=2, task='binary'),
                'prec': Precision(num_classes=2, task='binary')
            })

        self.train_metrics = create_metrics()
        self.val_metrics = create_metrics()
        
        self.fine_tune_mode = fine_tune_mode
        self.log_sigma_pair = nn.Parameter(torch.zeros(1))
        self.log_sigma_mlm = nn.Parameter(torch.zeros(1)) 
        self.log_sigma_spa = nn.Parameter(torch.zeros(1)) 
        self.outer_config = outer_config

    @classmethod
    def load_from_checkpoint_with_merge(cls, checkpoint_path, base_model, fine_tune_mode, outer_config, 
                                        rank=8, lora_alpha=16, device='cpu', strict=False):
        """
        Load a Lightning checkpoint with LoRA weights and merge them into the base model.
        """
        # Load checkpoint
        ckp = torch.load(checkpoint_path, map_location=device)
        state_dict = ckp["state_dict"]
        
        # Create a temporary LoRA model to load the checkpoint
        temp_model = cls(base_model, fine_tune_mode="lora", outer_config=outer_config, 
                        rank=rank, lora_alpha=lora_alpha)
        
        # Fix the state dict to match the structure
        fixed_state_dict = cls._fix_state_dict_for_loading(state_dict, temp_model.state_dict())
        
        # Load state dict into temporary LoRA model
        missing, unexpected = temp_model.load_state_dict(fixed_state_dict, strict=strict)
        
        if missing:
            print(f"Missing keys during load: {missing[:10]}...")  # Show first 10
        if unexpected:
            print(f"Unexpected keys during load: {unexpected[:10]}...")
        
        # Merge LoRA weights
        if fine_tune_mode == "lora":
            print("Merging LoRA weights into base model...")
            temp_model.base_model = temp_model.base_model.merge_and_unload()
        
        # Move to device
        temp_model.to(device)
        temp_model.eval()
        
        return temp_model
    @staticmethod
    def _fix_state_dict_for_loading(loaded_state_dict, model_state_dict):
        """
        Fix state dict key mismatches between checkpoint and model.
        Handles modules_to_save that were saved with .weight/.bias but model expects .original_module/.modules_to_save
        """
        fixed_state_dict = {}
        model_keys = set(model_state_dict.keys())
        
        # Define modules that should be in modules_to_save structure
        modules_to_save_patterns = ["classifier_head", "pair_head", "MEloss", "Pairloss"]
        
        for key, value in loaded_state_dict.items():
            # Check if this key is a simple weight that should be in modules_to_save
            is_modules_to_save = any(pattern in key for pattern in modules_to_save_patterns)
            
            if is_modules_to_save:
                # Check if the key has the simple structure
                if ".original_module." not in key and ".modules_to_save." not in key:
                    # This is a simple key like "base_model.base_model.model.classifier_head.weight"
                    # We need to map it to both original_module and modules_to_save.default
                    
                    # Extract the parts
                    parts = key.rsplit('.', 1)  # Split on last dot
                    if len(parts) == 2:
                        base_key, param_name = parts  # e.g., "base_model...classifier_head", "weight"
                        
                        # Create both keys
                        original_key = f"{base_key}.original_module.{param_name}"
                        saved_key = f"{base_key}.modules_to_save.default.{param_name}"
                        
                        # Add both (modules_to_save.default takes precedence during merge)
                        if original_key in model_keys:
                            fixed_state_dict[original_key] = value
                        if saved_key in model_keys:
                            fixed_state_dict[saved_key] = value
                        
                        continue
            
            # For all other keys, keep as is
            fixed_state_dict[key] = value
        
        return fixed_state_dict

    def forward(self, masked_indices, attention_mask, token_type_ids, sequence_length):   
        predictions = self.base_model(masked_indices, False, attention_mask, token_type_ids, sequence_length)
        return predictions

    # --- Training Step ---
    def training_step(self, batch, batch_idx):
        # 1. Compute Loss and update metrics
        output = self._compute_and_update(batch, "train")
        
        # 2. Log Losses (Values)
        self.log("train_total_loss", output["total_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_pair_loss", output["pair_loss"], on_step=True, on_epoch=True)
        self.log("train_mlm_loss", output["mlm_loss"], on_step=True, on_epoch=True)

        # 3. Log Metrics (Objects)
        # We iterate over the metrics and log the OBJECT, not the value.
        for name, metric in self.train_metrics.items():
            self.log(f"train_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)
            
        return output["total_loss"]

    # --- Validation Step ---
    def validation_step(self, batch, batch_idx):
        # 1. Compute Loss and update metrics
        output = self._compute_and_update(batch, "val")

        # 2. Log Losses (Values)
        self.log("val_total_loss", output["total_loss"], on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_pair_loss", output["pair_loss"], on_epoch=True, sync_dist=True)

        # 3. Log Metrics (Objects)
        for name, metric in self.val_metrics.items():
            self.log(f"val_{name}", metric, on_epoch=True, sync_dist=True, prog_bar=True)
            
        return output["total_loss"]


    def _compute_output(self, batch):
        # ... (Keep existing logic exactly as provided in your snippet) ...
        # Ensure batch items are on device, compute predictions, losses
        batch = complete_masking(batch, self.outer_config["masking_p"], self.outer_config["n_tokens"], 
                                 self.outer_config["cls_token"], self.outer_config["mask_token"], 
                                 self.outer_config["sep_token"], self.outer_config["pad_token"])
        
        masked_indices = batch['masked_indices']
        real_indices = batch['indices']
        attention_mask = batch['attention_mask']
        token_type_ids = batch["token_type_ids"]  
        pair_label = batch["pair_label"] 
        mask = batch['mask']   
        sequence_length = batch["sequence_length"]

        predictions = self(masked_indices, attention_mask, token_type_ids, sequence_length)
        mlm_predictions = predictions['mlm_prediction']
        
        real_indices = torch.where(mask==self.outer_config["mask_token"], real_indices, torch.tensor(-100, dtype=torch.long, device=self.device)).type(torch.int64)
        mlm_predictions = mlm_predictions.view(-1, self.outer_config["n_tokens"]+self.outer_config["n_atokens"])
        real_indices = real_indices.view(-1)
        
        MLM_loss = self.MEloss(mlm_predictions, real_indices)
        
        pair_predictions = predictions['pair_prediction']
        pair_logit = torch.argmax(pair_predictions, dim=1)
        pair_probs = torch.softmax(pair_predictions, dim=1)
        pair_positive_probs = pair_probs[:, 1]
        Pair_loss = self.Pairloss(pair_predictions, pair_label)

        total_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        total_loss += (torch.exp(-self.log_sigma_mlm[0]) * MLM_loss + self.log_sigma_mlm[0])
        total_loss += (torch.exp(-self.log_sigma_pair[0]) * Pair_loss + self.log_sigma_pair[0])

        output = {"mlm_loss": MLM_loss, "pair_loss": Pair_loss, "spa_loss": 0, "total_loss": total_loss, 
                  "pair_logit": pair_logit, "pair_positive_probs": pair_positive_probs}
        return output

    def _compute_and_update(self, batch, split) -> dict:
        """
        Helper that computes output and updates metric states.
        Does NOT return computed metric values.
        """
        # Move inputs to device
        batch['indices'] = batch['indices'].to(self.device)
        batch["pair_label"] = batch["pair_label"].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(self.device)
        labels = batch["pair_label"]

        # Get model outputs and loss
        output = self._compute_output(batch)
        
        preds = output["pair_logit"]
        pair_positive_probs = output["pair_positive_probs"]

        # UPDATE METRICS INTERNAL STATE
        # Do not call .compute() here!
        metric_set = self.train_metrics if split == "train" else self.val_metrics

        metric_set['accuracy'].update(preds, labels)
        metric_set['recall'].update(preds, labels)
        metric_set['specificity'].update(preds, labels)
        metric_set['auroc'].update(pair_positive_probs, labels)
        metric_set['f1'].update(preds, labels)
        metric_set['prec'].update(preds, labels)

        # Return only the output/loss dict
        return output

    def test_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        # print("start test step")

        batch['indices'] = batch['indices'].to(self.device)
        batch["pair_label"] = batch["pair_label"].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(self.device)
        batch["left_index"] = batch["left_index"].to(self.device)
        batch["right_index"] = batch["right_index"].to(self.device)
        batch["sequence_length"] = batch["sequence_length"].to(self.device)
        labels = batch["pair_label"]
        left_indexs = batch["left_index"]
        right_indexs = batch["right_index"]
        # print("left_indexs:", left_indexs)
        # print("right_indexs:", right_indexs)

        last_hidden_repr, pair_prob = self.base_model.get_embeddings(batch, [-1], True, False) #
        # import pdb; pdb.set_trace()

        preds = torch.argmax(pair_prob, dim=1) #get pred from the original model
        positive_probs = pair_prob #get prob from the original model

        # Calculate the loss (assuming it’s a binary classification task)
        # loss = F.cross_entropy(outputs, labels)
        # Update metrics
        # print("end test step")

        return preds, positive_probs, labels, left_indexs, right_indexs
    def test_epoch_end(self, outputs):

        #we support multi-gpus for inferencing
        rank = torch.distributed.get_rank() 
        results = torch.zeros((2, 2)).to(self.device)
        preds, labels, probs, left_indexs, right_indexs = [], [], [], [], []
        # for output in outputs:
        print("outputs", outputs)
        print("before reduce:", results)
        for output in outputs:
            print("output:", output)
            for pred, positive_prob, label, left_index, right_index in zip(*output):

                # for pred, label in zip(output):
                results[int(label), int(pred)] += 1
                preds.append(pred)
                labels.append(label)
                probs.append(positive_prob)
                left_indexs.append(left_index)
                right_indexs.append(right_index)
        # print("left_indices before aggregate:", left_indexs)

        # Create tensors from lists
        preds_tensor = torch.tensor(preds).to(self.device)
        labels_tensor = torch.tensor(labels).to(self.device)
        left_indexs_tensor = torch.tensor(left_indexs).to(self.device)
        right_indexs_tensor = torch.tensor(right_indexs).to(self.device)
        # print("left_indexs_tensor  before aggregate:", left_indexs_tensor)
        #gather to rank0
        # Gather the results on rank 0
        print("torch.distributed.get_world_size()", torch.distributed.get_world_size())
        # This assumes `preds_tensor` is your output tensor to gather and that the rank is set up correctly
        
        # Initialize gather_list only on the destination rank
        if rank == 0:
            gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(torch.distributed.get_world_size())]
            gathered_left_indexs = [torch.zeros_like(left_indexs_tensor) for _ in range(torch.distributed.get_world_size())]
            gathered_right_indexs = [torch.zeros_like(right_indexs_tensor) for _ in range(torch.distributed.get_world_size())]
            # print("gathered_right_indexs:", gathered_right_indexs)
        else:
            gathered_preds = None
            gathered_labels = None
            gathered_left_indexs = None
            gathered_right_indexs = None
        print(f"Rank {rank}: preds_tensor shape: {preds_tensor.shape}")
        dist.barrier()  # Ensure all ranks reach this point before continuing
        # Gather operations
        try:
            print(f"Rank {rank}: Gather operation starting.")
            
            torch.distributed.gather(preds_tensor, gather_list=gathered_preds, dst=0)
            torch.distributed.gather(labels_tensor, gather_list=gathered_labels, dst=0)
            torch.distributed.gather(left_indexs_tensor, gather_list=gathered_left_indexs, dst=0)
            torch.distributed.gather(right_indexs_tensor, gather_list=gathered_right_indexs, dst=0)

            print(f"Rank {rank}: Gather operation completed.")
        except Exception as e:
            print(f"Error during gather on rank {rank}: {e}")


        # Reduce results
        try:
            torch.distributed.reduce(results, 0, torch.distributed.ReduceOp.SUM)
        except Exception as e:
            print(f"Error during reduce on rank {rank}: {e}")
        # print("before reduce:", results)
        # print("after reduce:", results)
        acc = results.diag().sum() / results.sum()
        print("gathered_left_indexs after combination:", gathered_left_indexs)
        print("gathered_right_indexs after combination:", gathered_right_indexs)
            

        if self.trainer.is_global_zero:
            # Calculate accuracy
            acc = results.diag().sum() / results.sum()
            # preds_tensor
            # Concatenate results from all ranks
            all_preds = torch.cat(gathered_preds)
            all_labels = torch.cat(gathered_labels)
            print("after aggregation:", all_preds)
            all_left_indexs = torch.cat(gathered_left_indexs)
            all_right_indexs = torch.cat(gathered_right_indexs)
            
            #save the predicted results and the index
            import pickle
            import time
            from sklearn.metrics import roc_auc_score
            formatted_time = time.strftime('%Y%m%d_%H%M%S')
            pickle.dump(all_preds.cpu().numpy(), open(f"/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/all_preds_{formatted_time}.pkl", "wb"))
            pickle.dump(all_labels.cpu().numpy(), open(f"/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/all_labels_{formatted_time}.pkl", "wb"))
            pickle.dump(all_left_indexs.cpu().numpy(), open(f"/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/all_left_indexs_{formatted_time}.pkl", "wb"))
            pickle.dump(all_right_indexs.cpu().numpy(), open(f"/scratch/project_465001820/Spatialformer/downstream/cell_cell_communication/data/all_right_indexs_{formatted_time}.pkl", "wb"))

            # Extract confusion matrix components
            TP = results[1, 1]
            TN = results[0, 0]
            FP = results[0, 1]
            FN = results[1, 0]
        
            # Calculate metrics
            recall = TP / (TP + FN) if (TP + FN) > 0 else torch.tensor(0.)
            specificity = TN / (TN + FP) if (TN + FP) > 0 else torch.tensor(0.)
            precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor(0.)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.)
            auc_score = roc_auc_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
            # Log accuracy, recall, specificity, F1
            self.log("test_accuracy", acc, rank_zero_only=True, on_epoch=True)
            self.log("test_recall", recall, rank_zero_only=True, on_epoch=True)
            self.log("test_precision", precision, rank_zero_only=True, on_epoch=True)
            self.log("test_specificity", specificity, rank_zero_only=True, on_epoch=True)
            self.log("test_f1", f1, rank_zero_only=True, on_epoch=True)
            self.log("test_auc", auc_score, rank_zero_only=True, on_epoch=True)
            # print(acc, recall, specificity, f1, precision)
            # Optionally set results for further usage
            self.trainer.results = results
        # Clean up
        dist.destroy_process_group()
        del outputs  # Clear the output from memory


        
    def configure_optimizers(self):
        
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.outer_config["warmup"],
                                             max_epochs=self.outer_config["max_epochs"])
        
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]




