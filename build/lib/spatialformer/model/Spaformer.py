import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from typing import List
from torch import optim

import numpy as np
import math
import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]
model_dir = os.path.join(p_path, "model")
util_dir = os.path.join(p_path, "utils")
sys.path.append(util_dir)
sys.path.append(model_dir)
from utils import complete_masking, categorical_2d_masking
from model import *
# import wandb
import pickle
MASK_TOKEN = 2
CLS_TOKEN = 1
PAD_TOKEN = 0

class MaskedMSELoss(nn.Module):
    def __init__(self, mask_way = "MT", n_token = None):
        '''
        mask_way: MT means mask the token. ME means mask the expression level
        
        '''
        super(MaskedMSELoss, self).__init__()
        self.mask_way = mask_way
        self.n_token = n_token
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor, exp: torch.Tensor) -> torch.Tensor:
        # Here, the mask represents the masked tokens (with value != -100)
        # import pdb; pdb.set_trace()
        mask = target != -100  # Boolean mask where True indicates valid (non-masked) positions
        # Filter target and input tensors using the mask
        valid_target = target[mask]
        valid_exp = exp[mask]
        valid_input = input[mask]  # Shape (valid_batch, dim)
        # Ensure valid_target is in the correct shape for gather
        valid_target = valid_target.unsqueeze(1)  # Shape (valid_batch, 1)



        if self.mask_way == "ME":
            # Use gather to select the relevant predictions based on valid_target
            preds = valid_input.gather(1, valid_target).squeeze(1)  # Shape (valid_batch,)
            # Calculate the loss only for the masked elements
            loss = F.mse_loss(preds, valid_exp.float(), reduction='sum')
            # Normalize by the number of valid elements
            mask_num = mask.sum().float()
            loss = loss / mask_num
        elif self.mask_way == "MT":
            # import pdb; pdb.set_trace()
            # preds = valid_input[valid_target]  # Shape (valid_batch,)

            loss = self.loss(input, target)

        

        return loss
    
class MaskedMSE2DLoss(nn.Module):
    def __init__(self, sample_balance = False):
        super(MaskedMSE2DLoss, self).__init__()
        self.sample_balance = sample_balance

    def even_sample(self, pred, target):
        # Flatten matrices for easier indexing
        # import pdb; pdb.set_trace()
        predictions_flat = pred.view(-1)
        true_labels_flat = target.view(-1)
        
        # Identify positive and negative indices
        positive_indices = (true_labels_flat == 1).nonzero(as_tuple=True)[0]
        negative_indices = (true_labels_flat == 0).nonzero(as_tuple=True)[0]
        
        # Sample indices
        num_samples = min(len(positive_indices), len(negative_indices))
        selected_pos_indices = torch.randperm(len(positive_indices))[:num_samples]
        selected_neg_indices = torch.randperm(len(negative_indices))[:num_samples]

        
        sampled_indices = torch.cat((positive_indices[selected_pos_indices], negative_indices[selected_neg_indices]))
        
        # Calculate loss using sampled indices
        sampled_predictions = predictions_flat[sampled_indices]
        sampled_true_labels = true_labels_flat[sampled_indices]
        return sampled_predictions, sampled_true_labels


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Here, the mask represents the masked tokens (with value != -100)
        mask = target != -100  # 2D Boolean mask where True indicates valid (non-masked) positions
        # Filter target and input tensors using the mask
        valid_target = target*mask
        valid_input = input*mask  # Shape (valid_batch, valid_batch)
        # Ensure valid_target is in the correct shape for gather
        if self.sample_balance:
            valid_input, valid_target = self.even_sample(valid_input, valid_target)
            mask_num = len(valid_target)
            loss = F.binary_cross_entropy_with_logits(valid_input, valid_target.float(), reduction='sum')
            # import pdb; pdb.set_trace()
        else:
            loss = nn.CrossEntropyLoss(valid_input, valid_target.float(), reduction='sum')
            # Normalize by the number of valid elements
            mask_num = mask.sum().float()
        return loss / mask_num

    
class AdjacencyProjector(nn.Module):
    def __init__(self, embedding_dim):
        super(AdjacencyProjector, self).__init__()
        # Assume output dimension of 1 because we want a single score
        self.fc = nn.Linear(2 * embedding_dim, 1, bias=False)

    def forward(self, E):
        # import pdb; pdb.set_trace()
        #B x L X dim
        num_genes = E.size(1)
        # Create pairwise combinations
        E_i = E.unsqueeze(2).expand(-1, -1, num_genes, -1)##B X L X L X dim
        E_j = E.unsqueeze(1).expand(-1, num_genes, -1, -1)##B X L X L X dim
        pairs = torch.cat((E_i, E_j), dim=-1)
        
        # Compute adjacency scores
        adjacency_scores = self.fc(pairs).squeeze(-1)#B x L X L
        return adjacency_scores


class GraphSAGESpatialEmbedding(nn.Module):
    def __init__(self, freeze):
        super().__init__()
        pretrained_weights = pickle.load(open("../spatial_embeddings/gene_embeddings_GraphSAGE.pkl", "rb"))
        self.emb = nn.Embedding.from_pretrained(pretrained_weights, freeze=freeze)
        print("require grad:", self.emb.weight.requires_grad)

    def forward(self, x):
        # self.emb(x)
        # import pdb; pdb.set_trace()
        # t = torch.arange(x.shape[1], device=x.device)
        return self.emb(x)



class Spaformer(pl.LightningModule):
    
    def __init__(self, 
                 dim_model: int, 
                 nheads: int, 
                 nlayers: int, 
                 dropout: float,
                 masking_p: float, 
                 n_tokens: int,
                 n_atokens: int,
                 context_length: int,
                 lr: float, 
                 warmup: int, 
                 max_epochs: int,
                 pool: str = None,
                 learnable_pe: bool = True,
                 specie: bool = False,
                 assay: bool = False,
                 modality: bool = False,
                 bpp: bool = True,
                 bpp_scale: int = None,
                 ag_loss: bool = False,
                 mask_way: str = None,
                 outer_config: dict = None,

                 ):
        """
        Args:
            dim_model (int): Dimensionality of the model
            nheads (int): Number of attention heads
            masking_p (float): p value of Bernoulli for masking
            n_tokens (int): total number of tokens (WITHOUT auxiliar tokens), only the gene indices
            n_atokens (int): total number of auxiliar tokens
            context_length (int): length of the context, which means the fixed number of the input sequence
            lr (float): learning rate
            warmup (int): number of steps that the warmup takes
            max_epochs (int): number of steps until the learning rate reaches 0
            pool (str): could be None, 'cls' or 'mean'. CLS adds a token at the beginning, mean just averages all tokens. If not supervised task during training, is ignored
            learnable_pe (bool): if True, positional embeddings are learnable embeddings, otherwise are derived from trigonometric functions
            specie (bool): if True, add a token to identify the specie of the observation (human or mouse)
            assay (bool): if True, add a token to identify the assay of the observations 
            modality (bool): if True, add a token to identify the modality of the observations (spatial or dissociated)
            outer_config (dict): the overall config of the model for training
        """
        super().__init__()
        # import pdb; pdb.set_trace()
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nheads, dim_feedforward=dim_feedforward, batch_first=batch_first, dropout=dropout, layer_norm_eps=1e-12)
        # self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=nlayers, enable_nested_tensor=False)
        self.encoder = SpaEncoder(dim=dim_model , num_layers=nlayers, groups=dim_model, num_heads=nheads, bpp_size = context_length, bpp = bpp, bpp_scale=bpp_scale)
        # As in HuggingFace
        # The prediction head for each masked token
        self.classifier_head = nn.Linear(dim_model, n_tokens+n_atokens, bias=False)
        self.adjprojector = AdjacencyProjector(dim_model)

        
        bias = nn.Parameter(torch.zeros(n_tokens+n_atokens)) # each token has its own bias
        self.classifier_head.bias =  bias
            
        # As in HuggingFace
        # self.pooler_head = nn.Linear(dim_model, dim_model)
        self.activation = nn.Tanh()

        self.embeddings = nn.Embedding(num_embeddings=n_tokens+n_atokens, embedding_dim=dim_model, padding_idx=0)

        #loading the pretrained embeddings
        # import pdb; pdb.set_trace()
        if outer_config["spatial_embedding"]:
            self.spatialembeds = GraphSAGESpatialEmbedding(outer_config["spatial_embedding_freeze"])
       
           


        
        if pool == 'cls':
            context_length += 1
            
        # if not learnable_pe:
        #     self.positional_embedding = PositionalEncoding(d_model=dim_model, max_seq_len=context_length)
        # else:
        #     # uses learnable weights as positional embeddings
        #     self.positional_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=dim_model) 
        #     self.dropout = nn.Dropout(p=dropout)
        #     self.pos = torch.arange(0, context_length, dtype=torch.long)
        
        # MLM loss
        # self.loss = nn.CrossEntropyLoss()
        self.MEloss = MaskedMSELoss(mask_way = mask_way, n_token = n_tokens+n_atokens)#masked expression level loss
        self.MDloss = MaskedMSE2DLoss(sample_balance = True)
            
        self.save_hyperparameters()
        #initialize as 0 to get better proformance, to homoscedastic uncertainty
        self.log_sigma_class = nn.Parameter(torch.zeros(1))  # Log variance for classification
        self.log_sigma_reg1 = None    # Log variance for regression
        self.log_sigma_reg2 = None
        self.log_sigma_reg3 = None
        

        self.gc_freq = 5
        
        self.batch_train_losses = []
        
        # self.initialize_weights()
        self.batch_input = {}
        self.total_tokens = 0
        self.adjmtx = None
        self.attentions = None
        self.nheads = nheads
        self.outer_config = outer_config
        self.ag_loss = ag_loss
        self.setup_sigma()
        


            
    def forward(self, x, adjmtx, attention_mask, **kwargs):
                
        # x -> size: batch x (context_length) x 1
        # import pdb; pdb.set_trace()
        self.adjmtx = adjmtx
        token_embedding = self.embeddings(x) # batch x (context_length) x dim_model
        if self.outer_config["spatial_embedding"]:
            # import pdb; pdb.set_trace()
            token_embedding += self.spatialembeds(x)
        transformer_output, attn_scores = self.encoder(token_embedding, adjmtx, attention_mask) # batch x (n_tokens) x dim_model
        # 
        # MLM prediction
        
        #get the last layer of the model output
        prediction = self.classifier_head(transformer_output[-1])
        #get the pair-wise gene-gene interaction matrix
        dis_prediction = self.adjprojector(token_embedding)
        self.attentions = attn_scores
        # import pdb; pdb.set_trace()
        return {'mlm_prediction': prediction,
                'transformer_output': transformer_output,
                'attention_score': attn_scores,
                'spa_prediction': dis_prediction}
    
    def compute_ag_loss(self, lamda, **kwargs):
        """
        Adds a random loss based on attention values
        To test gradients
        outputs[-1] contains the attention values (tuple of size num_layers)
        and each element is of the shape [batch_size X num_heads X max_sequence_len X max_sequence_len]
        """
        # Matrices containing the attention patterns come from the adjmtx
        targets = self.adjmtx  # B X L X L

        # Ensure adjmtx has the correct shape
        if targets.dim() != 3:
            raise ValueError(f"Expected adjmtx to have 3 dimensions [B, L, L], but got {targets.dim()} dimensions.")
        # import pdb; pdb.set_trace()
        # Initialize the loss function and total loss variable
        loss_fn = nn.MSELoss()
        total_loss = 0.0

        # Calculate how many heads will be used based on the lambda
        apply_head_num = int(self.nheads * lamda)

        # Expand targets to match the shape of the attention heads
        targets = torch.unsqueeze(targets, 1)  # B X 1 X L X L 
        targets = targets.repeat(1, apply_head_num, 1, 1)  # B X H*lambda X L X L
        targets = targets.to(self.device)

        # Iterate over the layers of attention
        for attention in self.attentions:
            # Ensure attention tensor has the expected shape
            if attention.shape[1] < apply_head_num:
                raise ValueError(f"Expected at least {apply_head_num} heads, but got {attention.shape[1]} heads.")
            # import pdb; pdb.set_trace()
            # Calculate the loss for the selected heads in this layer
            layer_loss = loss_fn(targets, attention[:, :apply_head_num, :, :])
            total_loss += layer_loss

        return total_loss
    
    def dynamic_weight_average(self, losses, T):
        '''
        implemention of Dynamic Weight Average (DWA)
        '''
    
        n_tasks = len(losses)
        weight = torch.zeros(n_tasks).to(losses[0].device)

        for i in range(n_tasks):
            weight[i] = torch.exp(losses[i] / T)
        
        normalized_weights = weight / weight.sum()
        return normalized_weights
    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        # Training code
        if self.outer_config["mini_batch"]:
            mini_batch_list = self.mini_batch_setup(batch)    
            # import pdb; pdb.set_trace()
            total_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            target_dict = {0: "normalized_exp", 1: "nuc_exp", 2: "cyto_exp"}
            for i, exp_batch in enumerate(mini_batch_list[:-1]): #make sure the last one is the spatial mini batch
                # import pdb; pdb.set_trace()
                mlm_predictions, target, real_indices, attention_mask = self.predict_exp(exp_batch, target_dict[i]) #different task means different targets here
                MLM_loss = self.MEloss(mlm_predictions, real_indices, target) # MLM loss by MSE
                if self.outer_config["weight_strategy"] == "DWA":
                    
                    self.log(f'train_MLM_loss({target_dict[i]})', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                elif self.outer_config["weight_strategy"] == "UW":
                    self.log(f'train_MLM_loss({target_dict[i]})', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    log_sigma = getattr(self, f'log_sigma_reg{i+1}')[0]
                    self.log(f'log_sigma_reg({target_dict[i]}))', log_sigma, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    # self.log(f'val_MLM_loss({target_dict[i]})', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    total_loss += (torch.exp(-log_sigma) * MLM_loss + log_sigma)
            

            # import pdb; pdb.set_trace()
            spa_batch = mini_batch_list[-1]
            spa_predictions, adjmtx = self.predict_spa(spa_batch)
            Spa_loss = self.MDloss(spa_predictions, adjmtx)# spatial loss
            if self.outer_config["weight_strategy"] == "DWA":
                self.log('train_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                # Calculate dynamic weights
                weights = self.dynamic_weight_average([MLM_loss, Spa_loss], 2)

                # Weighted multi-task loss
                total_loss = weights[0] * MLM_loss + weights[1] * Spa_loss
                self.log('train_total_loss', total_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
            elif self.outer_config["weight_strategy"] == "UW":
                self.log('train_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                total_loss += (torch.exp(-self.log_sigma_class[0]) * Spa_loss * self.linear_schedule_for_scale() + self.log_sigma_class[0])
                self.log('log_sigma_class', self.log_sigma_class[0], sync_dist=True, prog_bar=True, reduce_fx='mean')
                self.log('train_total_loss', total_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')

        else:
            total_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            if self.outer_config["objective"] == "normalized_exp":
                batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens) #for mask expression prediction
                mlm_predictions, target, real_indices, attention_mask = self.predict_exp(batch, "normalized_exp") 
                MLM_loss = self.MEloss(mlm_predictions, real_indices, target) # MLM loss
                self.log('train_MLM_loss(normalized_exp)', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                self.log('train_total_loss', MLM_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
                total_loss += MLM_loss
            elif self.outer_config["objective"] == "spatial":
                spa_predictions, adjmtx = self.predict_spa(batch)
                Spa_loss = self.MDloss(spa_predictions, adjmtx)# spatial loss
                attention_mask = batch['attention_mask']
                self.log('train_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                self.log('train_total_loss', Spa_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
                total_loss += Spa_loss
        # There's a corner case that returns NaN loss: when there are no masked tokens
        # however, likelihood of that is (1-p)^context_length

        #getting the total tokens

        self.total_tokens += attention_mask.sum()
        self.log('total_tokens', self.total_tokens, prog_bar=True)

        mem_after = torch.cuda.memory_allocated()
        print(f"GPU memory usage: {mem_before/1e9} GB -> {mem_after/1e9} GB")               
       
        return total_loss
    def linear_schedule_for_scale(self, num_stagnant_steps=100):
        """
        Linear schedule for scale, the relative weight assigned
        to ag_loss
        """
        tot = self.outer_config["total_step"]
        cur = self.global_step
        if cur < num_stagnant_steps:
            return 1.0
        else:
            return max(0.0, float(tot - cur) / float(max(1, tot - num_stagnant_steps))) 

    def mini_batch_setup(self, batch):
        '''
        split the whole batch into two parts. one for masked expression level prediction (can also be split into cytoplasm & nucleus). one for co-occurrence prediction
        '''     
        div = int(self.outer_config["n_tasks"])
        # import pdb; pdb.set_trace()
        batch_size = batch["indices"].shape[0]
        print(f"Batch size = {batch_size}")
        
        # assert batch_size%div == 0, f"For minibatch implementation, the batch size should be set to {div}*n"
        mini_size = int(batch_size/div)
        print(f"mini batch size = {mini_size}")
        mini_batch_list = []
        #only apply for exp batch
        for mini_batch in range(div-1): #because one left for spatial
            # import pdb; pdb.set_trace()
            exp_batch = {k: v[int(mini_batch*mini_size):int((mini_batch+1)*mini_size)] for k, v in batch.items()}
            # import pdb; pdb.set_trace()
            exp_batch = complete_masking(exp_batch, self.hparams.masking_p, self.hparams.n_tokens) #for mask expression prediction
            mini_batch_list.append(exp_batch)
            # import pdb; pdb.set_trace()
        
        spa_batch = {k: v[-mini_size:] for k, v in batch.items()}
        mini_batch_list.append(spa_batch)
        # import pdb; pdb.set_trace()
        
        return mini_batch_list
    
    def predict_exp(self, batch, target = "normalized_exp"):
        #in case the mask is none
        with torch.no_grad():
            real_indices = batch['indices']
            mask = batch['mask']
            no_mask = torch.all(torch.where(real_indices != PAD_TOKEN, mask, 1) == 1)
            while no_mask:
                batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)
                real_indices = batch['indices']
                mask = batch['mask']
                no_mask = torch.all(torch.where(real_indices != PAD_TOKEN, mask, 1) == 1)
        # import pdb; pdb.set_trace()
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']
        norm_exp = batch[target]       
        adjmtx = batch['adjmtx']
        predictions = self.forward(masked_indices, adjmtx, attention_mask)
        mlm_predictions = predictions['mlm_prediction']
        # spa_predictions = predictions['spa_prediction']#
        # import pdb; pdb.set_trace()
        real_indices = torch.where(mask==MASK_TOKEN, real_indices, torch.tensor(-100, dtype=torch.long)).type(torch.int64)
        mlm_predictions = mlm_predictions.view(-1, self.hparams.n_tokens+self.hparams.n_atokens)
        # spa_predictions = spa_predictions.view(-1, self.hparams.n_tokens+self.hparams.n_atokens, self.hparams.n_tokens+self.hparams.n_atokens)
        real_indices = real_indices.view(-1)
        norm_exp = norm_exp.view(-1)
        return mlm_predictions, norm_exp, real_indices, attention_mask
    def predict_spa(self, batch):
        adjmtx = batch['adjmtx']
        indices = batch['indices']
        attention_mask = batch['attention_mask']
        predictions = self.forward(indices, adjmtx, attention_mask)
        spa_predictions = predictions['spa_prediction']
        return spa_predictions, adjmtx
    
    def setup_sigma(self):
        # import pdb; pdb.set_trace()
        n_tasks = self.outer_config["n_tasks"]
        for i in range(1, n_tasks):
            setattr(self, f'log_sigma_reg{i}', nn.Parameter(torch.zeros(1)))


    
    

    
    def validation_step(self, batch, batch_idx, *args, **kwargs):    

        if self.outer_config["mini_batch"]:
            # import pdb; pdb.set_trace()
            mini_batch_list = self.mini_batch_setup(batch)    
            # import pdb; pdb.set_trace()
            total_loss = torch.tensor(0, dtype=torch.float, device=self.device)

            target_dict = {0: "normalized_exp", 1: "nuc_exp", 2: "cyto_exp"}
            for i, exp_batch in enumerate(mini_batch_list[:-1]): #make sure the last one is the spatial mini batch
                # import pdb; pdb.set_trace()
                mlm_predictions, target, real_indices, attention_mask = self.predict_exp(exp_batch, target_dict[i]) #different task means different targets here
                MLM_loss = self.MEloss(mlm_predictions, real_indices, target) # MLM loss
                if self.outer_config["weight_strategy"] == "DWA":
                    self.log(f'val_MLM_loss({target_dict[i]})', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                elif self.outer_config["weight_strategy"] == "UW":
                    self.log(f'val_MLM_loss({target_dict[i]})', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    log_sigma = getattr(self, f'log_sigma_reg{i+1}')[0]
                    # self.log(f'val_MLM_loss({target_dict[i]})', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    total_loss += (torch.exp(-log_sigma) * MLM_loss + log_sigma)
            

            # import pdb; pdb.set_trace()
            spa_batch = mini_batch_list[-1]
            spa_predictions, adjmtx = self.predict_spa(spa_batch)
            Spa_loss = self.MDloss(spa_predictions, adjmtx)# spatial loss
            if self.outer_config["weight_strategy"] == "DWA":
                self.log('val_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                # Calculate dynamic weights
                weights = self.dynamic_weight_average([MLM_loss, Spa_loss], 2)
                # Weighted multi-task loss
                total_loss = weights[0] * MLM_loss + weights[1] * Spa_loss
                self.log('val_total_loss', total_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
            elif self.outer_config["weight_strategy"] == "UW":
                self.log('val_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                total_loss += (torch.exp(-self.log_sigma_class[0]) * Spa_loss * self.linear_schedule_for_scale() + self.log_sigma_class[0])
                self.log('val_total_loss', total_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')

        else:
            total_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            if self.outer_config["objective"] == "normalized_exp":
                batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens) #for mask expression prediction
                # import pdb; pdb.set_trace()
                mlm_predictions, target, real_indices, attention_mask = self.predict_exp(batch, "normalized_exp") 
                MLM_loss = self.MEloss(mlm_predictions, real_indices, target) # MLM loss
                self.log('val_MLM_loss(normalized_exp)', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                self.log('val_total_loss', MLM_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')

                total_loss += MLM_loss
            elif self.outer_config["objective"] == "spatial":
                spa_predictions, adjmtx = self.predict_spa(batch)
                Spa_loss = self.MDloss(spa_predictions, adjmtx)# spatial loss
                self.log('val_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                self.log('val_total_loss', Spa_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
                total_loss += Spa_loss
                
        # There's a corner case that returns NaN loss: when there are no masked tokens
        # however, likelihood of that is (1-p)^context_length
        
        # if self.hparams.masking_p == 0.0: # this case is uniquely for the fine tuning case (check _fine_tune_model)
        #     loss = torch.tensor(0.0, device=mlm_predictions.device)
        # else:
        #     if self.ag_loss:
        #         MLM_loss = self.loss(mlm_predictions, real_indices, norm_exp) # MLM loss
        #         ag_loss = self.compute_ag_loss(lamda = self.outer_config["lambda"])
        #         ag_loss_sche = ag_loss * self.outer_config["scale"] * self.linear_schedule_for_scale()      
        #         total_loss = MLM_loss + ag_loss_sche
        #         self.log('val_loss', MLM_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
        #         self.log('ag_loss', ag_loss_sche, sync_dist=True, prog_bar=False, reduce_fx='mean')
        #         self.log('total_loss', total_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')

        
        
        return total_loss
            
    
    def get_embeddings(self, batch, layers: List[int] = [11]):
        """
            This function gets representations to later load them in some script
            that computes a downstream task
            
            batch: batch who representation will be outputed
            layers (List[int]): list that contains the indices of the layers whose repr. will obtain
            function (str): "concat", "mean", "sum", "cls" or None to combine the hidden rep. obtained
        """        
        
        #batch['X'] = batch['X'][:, :self.hparams.context_length]
        # import pdb; pdb.set_trace()
        indices = batch["indices"].to(self.device)
        adjmtx = batch["adjmtx"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        predictions = self.forward(indices, adjmtx, attention_mask)
        
        hidden_repr = [predictions["transformer_output"][i] for i in layers]
       
     
        return hidden_repr
    def get_attention(self, batch, layers: List[int] = [11]):
        indices = batch["indices"].to(self.device)
        adjmtx = batch["adjmtx"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        predictions = self.forward(indices, adjmtx, attention_mask)
        attn_score = [predictions["attention_score"][i] for i in layers]
                  
        return attn_score
        
    
    def configure_optimizers(self):
        
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_epochs=self.hparams.max_epochs)
        
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)
                
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        import pdb; pdb.set_trace()
        encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding, persistent=False)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_epochs):
        self.warmup = warmup
        self.max_num_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [max(1e-5, base_lr * lr_factor) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_epochs))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
            
