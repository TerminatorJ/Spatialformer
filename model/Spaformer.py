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
from utils import complete_masking
from model import *
import pickle
MASK_TOKEN = 353
CLS_TOKEN = 2
PAD_TOKEN = 0


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
                 modality: bool = False
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
        """
        super().__init__()
        # import pdb; pdb.set_trace()
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nheads, dim_feedforward=dim_feedforward, batch_first=batch_first, dropout=dropout, layer_norm_eps=1e-12)
        # self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=nlayers, enable_nested_tensor=False)
        self.encoder = SpaEncoder(dim=dim_model , num_layers=nlayers, groups=dim_model, num_heads=nheads)
        # As in HuggingFace
        # The prediction head for each masked token
        self.classifier_head = nn.Linear(dim_model, n_tokens, bias=False)
        
        bias = nn.Parameter(torch.zeros(n_tokens)) # each token has its own bias
        self.classifier_head.bias =  bias
            
        # As in HuggingFace
        # self.pooler_head = nn.Linear(dim_model, dim_model)
        self.activation = nn.Tanh()

        # Token embedding learnable weights
        self.embeddings = nn.Embedding(num_embeddings=n_tokens+n_atokens, embedding_dim=dim_model, padding_idx=0)
        
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
        self.loss = nn.CrossEntropyLoss()
            
        self.save_hyperparameters()

        self.gc_freq = 5
        
        self.batch_train_losses = []
        
        self.initialize_weights()
        self.batch_input = {}
        self.total_tokens = 0

            
    def forward(self, x, adjmtx, attention_mask):
                
        # x -> size: batch x (context_length) x 1
        # import pdb; pdb.set_trace()
        token_embedding = self.embeddings(x) # batch x (context_length) x dim_model
        transformer_output = self.encoder(token_embedding, adjmtx, attention_mask) # batch x (n_tokens) x dim_model
        # import pdb; pdb.set_trace()
        # MLM prediction
        #get the last layer of the model output
        prediction = self.classifier_head(transformer_output[-1])
        # import pdb; pdb.set_trace()
        return {'mlm_prediction': prediction,
                'transformer_output': transformer_output}
    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        # Training code
    
    
        with torch.no_grad():
            #the mask includes special tokens
            batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)
            real_indices = batch['indices']
            mask = batch['mask']
            # import pdb; pdb.set_trace()
            no_mask = torch.all(torch.where(real_indices != PAD_TOKEN, mask, 1) == 1)
            # print("no_mask", no_mask)
            while no_mask:
                print("no_mask", no_mask)
                print("mask:", mask[0])
                print("real_indices", real_indices[0])
                batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)
                real_indices = batch['indices']
                mask = batch['mask']
                no_mask = torch.all(torch.where(real_indices != PAD_TOKEN, mask, 1) == 1)
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']
        # mask = batch['mask']
        
        adjmtx = batch['adjmtx']
        #save the batch index dict
        self.batch_input[batch_idx] = (masked_indices, adjmtx, attention_mask)



        # import pdb; pdb.set_trace()
        predictions = self.forward(masked_indices, adjmtx, attention_mask)
        mlm_predictions = predictions['mlm_prediction']
        # import pdb; pdb.set_trace()
        # we just evaluate on the masked tokens (mask = 0)
        # import pdb; pdb.set_trace()
        real_indices = torch.where(mask==MASK_TOKEN, real_indices, torch.tensor(-100, dtype=torch.long)).type(torch.int64)
        #also set the padding site as -100 to ignore the padding sites
        # real_indices = torch.where(real_indices == PAD_TOKEN, torch.tensor(-100, dtype=torch.long), real_indices).type(torch.int64)
        # import pdb; pdb.set_trace()
        mlm_predictions = mlm_predictions.view(-1, self.hparams.n_tokens)
        real_indices = real_indices.view(-1)
        masked_indices = masked_indices.view(-1)
        mask = mask.view(-1)


        #getting the total tokens

        self.total_tokens += attention_mask.sum()
        self.log('total_tokens', self.total_tokens, prog_bar=True)


        # There's a corner case that returns NaN loss: when there are no masked tokens
        # however, likelihood of that is (1-p)^context_length, but our sequence can be too short to get masked
        # Therefore, we need to make sure they have at lease one masked

        if self.hparams.masking_p == 0.0: # this case is uniquely for the fine tuning case (check _fine_tune_model)
            loss = torch.tensor(0.0, device=mlm_predictions.device)
        else:
            loss = self.loss(mlm_predictions, real_indices) # MLM loss
        # import pdb; pdb.set_trace()                 
        if torch.isnan(loss):
            pickle.dump(self.batch_input, open("./save_input.pkl", "wb"))
            param = next(iter(self.encoder.named_parameters()))
            #get the 
            import pdb; pdb.set_trace()  

        mem_after = torch.cuda.memory_allocated()
        print(f"GPU memory usage: {mem_before/1e9} GB -> {mem_after/1e9} GB")               
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
        
        return loss

    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        
        with torch.no_grad():
            batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)
            real_indices = batch['indices']
            mask = batch['mask']
            no_mask = torch.all(torch.where(real_indices != PAD_TOKEN, mask, 1) == 1)
            while no_mask:
                batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)
                real_indices = batch['indices']
                mask = batch['mask']
                no_mask = torch.all(torch.where(real_indices != PAD_TOKEN, mask, 1) == 1)
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']
        # mask = batch['mask']
        real_indices = batch['indices']
        adjmtx = batch['adjmtx']

        predictions = self.forward(masked_indices, adjmtx, attention_mask)
        mlm_predictions = predictions['mlm_prediction']
        
        real_indices = torch.where(mask==MASK_TOKEN, real_indices, torch.tensor(-100, dtype=torch.long)).type(torch.int64)
        #also set the padding site as -100 to ignore the padding sites
        # real_indices = torch.where(real_indices == PAD_TOKEN, torch.tensor(-100, dtype=torch.long), real_indices).type(torch.int64)

        mlm_predictions = mlm_predictions.view(-1, self.hparams.n_tokens)
        real_indices = real_indices.view(-1)
        masked_indices = masked_indices.view(-1)

        # There's a corner case that returns NaN loss: when there are no masked tokens
        # however, likelihood of that is (1-p)^context_length
        
        if self.hparams.masking_p == 0.0: # this case is uniquely for the fine tuning case (check _fine_tune_model)
            loss = torch.tensor(0.0, device=mlm_predictions.device)
        else:
            loss = self.loss(mlm_predictions, real_indices) # MLM loss
        
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
        
        return loss
            
    
    def get_embeddings(self, batch, layers: List[int] = [11]):
        """
            This function gets representations to later load them in some script
            that computes a downstream task
            
            batch: batch who representation will be outputed
            layers (List[int]): list that contains the indices of the layers whose repr. will obtain
            function (str): "concat", "mean", "sum", "cls" or None to combine the hidden rep. obtained
        """        
        
        #batch['X'] = batch['X'][:, :self.hparams.context_length]
        indices = batch["indices"]
        adjmtx = batch["adjmtx"]
        attention_mask = batch["attention_mask"]
        predictions = self.forward(indices, adjmtx, attention_mask)
        
        hidden_repr = [predictions["transformer_output"][i] for i in layers]
       
       
                        
        return hidden_repr
        
    
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
    
