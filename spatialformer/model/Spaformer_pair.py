import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from typing import List
from torch import optim
import numpy as np
import math
from ..utils import complete_masking, categorical_2d_masking
from .model import *
import pickle

MASK_TOKEN = 2
CLS_TOKEN = 1
PAD_TOKEN = 0
def to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, list):
        return [to_cpu(item) for item in data]
    elif isinstance(data, dict):
        return {key: to_cpu(value) for key, value in data.items()}
    return data


class PairLoss(nn.Module):
    def __init__(self):
        '''
        calculating the binary cross entropy loss for the cell pairs
        '''
        super(PairLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # import pdb; pdb.set_trace()
        target = target.long()
        loss = self.loss(input, target)
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self, mask_way = "MT", n_token = None, cls_token = None, sep_token = None):
        '''
        mask_way: MT means mask the token. ME means mask the expression level
        
        '''
        super(MaskedMSELoss, self).__init__()
        self.mask_way = mask_way
        self.n_token = n_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        #get rid of the special token with <SEP> and <CLS>
        target[(target == self.cls_token) | (target == self.sep_token)] = -100
        try:
            # import pdb; pdb.set_trace()
            loss = self.loss(input, target)
            if torch.isnan(loss) or torch.isinf(loss):
                print("input:", input)
                print("target:", target)
                raise ValueError("Loss is NaN or Inf")
                
        except:
            print("Loss computation failed, assigning default loss value.")
            loss = torch.tensor(5, device=input.device)
        return loss
    
class MaskedMSE2DLoss(nn.Module):
    def __init__(self, sample_balance = False):
        super(MaskedMSE2DLoss, self).__init__()
        self.sample_balance = sample_balance

    def even_sample(self, pred, target):
        # Flatten matrices for easier indexing
        predictions_flat = pred.view(-1)
        true_labels_flat = target.reshape(-1)
        
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
        # import pdb; pdb.set_trace()

        # Ensure valid_target is in the correct shape for gather
        if self.sample_balance:
            valid_input, valid_target = self.even_sample(input, target)
            mask_num = len(valid_target)
            try:
                spa_loss = F.binary_cross_entropy_with_logits(valid_input, valid_target.float(), reduction='mean')
                # spa_loss = loss / mask_num
                if torch.isnan(spa_loss) or torch.isinf(spa_loss):
                    print("valid_input:", valid_input)
                    print("valid_target:", valid_target)
                    raise ValueError("Loss is NaN or Inf")
            except:
                print("Loss computation failed, assigning default loss value.")
                spa_loss = torch.tensor(0.5, device=valid_input.device)

        return spa_loss
    



    
class AdjacencyProjector(nn.Module):
    def __init__(self, embedding_dim):
        super(AdjacencyProjector, self).__init__()
        # Assume output dimension of 1 because we want a single score
        self.fc = nn.Linear(2 * embedding_dim, 1, bias=False)

    def forward(self, E):
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
    def __init__(self, freeze, embedding_path):
        super().__init__()
        pretrained_weights = pickle.load(open(embedding_path, "rb"))
        #adding the cls random value
        num_features = pretrained_weights.shape[1]

        # Generate a new random row with the same number of columns
        new_row = torch.randn(1, num_features)  # Shape will be (1, 512)

        # Concatenate the new row to the existing tensor along axis 0 (rows)
        updated_weights = torch.cat((pretrained_weights, new_row), dim=0)
        self.emb = nn.Embedding.from_pretrained(updated_weights, freeze=freeze)
        print("require grad:", self.emb.weight.requires_grad)

    def forward(self, x):
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
            outer_config (dict): the overall config of the model for training
        """
        super().__init__()
        # self.batch_length = None
        self.encoder = SpaEncoder(dim=dim_model , num_layers=nlayers, groups=dim_model, num_heads=nheads, bpp_size = context_length, bpp = bpp, bpp_scale=bpp_scale)
        # As in HuggingFace
        # The prediction head for each masked token
        self.classifier_head = nn.Linear(dim_model, n_tokens+n_atokens, bias=False)
        self.pair_head = nn.Linear(dim_model, 2, bias=False)
        self.adjprojector = AdjacencyProjector(dim_model)

        
        bias = nn.Parameter(torch.zeros(n_tokens+n_atokens)) # each token has its own bias
        self.classifier_head.bias =  bias
            
        # As in HuggingFace
        self.activation = nn.Tanh()

        self.embeddings = nn.Embedding(num_embeddings=n_tokens+n_atokens, embedding_dim=dim_model, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(num_embeddings=3, embedding_dim=dim_model)
        self.token_type_embeddings.weight.requires_grad = False

        #loading the pretrained embeddings
        if outer_config["spatial_embedding"]:
            self.spatialembeds = GraphSAGESpatialEmbedding(outer_config["spatial_embedding_freeze"], outer_config["embedding_path"])

        if pool == 'cls':
            context_length += 1

        # MLM loss
        self.MEloss = MaskedMSELoss(mask_way = mask_way, n_token = n_tokens+n_atokens, cls_token = outer_config["cls_token"], sep_token = outer_config["sep_token"])#masked token loss
        self.MDloss = MaskedMSE2DLoss(sample_balance = True)
        self.Pairloss = PairLoss()
            
        self.save_hyperparameters()
        #initialize as 0 to get better proformance, to homoscedastic uncertainty
        self.log_sigma_class = nn.Parameter(torch.zeros(1))  # Log variance for classification
        self.log_sigma_reg1 = None    # Log variance for regression
        self.log_sigma_reg2 = None
        self.log_sigma_reg3 = None
        

        self.gc_freq = 5
        
        self.batch_train_losses = []
        
        self.batch_input = {}
        self.total_tokens = 0
        self.adjmtx = None
        self.attentions = None
        self.nheads = nheads
        self.outer_config = outer_config
        self.ag_loss = ag_loss
        self.setup_sigma()
        self.last_train_loss = None
        self.last_val_loss = None
        self.mask_way = mask_way
        self.mask_token = outer_config["mask_token"]
        self.pad_token = outer_config["pad_token"]
        self.sep_token = outer_config["sep_token"]
        self.cls_token = outer_config["cls_token"]
        self.dim_model = dim_model
        self.nlayers = nlayers
        self.nheads = nheads
        self.bpp = bpp
        self.bpp_scale = bpp_scale



        
        
    # def set_variable_length(self, new_batch_length):
    #     # Reinitialize the encoder with the new batch_length
    #     self.encoder = SpaEncoder(
    #         dim=self.dim_model,
    #         num_layers=self.nlayers,
    #         groups=self.dim_model,
    #         num_heads=self.nheads,
    #         bpp_size=new_batch_length,
    #         bpp=self.bpp,
    #         bpp_scale=self.bpp_scale
    #     )
    #     self.encoder.to(self.device)

            
    def forward(self, x, adjmtx, attention_mask, token_type_ids, **kwargs):
        # x = x.to(self.device)
        # adjmtx = adjmtx.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        # token_type_ids = token_type_ids.to(self.device)
        # x -> size: batch x (context_length) x 1
        # import pdb; pdb.set_trace()
        # adjmtx = torch.cat(adj_left, adj_right)
        token_embedding = self.embeddings(x) # batch x (context_length) x dim_model
        #adding the token type embeddings
        #get the first sep site

        token_embedding += self.token_type_embeddings(token_type_ids)
        if self.outer_config["spatial_embedding"]:
            token_embedding += self.spatialembeds(x)
        
        # import pdb; pdb.set_trace()
        # print("self.encoder device:", self.encoder.device)
        # print("token_embedding device:", token_embedding.device)
        # print("attention_mask device:", attention_mask.device)
        # print("adjmtx device:", adjmtx.device)
        # import pdb; pdb.set_trace()
        
        transformer_output, attn_scores = self.encoder(token_embedding, False, attention_mask) # batch x (n_tokens) x dim_model
        # import pdb; pdb.set_trace()
        #get the last layer of the model output
        prediction = self.classifier_head(transformer_output[-1])
        #get the pair-wise gene-gene interaction matrix
        dis_prediction = self.adjprojector(token_embedding)
        self.attentions = attn_scores
        pair_prediction = self.pair_head(transformer_output[-1][:,0]) # the first token embeddings as input and predict whether two cells are paired

        return {'mlm_prediction': prediction,
                'pair_prediction': pair_prediction,
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
    def get_acc(self, pred, target):
        # import pdb; pdb.set_trace()
        _, pred = torch.max(pred.data, 1)  # Get the predicted class
        total = target.size(0)  # Total number of samples
        correct = (pred == target).sum().item()  # Count correct predictions
        accuracy = correct / total
        return accuracy
    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        #skip the None batch, which means the gene-gene matrix is sum as 0
        torch.cuda.synchronize()
        attention_mask = batch['attention_mask']
        batch['adjmtx'] = batch['adjmtx'].to(self.device)
        batch['indices'] = batch['indices'].to(self.device)
        batch["pair_label"] = batch["pair_label"].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(self.device)
        # import pdb; pdb.set_trace()
        # mem_before = torch.cuda.memory_allocated()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        # used_memory = torch.cuda.memory_reserved() 
        # length = batch["indices"].shape[1]
        # print(f"Allocated Memory before: {mem_before / 1e6:.2f} MB")  # Convert to MB
        # print(f"Total Memory before: {total_memory / 1e6:.2f} MB")  # Convert to MB
        # print(f"Used Memory before: {used_memory / 1e6:.2f} MB") 
        # print(f"batch variable length: {length}")
        # Training code
        if batch is not None:
            try:
                if self.outer_config["mini_batch"]:
                    #set up minibatch
                    mini_batch_list = self.mini_batch_setup(batch)    
                    total_loss = torch.tensor(0, dtype=torch.float, device=self.device)

                    #for mlm minibatch
                    exp_batch = mini_batch_list[0]
                    mlm_predictions, real_indices = self.predict_exp(exp_batch, "normalized_exp") #different task means different targets here
                    MLM_loss = self.MEloss(mlm_predictions, real_indices) # MLM loss
                    self.log(f'train_MLM_loss(normalized_exp)', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    log_sigma_mlm = getattr(self, f'log_sigma_reg1')[0]
                    self.log(f'log_sigma_reg(normalized_exp)', log_sigma_mlm, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    total_loss += (torch.exp(-log_sigma_mlm) * MLM_loss + log_sigma_mlm) ##
                    # import pdb; pdb.set_trace()
                    #for spatial minibatch, including gene and cell spatial info
                    spa_batch = mini_batch_list[-1]
                    #for spatial jobs
                    spa_predictions, adjmtx, pair_predictions, pair_label = self.predict_spa(spa_batch)
                    # import pdb; pdb.set_trace()
                    Spa_loss = self.MDloss(spa_predictions, adjmtx)# spatial loss
                    Pair_loss = self.Pairloss(pair_predictions, pair_label) #Paired loss
                    self.log(f'train_Pair_loss', Pair_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    accuracy = self.get_acc(pair_predictions, pair_label)
                    self.log(f'train_pair_accuracy', accuracy, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    log_sigma_pair = getattr(self, f'log_sigma_reg2')[0]
                    self.log(f'log_sigma_reg(pair)', log_sigma_pair, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    total_loss += (torch.exp(-log_sigma_pair) * Pair_loss * self.linear_schedule_for_scale() + log_sigma_pair)
                    # total_loss += 2 * Pair_loss
                    # import pdb; pdb.set_trace()
                    self.log('train_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    total_loss += (torch.exp(-self.log_sigma_class[0]) * Spa_loss * self.linear_schedule_for_scale() + self.log_sigma_class[0]) ##
                    self.log('log_sigma_class', self.log_sigma_class[0], sync_dist=True, prog_bar=True, reduce_fx='mean')
                    self.log('train_total_loss', total_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
                else:
                    #no minibatch  
                    total_loss = torch.tensor(0, dtype=torch.float, device=self.device)
                    # import pdb; pdb.set_trace()
                    #for mlm minibatch
                    batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens, self.cls_token, self.mask_token, self.sep_token, self.pad_token) #for mask expression prediction
                    mlm_predictions, real_indices, spa_predictions, pair_predictions, pair_label, adjmtx = self.predict_exp(batch, "normalized_exp") #different task means different targets here
                    MLM_loss = self.MEloss(mlm_predictions, real_indices) # MLM loss
                    self.log(f'train_MLM_loss(normalized_exp)', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    log_sigma_mlm = getattr(self, f'log_sigma_reg1')[0]
                    self.log(f'log_sigma_reg(normalized_exp)', log_sigma_mlm, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    total_loss += (torch.exp(-log_sigma_mlm) * MLM_loss + log_sigma_mlm) ##
                    # import pdb; pdb.set_trace()
                    # import pdb; pdb.set_trace()
                    Spa_loss = self.MDloss(spa_predictions, adjmtx)# spatial loss
                    Pair_loss = self.Pairloss(pair_predictions, pair_label) #Paired loss
                    self.log(f'train_Pair_loss', Pair_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    accuracy = self.get_acc(pair_predictions, pair_label)
                    self.log(f'train_pair_accuracy', accuracy, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    log_sigma_pair = getattr(self, f'log_sigma_reg2')[0]
                    self.log(f'log_sigma_reg(pair)', log_sigma_pair, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    total_loss += (torch.exp(-log_sigma_pair) * Pair_loss * self.linear_schedule_for_scale() + log_sigma_pair)
                    # total_loss += 2 * Pair_loss
                    # import pdb; pdb.set_trace()
                    self.log('train_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    total_loss += (torch.exp(-self.log_sigma_class[0]) * Spa_loss * self.linear_schedule_for_scale() + self.log_sigma_class[0]) ##
                    self.log('log_sigma_class', self.log_sigma_class[0], sync_dist=True, prog_bar=True, reduce_fx='mean')
                    self.log('train_total_loss', total_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')

                # There's a corner case that returns NaN loss: when there are no masked tokens
                # however, likelihood of that is (1-p)^context_length

                #getting the total tokens

                self.total_tokens += attention_mask.sum()
                self.log('total_tokens', self.total_tokens, prog_bar=True, sync_dist=True, reduce_fx='sum')
            
                # mem_after = torch.cuda.memory_allocated()
                # print(f"GPU memory usage: {mem_before/1e9} GB -> {mem_after/1e9} GB") 
                # mem_after = torch.cuda.memory_allocated()
                # total_memory = torch.cuda.get_device_properties(0).total_memory
                # used_memory = torch.cuda.memory_reserved() 
                # print(f"Allocated Memory after: {mem_after / 1e6:.2f} MB")  # Convert to MB
                # print(f"Total Memory after: {total_memory / 1e6:.2f} MB")  # Convert to MB
                # print(f"Used Memory after: {used_memory / 1e6:.2f} MB") 
                self.last_train_loss = total_loss.item()  
                
                # Check for NaN loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    encoder_state_dict = self.encoder.state_dict()
                    encoder_state_dict_cpu = {k: v.cpu() for k, v in encoder_state_dict.items()}
                    torch.save(encoder_state_dict, "/scratch/project_465001027/Spatialformer/data/encoder_state_dict.pth")
                # if torch.isnan(total_loss).any():
                #     #save everything
                    pickle.dump(to_cpu(batch), open("/scratch/project_465001027/Spatialformer/data/nanbatch.pkl", "wb"))
                    pickle.dump(to_cpu(mini_batch_list), open("/scratch/project_465001027/Spatialformer/data/minibatchlist.pkl", "wb"))
                    pickle.dump(to_cpu(mlm_predictions), open("/scratch/project_465001027/Spatialformer/data/mlm_predictions.pkl", "wb"))
                    pickle.dump(to_cpu(real_indices), open("/scratch/project_465001027/Spatialformer/data/real_indices.pkl", "wb"))
                    pickle.dump(to_cpu(attention_mask), open("/scratch/project_465001027/Spatialformer/data/attention_mask.pkl", "wb"))
                    pickle.dump(to_cpu(MLM_loss), open("/scratch/project_465001027/Spatialformer/data/MLM_loss.pkl", "wb"))
                    # pickle.dump(to_cpu(log_sigma), open("/scratch/project_465001027/Spatialformer/data/log_sigma.pkl", "wb"))
                    pickle.dump(to_cpu(spa_predictions), open("/scratch/project_465001027/Spatialformer/data/spa_predictions.pkl", "wb"))
                    pickle.dump(to_cpu(adjmtx), open("/scratch/project_465001027/Spatialformer/data/adjmtx.pkl", "wb"))
                    pickle.dump(to_cpu(Spa_loss), open("/scratch/project_465001027/Spatialformer/data/Spa_loss.pkl", "wb"))
                    pickle.dump(to_cpu(self.log_sigma_class[0]), open("/scratch/project_465001027/Spatialformer/data/log_sigma_class.pkl", "wb"))
                    pickle.dump(to_cpu(self.linear_schedule_for_scale()), open("/scratch/project_465001027/Spatialformer/data/linear_schedule_for_scale.pkl", "wb"))
                    raise ValueError(f"NaN or Inf loss detected on batch index {batch_idx}.")
                return total_loss   
            except ValueError as e:
                print(f"Stopping training due to: {e}")
                # This could stop the training loop
                return 
        else:

            return



       

             


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
        div = self.outer_config["n_tasks"]
        batch_size = batch["indices"].shape[0]
        
        # assert batch_size%div == 0, f"For minibatch implementation, the batch size should be set to {div}*n"
        mini_size = int(batch_size/div)
        # print(f"mini batch size = {mini_size}")
        mini_batch_list = []
        #only apply for exp batch
        #for masked token prediction task
        mini_batch = 0
        exp_batch = {k: v[int(mini_batch*mini_size):int((mini_batch+1)*mini_size)] for k, v in batch.items()}
        exp_batch = complete_masking(exp_batch, self.hparams.masking_p, self.hparams.n_tokens, self.cls_token, self.mask_token, self.sep_token, self.pad_token) #for mask expression prediction
        mini_batch_list.append(exp_batch)
        if div > 1:
        #for paired cell prediction and gene-gene co-occurrence prediction
            mini_batch = 1
            pc_batch = {k: v[int(mini_batch*mini_size):int((mini_batch+1)*mini_size)] for k, v in batch.items()}
            mini_batch_list.append(pc_batch)
        #for gene co-occurrence prediction        
        # spa_batch = {k: v[-mini_size:] for k, v in batch.items()}
        # mini_batch_list.append(spa_batch)
        
        return mini_batch_list
    
    def predict_exp(self, batch, target = "normalized_exp"):
        #in case the mask is none
        with torch.no_grad():
            real_indices = batch['indices']
            mask = batch['mask']
            no_mask = torch.all(torch.where(real_indices != PAD_TOKEN, mask, 1) == 1)
            while no_mask:
                batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens, self.cls_token, self.mask_token, self.sep_token, self.pad_token)
                real_indices = batch['indices']
                mask = batch['mask']
                no_mask = torch.all(torch.where(real_indices != PAD_TOKEN, mask, 1) == 1)
        # import pdb; pdb.set_trace()
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']
        token_type_ids = batch["token_type_ids"]      
        adjmtx = batch['adjmtx']
        # import pdb; pdb.set_trace()
        # batch = pickle.load(open("/scratch/project_465001027/Spatialformer/data/nanbatch.pkl","rb"))


        predictions = self.forward(masked_indices, adjmtx, attention_mask, token_type_ids)
        mlm_predictions = predictions['mlm_prediction']
        real_indices = torch.where(mask==MASK_TOKEN, real_indices, torch.tensor(-100, dtype=torch.long)).type(torch.int64)
        mlm_predictions = mlm_predictions.view(-1, self.hparams.n_tokens+self.hparams.n_atokens)
        # import pdb; pdb.set_trace()
        real_indices = real_indices.view(-1)

        if self.outer_config["mini_batch"] == False:
            pair_label = batch["pair_label"]
            spa_predictions = predictions['spa_prediction']
            pair_predictions = predictions['pair_prediction']
            return mlm_predictions, real_indices, spa_predictions, pair_predictions, pair_label, adjmtx
        else:

            return mlm_predictions, real_indices

    def predict_spa(self, batch):
        adjmtx = batch['adjmtx']
        indices = batch['indices']
        pair_label = batch["pair_label"]
        attention_mask = batch['attention_mask']
        token_type_ids = batch["token_type_ids"]
        # import pdb; pdb.set_trace()
        predictions = self.forward(indices, adjmtx, attention_mask, token_type_ids)
        spa_predictions = predictions['spa_prediction']
        pair_predictions = predictions['pair_prediction']
        return spa_predictions, adjmtx, pair_predictions, pair_label
    
    def setup_sigma(self):
        n_tasks = 3
        for i in range(1, n_tasks):
            setattr(self, f'log_sigma_reg{i}', nn.Parameter(torch.zeros(1)))


    
    

    
    def validation_step(self, batch, batch_idx, *args, **kwargs): 
        batch['adjmtx'] = batch['adjmtx'].to(self.device)
        batch['indices'] = batch['indices'].to(self.device)
        batch["pair_label"] = batch["pair_label"].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(self.device)
        # import pdb; pdb.set_trace()
        if batch is not None:
            try:
                if self.outer_config["mini_batch"]:
                    #set up minibatch
                    mini_batch_list = self.mini_batch_setup(batch)    
                    total_loss = torch.tensor(0, dtype=torch.float, device=self.device)

                    #for mlm minibatch
                    exp_batch = mini_batch_list[0]
                    mlm_predictions, real_indices = self.predict_exp(exp_batch, "normalized_exp") #different task means different targets here
                    MLM_loss = self.MEloss(mlm_predictions, real_indices) # MLM loss
                    self.log(f'val_MLM_loss(normalized_exp)', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    log_sigma_mlm = getattr(self, f'log_sigma_reg1')[0]
                    total_loss += (torch.exp(-log_sigma_mlm) * MLM_loss + log_sigma_mlm)
      
                    #for spatial minibatch, including gene and cell spatial info
                    spa_batch = mini_batch_list[-1]
                    #for spatial jobs
                    spa_predictions, adjmtx, pair_predictions, pair_label = self.predict_spa(spa_batch)
                    Spa_loss = self.MDloss(spa_predictions, adjmtx)# spatial loss
                    Pair_loss = self.Pairloss(pair_predictions, pair_label) #Paired loss
                    self.log(f'val_Pair_loss', Pair_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    accuracy = self.get_acc(pair_predictions, pair_label)
                    self.log(f'val_pair_accuracy', accuracy, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    log_sigma_pair = getattr(self, f'log_sigma_reg2')[0]
                    total_loss += (torch.exp(-log_sigma_pair) * Pair_loss * self.linear_schedule_for_scale() + log_sigma_pair)
                    self.log('val_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    total_loss += (torch.exp(-self.log_sigma_class[0]) * Spa_loss * self.linear_schedule_for_scale() + self.log_sigma_class[0])
                    self.log('val_total_loss', total_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
                else:
                    # import pdb; pdb.set_trace()
                    batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens, self.cls_token, self.mask_token, self.sep_token, self.pad_token)
                    total_loss = torch.tensor(0, dtype=torch.float, device=self.device)

                    mlm_predictions, real_indices, spa_predictions, pair_predictions, pair_label, adjmtx = self.predict_exp(batch, "normalized_exp")#different task means different targets here
                    MLM_loss = self.MEloss(mlm_predictions, real_indices) # MLM loss
                    self.log(f'val_MLM_loss(normalized_exp)', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    log_sigma_mlm = getattr(self, f'log_sigma_reg1')[0]
                    total_loss += (torch.exp(-log_sigma_mlm) * MLM_loss + log_sigma_mlm)

                    Spa_loss = self.MDloss(spa_predictions, adjmtx)# spatial loss
                    Pair_loss = self.Pairloss(pair_predictions, pair_label) #Paired loss
                    self.log(f'val_Pair_loss', Pair_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    accuracy = self.get_acc(pair_predictions, pair_label)
                    self.log(f'val_pair_accuracy', accuracy, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    log_sigma_pair = getattr(self, f'log_sigma_reg2')[0]
                    total_loss += (torch.exp(-log_sigma_pair) * Pair_loss * self.linear_schedule_for_scale() + log_sigma_pair)
                    self.log('val_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    total_loss += (torch.exp(-self.log_sigma_class[0]) * Spa_loss * self.linear_schedule_for_scale() + self.log_sigma_class[0])
                    self.log('val_total_loss', total_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
            # Check for NaN loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                
                # if torch.isnan(total_loss).any():
                #     #save everything
                    pickle.dump(to_cpu(batch), open("/scratch/project_465001027/Spatialformer/data/nanbatch.pkl", "wb"))
                    pickle.dump(to_cpu(mini_batch_list), open("/scratch/project_465001027/Spatialformer/data/minibatchlist.pkl", "wb"))
                    pickle.dump(to_cpu(mlm_predictions), open("/scratch/project_465001027/Spatialformer/data/mlm_predictions.pkl", "wb"))
                    pickle.dump(to_cpu(real_indices), open("/scratch/project_465001027/Spatialformer/data/real_indices.pkl", "wb"))
                    pickle.dump(to_cpu(MLM_loss), open("/scratch/project_465001027/Spatialformer/data/MLM_loss.pkl", "wb"))
                    pickle.dump(to_cpu(spa_predictions), open("/scratch/project_465001027/Spatialformer/data/spa_predictions.pkl", "wb"))
                    pickle.dump(to_cpu(adjmtx), open("/scratch/project_465001027/Spatialformer/data/adjmtx.pkl", "wb"))
                    pickle.dump(to_cpu(Spa_loss), open("/scratch/project_465001027/Spatialformer/data/Spa_loss.pkl", "wb"))
                    pickle.dump(to_cpu(self.log_sigma_class[0]), open("/scratch/project_465001027/Spatialformer/data/log_sigma_class.pkl", "wb"))
                    pickle.dump(to_cpu(self.linear_schedule_for_scale()), open("/scratch/project_465001027/Spatialformer/data/linear_schedule_for_scale.pkl", "wb"))
                    raise ValueError(f"NaN or Inf loss detected on batch index {batch_idx}.")
                return total_loss    
            except ValueError as e:
                print(f"Stopping training due to: {e}")
                # This could stop the training loop
                return 
        else:
            
            return



          
       
  
            
    
    def get_embeddings(self, batch, layers: List[int] = [11], pair_prediction=False, co_prediction=False):
        """
            This function gets representations to later load them in some script
            that computes a downstream task
            
            batch: batch who representation will be outputed
            layers (List[int]): list that contains the indices of the layers whose repr. will obtain
            function (str): "concat", "mean", "sum", "cls" or None to combine the hidden rep. obtained
        """        
        
        
        indices = batch["indices"].to(self.device)
        # adjmtx = batch["adjmtx"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        predictions = self.forward(indices, False, attention_mask, token_type_ids)
        if not pair_prediction:
            if not co_prediction:
                hidden_repr = [predictions["transformer_output"][i] for i in layers]
            
            
                return hidden_repr
            elif co_prediction:
                hidden_repr = [predictions["transformer_output"][i] for i in layers]
                spa_prediction = predictions["spa_prediction"]#adj matrix
                probabilities = torch.nn.functional.softmax(spa_prediction, dim=0)
                # import pdb; pdb.set_trace()
                return hidden_repr, probabilities

        else:
            
            hidden_repr = [predictions["transformer_output"][i] for i in layers]
            pair_result_logit = predictions["pair_prediction"]
            #to probabilities
            probabilities = torch.nn.functional.softmax(pair_result_logit, dim=1)
            return hidden_repr, probabilities
    def get_attention(self, batch, layers: List[int] = [11]):
        indices = batch["indices"].to(self.device)
        adjmtx = batch["adjmtx"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        new_batch_length = indices.shape[1]
        # self.set_variable_length(new_batch_length)
        predictions = self.forward(indices, adjmtx, attention_mask, token_type_ids)
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
        # import pdb; pdb.set_trace()
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
    
            
