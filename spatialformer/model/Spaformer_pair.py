import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Dict, Tuple, Optional, Any
from torch import optim
import numpy as np
import math
import torch.distributed as dist
from spatialformer.utils import complete_masking, categorical_2d_masking
from .model import SpaEncoder
import pickle
import os
MASK_TOKEN = 2
CLS_TOKEN = 1
PAD_TOKEN = 0

def to_cpu(data):
    """
    Recursively move data from GPU to CPU. 
    
    Args:
        data: Input data (tensor, list, dict, or other)
        
    Returns:
        Data moved to CPU with gradients detached
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, list):
        return [to_cpu(item) for item in data]
    elif isinstance(data, dict):
        return {key: to_cpu(value) for key, value in data.items()}
    return data

# class PairFocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0):
#         """
#         Focal loss for handling class imbalance.
#         alpha: weight for positive class (lower = downweight positives)
#         gamma: focusing parameter (higher = focus on hard examples)
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
    
#     def forward(self, logits, labels):
#         """
#         Args:
#             logits: [batch_size, 2] - class logits
#             labels: [batch_size] - class labels (0 or 1)
#         """
#         ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
#         # Get probabilities
#         probs = F.softmax(logits, dim=1)
#         p_t = probs[range(len(labels)), labels]
        
#         # Focal weight
#         focal_weight = (1 - p_t) ** self.gamma
        
#         # Alpha weighting (apply to positive class)
#         alpha_t = torch.where(labels == 1, 
#                              torch.tensor(self.alpha, device=labels.device),
#                              torch.tensor(1-self.alpha, device=labels.device))
        
#         loss = alpha_t * focal_weight * ce_loss
#         return loss.mean()


class PairLoss_with_weight(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is None:
            pos_weight = [1.0, 3.0]
        self._pos_weight = pos_weight
    
    def forward(self, logits, labels):
        weight = torch.tensor(self._pos_weight, device=logits.device, dtype=logits.dtype)
        labels = labels.to(dtype=torch.int64)
        return F.cross_entropy(logits, labels, weight=weight)
# class PairLoss(nn.Module):
#     def __init__(self):
#         '''
#         calculating the binary cross entropy loss for the cell pairs
#         '''
#         super(PairLoss, self).__init__()
#         self.loss = nn.CrossEntropyLoss()
#     def forward(self, input: torch.Tensor, target: torch.Tensor):
#         # import pdb; pdb.set_trace()
#         target = target.long()
#         loss = self.loss(input, target)
#         return loss


# class MaskedMSELoss(nn.Module):
#     """
#     Cross-entropy loss for masked language modeling (MLM).
#     Only computes loss on masked tokens, ignoring special tokens.
#     """
    
#     def __init__(self, mask_way: str = "MT", n_token: Optional[int] = None, 
#                  cls_token: Optional[int] = None, sep_token: Optional[int] = None):
#         """
#         Args:
#             mask_way: Masking strategy ("MT" for mask token, "ME" for mask expression)
#             n_token: Total number of tokens in vocabulary
#             cls_token: CLS token ID to ignore in loss
#             sep_token: SEP token ID to ignore in loss
#         """
#         super(MaskedMSELoss, self).__init__()
#         self.mask_way = mask_way
#         self.n_token = n_token
#         self.cls_token = cls_token
#         self. sep_token = sep_token
#         self.loss = nn.CrossEntropyLoss()

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch. Tensor:
#         """
#         Compute MLM loss on masked tokens only.
        
#         Args:
#             input: Predicted logits [batch_size * seq_len, vocab_size]
#             target: Ground truth token IDs [batch_size * seq_len]
#                    -100 indicates positions to ignore
            
#         Returns:
#             Scalar loss value
#         """
#         # Mark special tokens to ignore in loss calculation
#         target[(target == self.cls_token) | (target == self.sep_token)] = -100
        
#         try:
#             loss = self.loss(input, target)
#             if torch.isnan(loss) or torch.isinf(loss):
#                 print("input:", input)
#                 print("target:", target)
#                 raise ValueError("Loss is NaN or Inf")
#         except:
#             print("Loss computation failed, assigning default loss value.")
#             loss = torch.tensor(5.0, device=input.device)
#         return loss

class AdjacencyProjector(nn.Module):
    """
    Low-rank bilinear projection for predicting gene-gene adjacency matrices.
    Computes block-diagonal adjacency for paired sequences.
    Returns blocks instead of full dense matrix to save memory.
    """
    
    def __init__(self, embedding_dim: int, rank: int = 128):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            rank: Rank of bilinear form (lower rank = fewer parameters)
        """
        super(AdjacencyProjector, self).__init__()
        self.embedding_dim = embedding_dim
        self.rank = rank
        
        # Low-rank bilinear projection
        self.project_i = nn.Linear(embedding_dim, rank, bias=False)
        self.project_j = nn.Linear(embedding_dim, rank, bias=False)
    
    def forward(self, E: torch.Tensor, seq_lengths: torch.Tensor) -> Dict[str, Any]:
        """
        Compute block-diagonal adjacency matrix for variable-length sequences.
        
        Args:
            E: Padded embeddings [batch_size, max_seq_len, embedding_dim]
            seq_lengths: Actual sequence lengths [batch_size, 2]
                        where seq_lengths[i] = [len1, len2] for sample i
        
        Returns:
            dict with:
                - 'block1_list': List of [L1, L1] tensors for sequence 1
                - 'block2_list': List of [L2, L2] tensors for sequence 2
                - 'seq_lengths': Original sequence lengths [batch_size, 2]
                - 'batch_size': B
        """
        B, L_max, dim = E.shape
        device = E.device
        
        # Project to interaction space
        E_i = self.project_i(E)  # [B, L_max, rank]
        E_j = self.project_j(E)  # [B, L_max, rank]
        
        # Compute blocks for each sample
        block1_list = []
        block2_list = []

        for b in range(B):
            L1 = seq_lengths[b, 0].item()
            L2 = seq_lengths[b, 1].item()

            # Block 1: sequence 1 self-adjacency [L1, L1]
            if L1 > 0:
                block1 = torch.mm(
                    E_i[b, :L1, :],      # [L1, rank]
                    E_j[b, :L1, :]. T     # [rank, L1]
                )  # [L1, L1]

                block1_list.append(block1)
            else:
                # Empty block
                block1_list.append(torch.zeros(0, 0, device=device, dtype=E.dtype))
            
            # Block 2: sequence 2 self-adjacency [L2, L2]
            if L2 > 0:
                start = L1
                end = L1 + L2
                block2 = torch.mm(
                    E_i[b, start:end, :],  # [L2, rank]
                    E_j[b, start:end, :].T # [rank, L2]
                )  # [L2, L2]

                block2_list.append(block2)
            else:
                # Empty block
                block2_list.append(torch.zeros(0, 0, device=device, dtype=E.dtype))

        return {
            'block1_list': block1_list,
            'block2_list': block2_list,
            'seq_lengths': seq_lengths,
            'batch_size': B
        }

class MaskedMSE2DLoss(nn.Module):
    """
    Focal loss for binary adjacency matrix prediction. 
    Handles class imbalance in sparse adjacency matrices.
    Works with block representations instead of full dense matrices.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Loss reduction method ('mean' or 'sum')
        """
        super(MaskedMSE2DLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred_blocks: Dict[str, Any], target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for adjacency prediction using block representation.
        
        Args:
            pred_blocks: Dict from AdjacencyProjector. forward() with:
                - 'block1_list': List of predicted blocks for sequence 1
                - 'block2_list': List of predicted blocks for sequence 2
                - 'seq_lengths': [batch_size, 2]
                - 'batch_size': B
            target: Ground truth adjacency [batch_size, max_seq_len, max_seq_len]
                   Can be sparse or dense
        
        Returns:
            Scalar loss value
        """
        block1_list = pred_blocks['block1_list']
        block2_list = pred_blocks['block2_list']
        seq_lengths = pred_blocks['seq_lengths']
        B = pred_blocks['batch_size']
        total_loss = 0
        total_elements = 0
        
        for b in range(B):
            L1 = seq_lengths[b, 0].item()
            L2 = seq_lengths[b, 1].item()
            # Get ground truth for this sample
            gt_sample = target[b]
            if gt_sample.is_sparse:
                gt_sample = gt_sample.to_dense()

            # Process Block 1
            if L1 > 0:
                pred_block1 = block1_list[b]  # [L1, L1]
                gt_block1 = gt_sample[:L1, :L1]  # [L1, L1]
                
                loss1, n1 = self._compute_focal_loss(pred_block1, gt_block1)
                total_loss += loss1
                total_elements += n1

            # Process Block 2
            if L2 > 0:
                pred_block2 = block2_list[b]  # [L2, L2]
                start = L1
                end = L1 + L2
                gt_block2 = gt_sample[start:end, start:end]  # [L2, L2]
                
                loss2, n2 = self._compute_focal_loss(pred_block2, gt_block2)
                total_loss += loss2
                total_elements += n2
        # Reduce
        if total_elements == 0:
            return torch. tensor(0.0, device=target.device, requires_grad=True)
        
        if self.reduction == 'mean':
            return total_loss / total_elements
        elif self.reduction == 'sum':
            return total_loss
        else:
            return total_loss
    
    def _compute_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Compute focal loss for a single block. 
        
        Args:
            pred: Predicted logits [H, W]
            target: Ground truth [H, W]
        
        Returns:
            loss: Summed focal loss for this block
            num_elements: Number of elements in this block
        """
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device), 0
        
        # Flatten
        pred_flat = pred.reshape(-1)
        target_flat = target. reshape(-1). float()
        
        # Compute binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_flat, 
            target_flat, 
            reduction='none'
        )
        
        # Compute focal loss components
        probs = torch.sigmoid(pred_flat)
        p_t = probs * target_flat + (1 - probs) * (1 - target_flat)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * target_flat + (1 - self. alpha) * (1 - target_flat)
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.sum(), pred_flat.numel()

# class GraphSAGESpatialEmbedding(nn.Module):
#     """
#     Pretrained GraphSAGE spatial embeddings for genes.
#     Loads pretrained embeddings and adds a random embedding for CLS token.
#     """
    
#     def __init__(self, freeze: bool, embedding_path: str):
#         """
#         Args:
#             freeze: Whether to freeze pretrained embeddings during training
#             embedding_path: Path to pretrained embedding file (. pkl)
#         """
#         super().__init__()
#         current_file = os.path.abspath(__file__)
#         pretrained_weights = pickle.load(open(embedding_path, "rb"))
        
#         # Add random embedding for CLS token
#         num_features = pretrained_weights. shape[1]
#         new_row = torch.randn(1, num_features)
#         updated_weights = torch.cat((pretrained_weights, new_row), dim=0)
        
#         self.emb = nn.Embedding.from_pretrained(updated_weights, freeze=freeze)

#     def forward(self, x: torch.Tensor) -> torch. Tensor:
#         """
#         Get spatial embeddings for token IDs.
        
#         Args:
#             x: Token IDs [batch_size, seq_len]
            
#         Returns:
#             Spatial embeddings [batch_size, seq_len, embedding_dim]
#         """
#         return self.emb(x)

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
    
# class MaskedMSE2DLoss(nn.Module):
#     def __init__(self, sample_balance = False):
#         super(MaskedMSE2DLoss, self).__init__()
#         self.sample_balance = sample_balance

#     def even_sample(self, pred, target):
#         # Flatten matrices for easier indexing
#         predictions_flat = pred.view(-1)
#         true_labels_flat = target.reshape(-1)
        
#         # Identify positive and negative indices
#         positive_indices = (true_labels_flat == 1).nonzero(as_tuple=True)[0]
#         negative_indices = (true_labels_flat == 0).nonzero(as_tuple=True)[0]
        
#         # Sample indices
#         num_samples = min(len(positive_indices), len(negative_indices))
#         selected_pos_indices = torch.randperm(len(positive_indices))[:num_samples]
#         selected_neg_indices = torch.randperm(len(negative_indices))[:num_samples]

#         sampled_indices = torch.cat((positive_indices[selected_pos_indices], negative_indices[selected_neg_indices]))
        
#         # Calculate loss using sampled indices
#         sampled_predictions = predictions_flat[sampled_indices]
#         sampled_true_labels = true_labels_flat[sampled_indices]
#         return sampled_predictions, sampled_true_labels


#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         # Here, the mask represents the masked tokens (with value != -100)
#         # import pdb; pdb.set_trace()

#         # Ensure valid_target is in the correct shape for gather
#         if self.sample_balance:
#             valid_input, valid_target = self.even_sample(input, target)
#             mask_num = len(valid_target)
#             try:
#                 spa_loss = F.binary_cross_entropy_with_logits(valid_input, valid_target.float(), reduction='mean')
#                 # spa_loss = loss / mask_num
#                 if torch.isnan(spa_loss) or torch.isinf(spa_loss):
#                     raise ValueError("Loss is NaN or Inf")
#             except:
#                 print("Loss computation failed, assigning default loss value.")
#                 spa_loss = torch.tensor(0.5, device=valid_input.device)

#         return spa_loss
    



    
# class AdjacencyProjector(nn.Module):
#     def __init__(self, embedding_dim):
#         super(AdjacencyProjector, self).__init__()
#         # Assume output dimension of 1 because we want a single score
#         self.fc = nn.Linear(2 * embedding_dim, 1, bias=False)

#     def forward(self, E):
#         #B x L X dim
#         num_genes = E.size(1)
#         # Create pairwise combinations
#         E_i = E.unsqueeze(2).expand(-1, -1, num_genes, -1)##B X L X L X dim
#         E_j = E.unsqueeze(1).expand(-1, num_genes, -1, -1)##B X L X L X dim
#         pairs = torch.cat((E_i, E_j), dim=-1)
        
#         # Compute adjacency scores
#         adjacency_scores = self.fc(pairs).squeeze(-1)#B x L X L
#         return adjacency_scores


class GraphSAGESpatialEmbedding(nn.Module):
    def __init__(self, freeze, embedding_path):
        super().__init__()
        pretrained_weights = pickle.load(open(embedding_path, "rb"))
        #adding the cls random value
        num_features = pretrained_weights.shape[1]

        # Generate a new random row with the same number of columns
        new_row = torch.randn(1, num_features)  # Shape will be (1, 512)
        # import pdb; pdb.set_trace()
        # Concatenate the new row to the existing tensor along axis 0 (rows)
        # updated_weights = torch.cat((pretrained_weights, new_row), dim=0)
        # import pdb; pdb.set_trace()
        self.emb = nn.Embedding.from_pretrained(pretrained_weights, freeze=freeze)

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
                 lr: float, 
                 warmup: int, 
                 max_epochs: int,
                 context_length: int = 500,
                 pool: str = None,
                 bpp: bool = False,
                 bpp_scale: int = None,
                 ag_loss: bool = False,
                 mask_way: str = None,
                 outer_config: dict = None,
                 use_flash_attn: bool = False,
                 **kwargs
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
            use_flash_attn (bool): whether to use flash attention for transformer layers
        """
        super().__init__()
        # self.batch_length = None
        # print("effective_batch_size:", effective_batch_size)
        if outer_config["input_type"] == "pair":
            effective_batch_size = outer_config["batch_size"]*(outer_config["num_positives_per_query"]+outer_config["num_hard_negatives_per_query"]+outer_config["num_easy_negatives_per_query"])
        elif outer_config["input_type"] == "single":
            effective_batch_size = outer_config["batch_size"]
        self.encoder = SpaEncoder(dim=dim_model , 
                                  num_layers=nlayers, 
                                  groups=dim_model, 
                                  num_heads=nheads, 
                                  bpp_size = context_length, 
                                  bpp = bpp, 
                                  bpp_scale=bpp_scale, 
                                  flash_attn=use_flash_attn)
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
        #Remark
        # self.MDloss = MaskedMSE2DLoss(sample_balance = True)
        self.MDloss = MaskedMSE2DLoss()

        #pair loss with/wighout imbalance
        if outer_config["num_positives_per_query"] / (outer_config["num_hard_negatives_per_query"]+outer_config["num_easy_negatives_per_query"]) == 1:
            self.Pairloss = PairLoss()
        else:
            weight = (outer_config["num_hard_negatives_per_query"]+outer_config["num_easy_negatives_per_query"])/outer_config["num_positives_per_query"]
            self.Pairloss = PairLoss_with_weight(torch.tensor([1, weight]))
            
        self.save_hyperparameters()
        #initialize as 0 to get better proformance, to homoscedastic uncertainty
        self.log_sigma_class = nn.Parameter(torch.zeros(1))  # Log variance for classification
        self.log_sigma_reg1 = None    # Log variance for regression
        self.log_sigma_reg2 = None
        self.log_sigma_reg3 = None
    
    
        
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
        self.input_type = outer_config["input_type"]
        self.dim_model = dim_model
        self.nlayers = nlayers
        self.nheads = nheads
        self.bpp = bpp
        self.bpp_scale = bpp_scale
        self.running_max_len = 0




            
    def forward(self, x, adjmtx, attention_mask, token_type_ids, sequence_length, **kwargs):

        token_embedding = self.embeddings(x) # batch x (context_length) x dim_model
        #adding the token type embeddings
        #get the first sep site
        if token_embedding.shape[1] > self.running_max_len:
            self.running_max_len = token_embedding.shape[1]
            print(f"running max length: {self.running_max_len}")
        token_embedding += self.token_type_embeddings(token_type_ids)
        if self.outer_config["spatial_embedding"]:
            token_embedding += self.spatialembeds(x)

        transformer_output, attn_scores = self.encoder(token_embedding, False, attention_mask) # batch x (n_tokens) x dim_model
        #get the last layer of the model output
        prediction = self.classifier_head(transformer_output[-1])
        #get the pair-wise gene-gene interaction matrix
        #Remark
        dis_prediction = self.adjprojector(transformer_output[-1], sequence_length)
        self.attentions = attn_scores
        if self.input_type == "pair":
            pair_prediction = self.pair_head(transformer_output[-1][:,0]) # the first token embeddings as input and predict whether two cells are paired
        elif self.input_type == "single":
            pair_prediction = None

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
        # import pdb; pdb.set_trace()
        #skip the None batch, which means the gene-gene matrix is sum as 0
        torch.cuda.synchronize()
        attention_mask = batch['attention_mask']
        batch['adjmtx'] = batch['adjmtx'].to(self.device)
        batch['indices'] = batch['indices'].to(self.device)
        if self.input_type == "pair":
            batch["pair_label"] = batch["pair_label"].to(self.device)
            pos_num = (batch["pair_label"] == 1).sum().to(torch.float32)
            neg_num = (batch["pair_label"] == 0).sum().to(torch.float32)
            self.log(f'train_num_positives', pos_num, sync_dist=True, prog_bar=False, reduce_fx='sum')
            self.log(f'train_num_negatives', neg_num, sync_dist=True, prog_bar=False, reduce_fx='sum')
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(self.device)
        batch["sequence_length"] = batch["sequence_length"].to(self.device)

        
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
                    #for spatial minibatch, including gene and cell spatial info
                    spa_batch = mini_batch_list[-1]
                    #for spatial jobs
                    spa_predictions, adjmtx, pair_predictions, pair_label = self.predict_spa(spa_batch)
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
                    #for mlm minibatch
                    batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens, self.cls_token, self.mask_token, self.sep_token, self.pad_token) #for mask token prediction
                    mlm_predictions, real_indices, spa_predictions, pair_predictions, pair_label, adjmtx = self.predict_exp(batch, "normalized_exp") #different task means different targets here
                    MLM_loss = self.MEloss(mlm_predictions, real_indices) # MLM loss
                    self.log(f'train_MLM_loss(normalized_exp)', MLM_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    log_sigma_mlm = getattr(self, f'log_sigma_reg1')[0]
                    self.log(f'log_sigma_reg(normalized_exp)', log_sigma_mlm, sync_dist=True, prog_bar=True, reduce_fx='mean')
                    total_loss += (torch.exp(-log_sigma_mlm) * MLM_loss + log_sigma_mlm) ##
                    Spa_loss = self.MDloss(spa_predictions, adjmtx)# spatial loss
                    if self.input_type == "pair":
                        Pair_loss = self.Pairloss(pair_predictions, pair_label) #Paired loss
                        self.log(f'train_Pair_loss', Pair_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
                        accuracy = self.get_acc(pair_predictions, pair_label)
                        self.log(f'train_pair_accuracy', accuracy, sync_dist=True, prog_bar=True, reduce_fx='mean')
                        log_sigma_pair = getattr(self, f'log_sigma_reg2')[0]
                        self.log(f'log_sigma_reg(pair)', log_sigma_pair, sync_dist=True, prog_bar=True, reduce_fx='mean')
                        total_loss += (torch.exp(-log_sigma_pair) * Pair_loss + log_sigma_pair)

                    self.log('train_Spa_loss', Spa_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                    total_loss += (torch.exp(-self.log_sigma_class[0]) * Spa_loss * self.linear_schedule_for_scale() + self.log_sigma_class[0]) ##
                    self.log('log_sigma_class', self.log_sigma_class[0], sync_dist=True, prog_bar=True, reduce_fx='mean')
                    self.log('train_total_loss', total_loss, sync_dist=True, prog_bar=True, reduce_fx='mean')

                # There's a corner case that returns NaN loss: when there are no masked tokens
                # however, likelihood of that is (1-p)^context_length

                #getting the total tokens

                self.total_tokens += attention_mask.sum()
                self.log('total_tokens', self.total_tokens, prog_bar=True, sync_dist=True, reduce_fx='sum')

                self.last_train_loss = total_loss.item()  
                
                # Check for NaN loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    encoder_state_dict = self.encoder.state_dict()
                    encoder_state_dict_cpu = {k: v.cpu() for k, v in encoder_state_dict.items()}
                    torch.save(encoder_state_dict, "/scratch/project_465001027/Spatialformer/data/encoder_state_dict.pth")
                #     #save everything
                    pickle.dump(to_cpu(batch), open("/scratch/project_465001027/Spatialformer/data/nanbatch.pkl", "wb"))
                    pickle.dump(to_cpu(mlm_predictions), open("/scratch/project_465001027/Spatialformer/data/mlm_predictions.pkl", "wb"))
                    pickle.dump(to_cpu(real_indices), open("/scratch/project_465001027/Spatialformer/data/real_indices.pkl", "wb"))
                    pickle.dump(to_cpu(attention_mask), open("/scratch/project_465001027/Spatialformer/data/attention_mask.pkl", "wb"))
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
        
        mini_size = int(batch_size/div)
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
        masked_indices = batch['masked_indices']
        attention_mask = batch['attention_mask']
        token_type_ids = batch["token_type_ids"]      
        adjmtx = batch['adjmtx']
        sequence_length = batch["sequence_length"]

        predictions = self.forward(masked_indices, adjmtx, attention_mask, token_type_ids, sequence_length)
        mlm_predictions = predictions['mlm_prediction']
        real_indices = torch.where(mask==MASK_TOKEN, real_indices, torch.tensor(-100, dtype=torch.long)).type(torch.int64)
        mlm_predictions = mlm_predictions.view(-1, self.hparams.n_tokens+self.hparams.n_atokens)
        real_indices = real_indices.view(-1)

        if self.outer_config["mini_batch"] == False:
            if self.input_type == "pair":
                pair_label = batch["pair_label"]
                spa_predictions = predictions['spa_prediction']
                pair_predictions = predictions['pair_prediction']
                return mlm_predictions, real_indices, spa_predictions, pair_predictions, pair_label, adjmtx
            else:
                pair_label = None
                spa_predictions = predictions['spa_prediction']
                pair_predictions = None
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
        # import pdb; pdb.set_trace()
        batch['adjmtx'] = batch['adjmtx'].to(self.device)
        batch['indices'] = batch['indices'].to(self.device)
        if self.input_type == "pair":
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
                    if self.input_type == "pair":
                        Pair_loss = self.Pairloss(pair_predictions, pair_label) #Paired loss
                        self.log(f'val_Pair_loss', Pair_loss, sync_dist=True, prog_bar=False, reduce_fx='mean')
                        accuracy = self.get_acc(pair_predictions, pair_label)
                        self.log(f'val_pair_accuracy', accuracy, sync_dist=True, prog_bar=True, reduce_fx='mean')
                        log_sigma_pair = getattr(self, f'log_sigma_reg2')[0]
                        total_loss += (torch.exp(-log_sigma_pair) * Pair_loss + log_sigma_pair)
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
                return hidden_repr, probabilities

        else:
            
            hidden_repr = [predictions["transformer_output"][i] for i in layers]
            pair_result_logit = predictions["pair_prediction"]
            #to probabilities
            probabilities = torch.nn.functional.softmax(pair_result_logit, dim=1)
            return hidden_repr, probabilities
    def get_pair(self, batch):
        indices = batch["indices"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        predictions = self.forward(indices, False, attention_mask, token_type_ids)
        pair_result_logit = predictions["pair_prediction"]
        return pair_result_logit


    def get_attention(self, batch, layers: List[int] = [11]):
        indices = batch["indices"].to(self.device)
        adjmtx = batch["adjmtx"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        new_batch_length = indices.shape[1]
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


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    """

    def __init__(self, optimizer: optim.Optimizer, warmup: int, max_epochs: int):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup: Number of warmup steps
            max_epochs: Total number of training epochs/steps
        """
        self. warmup = warmup
        self.max_num_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate for current step.
        
        Returns:
            List of learning rates for each parameter group
        """
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [max(1e-5, base_lr * lr_factor) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """
        Compute learning rate multiplier for given epoch.
        
        Args:
            epoch: Current epoch/step number
        
        Returns:
            Learning rate multiplier (0.0 to 1.0)
        """
        # Cosine annealing
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_epochs))
        
        # Linear warmup
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
            
        return lr_factor

