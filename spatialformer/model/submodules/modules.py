from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[0]
sys.path.append(p_path)

from layer_modules import DropPath, ScaleBiasLayer
from masked_batchnorm import MaskedBatchNorm1d
from masked_conv import MaskedConv1d
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input

def get_act_fn(activation):
    if activation == 'swish':
        return nn.SiLU()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'mish':
        return nn.Mish()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'elu':
        return nn.ELU()
    else:
        raise NotImplmentedError

class GLU(nn.Module):
    def __init__(self, dim: int, activation: str = 'sigmoid') -> None:
        super(GLU, self).__init__()
        self.dim = dim
        self.activation = get_act_fn(activation)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * self.activation(gate)

class Mlp(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        dropout: float = 0.1,
        bias : bool = True,
        activation: str = 'gelu'
    ) -> None:
        super(Mlp, self).__init__()

        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.act = get_act_fn(activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand, dim, bias=bias)
        # self.do2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        # x = self.do2(x)

        return x

class GLUMlp(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        dropout: float = 0.1,
        bias : bool = True,
        activation: str = 'gelu'
    ) -> None:
        super(GLUMlp, self).__init__()

        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.glu = GLU(dim=-1, activation=activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand // 2, dim, bias=bias)
        # self.do2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.glu(x)
        x = self.do1(x)
        x = self.ffn2(x)
        # x = self.do2(x)

        return x


class MaskedSoftmax(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # self.softmax = nn.Softmax(self.dim)

    def forward(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            # adder = (1.0 - mask.to(inputs.dtype)) * (
            #     torch.finfo(inputs.dtype).min
            # )

            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            # inputs += adder
            # import pdb; pdb.set_trace()
            inputs = inputs.masked_fill(~mask, torch.finfo(inputs.dtype).min)
        return F.softmax(inputs, dim=self.dim)#, dtype=torch.float32)

    
class AltAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.attn_score = None
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        # self.proj_drop = nn.Dropout(dropout)

    def forward(self, inputs, mask=None, alibi_bias=None):
        # import pdb; pdb.set_trace()
        qkv = self.qkv(inputs)# B x L X D -> B x L X 3D
        qkv = qkv.view(-1, inputs.shape[1], self.num_heads, self.dim * 3 // self.num_heads).permute(0, 2, 1, 3) # B X H X L X 3D/H
        q, k, v = qkv.split([self.dim // self.num_heads] * 3, dim=-1)# B X H X L X D/H

        if mask is not None:
            mask = mask[:, None, None, :] # B X 1 X 1 X L
        # import pdb; pdb.set_trace()
        #calculating the attention score
        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale
        
        #attention score add with bias
        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn += alibi_bias
        # import pdb; pdb.set_trace()
        attn = MaskedSoftmax(dim=-1)(attn, mask=mask)#.to(q.dtype)
        self.attn_score = attn
        attn = self.attn_drop(attn)

        x = attn @ v # B X H X L X D/H
        x = x.permute(0, 2, 1, 3).reshape(-1, inputs.shape[1], self.dim)# B X L X D
        #feedforward# B X L X D
        #getting the final Z
        x = self.proj(x)# B X L X D
        # x = self.proj_drop(x)
        return x

class FlashAttentionV2ALiBiWithMask(nn.Module):
    """
    FlashAttention v2 with ALiBi and padding mask support.
    """
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_score = None
        assert dim % num_heads == 0
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout = dropout
        # import pdb; pdb.set_trace()
        self.register_buffer(
            '_alibi_slopes_base',
            self._build_alibi_slopes(num_heads),
            persistent=False
        )
    
    def _build_alibi_slopes(self, num_heads):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(torch.log2(torch.tensor(n, dtype=torch.float32)) - 3)))
                return start * torch.pow(start, torch.arange(n, dtype=torch.float32))
            
            if n & (n - 1) == 0:
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** int(torch.log2(torch.tensor(n, dtype=torch.float32)).floor())
                return torch.cat([
                    get_slopes_power_of_2(closest_power_of_2),
                    get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
                ])
        # import pdb; pdb.set_trace()
        return get_slopes(num_heads)
        
    def _get_alibi_slopes(self, batch_size):
        """Expand slopes to batch size: (num_heads,) -> (B, num_heads)"""
        return self._alibi_slopes_base.unsqueeze(0).expand(batch_size, -1).contiguous()
    
    def forward(self, x, mask=None, alibi_bias=None):
        """
        Args:
            x: (B, N, C) input tensor
            mask: (B, N) boolean mask where True = valid, False = padding
        
        Returns:
            (B, N, C) output tensor
        """
        B, N, C = x.shape
        # import pdb; pdb.set_trace()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # import pdb; pdb.set_trace()
        if mask is None:
            # No mask - simple path
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
                alibi_slopes=self.alibi_slopes
            )
        else:
            # With mask - use variable length
            attn_output = self._forward_with_mask(q, k, v, mask, B, N)
        
        attn_output = attn_output.reshape(B, N, C)
        return self.proj(attn_output)
    
    def _forward_with_mask(self, q, k, v, mask, B, N):
        """Handle masked attention using variable length."""
        if mask.dtype != torch.bool:
            mask = mask.bool()
        # import pdb; pdb.set_trace()
        # Get sequence lengths
        seqlens = mask.sum(dim=1).to(torch.int32)
        max_seqlen = seqlens.max().item()
        # import pdb; pdb.set_trace()
        # Cumulative sequence lengths
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        cu_seqlens[1:] = seqlens.cumsum(dim=0)
        # import pdb; pdb.set_trace()
        # Flatten and filter
        q_flat = q.reshape(B * N, self.num_heads, self.head_dim)
        k_flat = k.reshape(B * N, self.num_heads, self.head_dim)
        v_flat = v.reshape(B * N, self.num_heads, self.head_dim)
        mask_flat = mask.reshape(B * N)
        # import pdb; pdb.set_trace()
        q_unpad = q_flat[mask_flat]
        k_unpad = k_flat[mask_flat]
        v_unpad = v_flat[mask_flat]
        # FlashAttention with ALiBi on unpadded sequences
        alibi_slopes = self._get_alibi_slopes(B)
        # import pdb; pdb.set_trace()
        attn_output_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False,
            alibi_slopes=alibi_slopes
        )
        # import pdb; pdb.set_trace()
        # Pad back
        attn_output = torch.zeros(B * N, self.num_heads, self.head_dim,
                                   dtype=q.dtype, device=q.device)
        attn_output[mask_flat] = attn_output_unpad
        attn_output = attn_output.reshape(B, N, self.num_heads, self.head_dim)
        # import pdb; pdb.set_trace()
        return attn_output


class AltBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, expand=4, attn_dropout=0.2, mlp_dropout=0.2, drop_path=0., activation='gelu', prenorm=True, flash_attn=False, **kwargs):
        super().__init__(**kwargs)
        self.flash_attn = flash_attn
        self.norm1 = nn.LayerNorm(dim)#MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        if not flash_attn:
            self.self_attn = AltAttention(dim=dim,num_heads=num_heads,dropout=attn_dropout)
        else:
            self.self_attn = FlashAttentionV2ALiBiWithMask(dim=dim,num_heads=num_heads,dropout=attn_dropout)

        self.drop1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)#MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.mlp = GLUMlp(dim, expand, dropout=mlp_dropout, activation=activation)
        self.drop2 = DropPath(drop_path)

        self.prenorm = prenorm
        self.attn_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        
    def forward(self, inputs, mask=None, alibi_bias=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        # import pdb; pdb.set_trace()
        x = self.self_attn(x,mask=mask,alibi_bias=alibi_bias)
        x = self.drop1(x)
        x = self.attn_scale(x)
        x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)
        attn_out = x

        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        x = x + attn_out
        if not self.prenorm:
            x = self.norm2(x)
        return x
        
class Conv1DBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=17,
                 groups=4,
                 dilation=1,
                 stride=1,
                 conv_dropout=0.0,
                 mlp_dropout=0.0,
                 drop_path=0.0,
                 expand=4,
                 activation='swish',
                 prenorm=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.prenorm = prenorm
        self.stride = stride

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.glu = GLU(dim=-1, activation=activation)
        self.expand_conv = nn.Linear(dim, 2*dim)
        self.conv = MaskedConv1d(dim, dim, kernel_size=kernel_size, groups=groups)
        self.conv_norm = MaskedBatchNorm1d(dim, momentum=0.05)
        self.conv_act = get_act_fn(activation)
        self.conv_proj = nn.Linear(dim, dim)
        self.mlp = GLUMlp(dim, expand, mlp_dropout, activation=activation)
        self.conv_dropout = nn.Dropout(conv_dropout)
        # self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.drop1 = DropPath(drop_path)
        self.drop2 = DropPath(drop_path)
        self.conv_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)

    def compute_mask(self, inputs, mask=None):
      if mask is not None:
        if self.stride > 1:
          mask = mask[:,::self.stride]
      return mask

    def forward(self, inputs, mask=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.expand_conv(x)
        x = self.glu(x)
        x = x.permute(0,2,1)
        x = self.conv(x,mask=mask)
        mask = self.compute_mask(inputs,mask)
        x = self.conv_norm(x,mask=mask)
        x = self.conv_act(x)
        x = self.conv_dropout(x)
        x = x.permute(0,2,1)
        x = self.conv_proj(x)
        x = self.drop1(x)
        x = self.conv_scale(x)
        if self.stride == 1:
            x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)

        conv_out = x
        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        # x = self.mlp_dropout(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        if self.stride == 1:
            x = x + conv_out
        if not self.prenorm:
            x = self.norm2(x)
        return x