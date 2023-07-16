import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import base64
import json
import math
from dataclasses import dataclass
import numpy as np

# forked and simplified from nanoGPT
    
class ConvSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        #self.kqv_proj = nn.Conv2d(config.n_embd, 3 * config.n_embd, 1, bias=config.bias)#.to(device)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # batch_size, test_env_d*perception_channel_multiplier, y_dim, x_dim

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        #print(q.shape)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if True:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y
    

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=False)
        self.attn = ConvSelfAttention(config)
        #self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=0.0, bias=True, batch_first=True)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=False)
        self.mlp = MLP(config)

    def forward(self, x):
        #att, _ = self.attn(self.ln_1(x))
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class CATConfig:
    n_layer: int = 1
    n_head: int = 4
    n_embd: int = 48 #16
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster  


class CAModelTransformer(nn.Module):

    def __init__(self, env_d):
        super().__init__()
        config = CATConfig(n_embd=env_d*3)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=False),
            out_proj = nn.Conv2d(config.n_embd, env_d, 1)
        ))
        nn.init.zeros_(self.transformer.out_proj.weight)
        nn.init.zeros_(self.transformer.out_proj.bias)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
            
    def fold_to_kernels(self, input_env):
        kernel_size = 3
        kernel_count = kernel_size*kernel_size
        neighbors = F.unfold(F.pad(input_env, (1,1,1,1), mode='circular'), (kernel_size, kernel_size) )
        return rearrange(neighbors, "b (c t) p -> (b p) t c", t=kernel_count)

    def forward(self, in_x):
        device = in_x.device

        # (b, t, n_embd)
        x = self.fold_to_kernels(in_x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = reduce(x, "(b h w) t c -> b c h w", "sum", b=in_x.shape[0], h=in_x.shape[2], w=in_x.shape[3])
        # a hack? maybe.
        x = self.transformer.out_proj(x)

        return x
