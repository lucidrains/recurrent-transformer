from __future__ import annotations
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm, Linear, Sequential

from einops import einsum, rearrange
from einops.layers.torch import Rearrange

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def inv_sqrt(n):
    return n ** -0.5

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = inv_sqrt(dim_head)

        self.norm = RMSNorm(dim)
        dim_inner = dim_head * heads

        self.to_queries = LinearNoBias(dim, dim_inner)
        self.to_keys_values = LinearNoBias(dim, dim_inner * 2)

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        tokens
    ):
        tokens = self.norm(tokens)

        q = self.to_queries(tokens)
        k, v = self.to_keys_values(tokens).chunk(2, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        q = self.q_norm(q)
        k = self.k_norm(k)

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        return self.to_out(out)

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(
    dim,
    expansion = 4.
):
    dim_inner = int(dim * expansion * 2 / 3)

    return Sequential(
        nn.RMSNorm(dim),
        Linear(dim, dim_inner * 2),
        GEGLU(),
        Linear(dim_inner, dim),
    )

# classes

class RecurrentTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_expansion = 4.
    ):
        super().__init__()

        # embed

        self.token_emb = nn.Embedding(num_tokens, dim)

        # layers

        layers = ModuleList([])

        for layer_index in range(depth):
            layer_depth = layer_index + 1

            attn = Attention(dim = dim, dim_head = dim_head, heads = heads)

            ff = FeedForward(dim = dim, expansion = ff_expansion)

            layers.append(ModuleList([attn, ff]))

            # depth residual scaling - Yang et al., following Wortsman init scheme

            attn_out, ff_out = attn.to_out, ff[-1]

            nn.init.normal_(attn_out.weight, std = inv_sqrt(2 * attn_out.in_features * layer_depth))
            nn.init.normal_(ff_out.weight, std = inv_sqrt(2 * ff_out.in_features * layer_depth))

        self.layers = layers

        # unembed

        self.to_logits = Sequential(
            nn.RMSNorm(dim),
            LinearNoBias(dim, num_tokens)
        )

    def forward(
        self,
        ids,
        return_loss = False
    ):
        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        # embed

        tokens = self.token_emb(ids)

        # layers

        for attn, ff in self.layers:
            tokens = attn(tokens) + tokens
            tokens = ff(tokens) + tokens

        # unembed

        logits = self.to_logits(tokens)

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels
        )

        return loss
