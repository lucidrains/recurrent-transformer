from __future__ import annotations
from functools import partial

import torch
from torch import nn, Tensor, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm, Linear, Sequential

import einx
from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from x_mlps_pytorch import MLP

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def inv_sqrt(n):
    return n ** -0.5

# lora

class LoRA(Module):
    def __init__(
        self,
        dim,
        dim_out,
        low_rank = 32
    ):
        super().__init__()
        assert low_rank < dim and low_rank < dim_out, f'low rank ({low_rank}) must be less than dim ({dim}) and dim_out ({dim_out})'

        self.up = LinearNoBias(dim, low_rank)
        self.down = LinearNoBias(low_rank, dim_out)

    def forward(self, x):
        return self.down(self.up(x))

# positional bias

# relative positional bias borrowed from Thinking Machines' Inkling model
# the learned alibi variant builds on ALiBi - Press et al. https://arxiv.org/abs/2108.12409

class RelativePositionBias(Module):
    def __init__(
        self,
        heads,
        num_distance_basis = 16,
        max_dist = 1024,
        learned_alibi = True,
        distance_basis = False
    ):
        super().__init__()
        assert learned_alibi or distance_basis

        self.max_dist = max_dist
        self.learned_alibi = learned_alibi
        self.distance_basis = distance_basis

        if learned_alibi:
            self.to_alibi_slopes = nn.Parameter(torch.ones(heads))

        if distance_basis:
            self.distance_bank = nn.Parameter(torch.randn(num_distance_basis, max_dist) * 0.02)
            self.to_head_weights = nn.Parameter(torch.zeros(heads, num_distance_basis))

    def forward(
        self,
        seq_len_q,
        seq_len_k = None
    ):
        device = next(self.parameters()).device
        seq_len_k = default(seq_len_k, seq_len_q)

        pos_q = torch.arange(seq_len_q, device = device) + (seq_len_k - seq_len_q)
        pos_k = torch.arange(seq_len_k, device = device)

        rel_dist = einx.subtract('i, j -> i j', pos_q, pos_k)

        valid_mask = rel_dist >= 0
        clamped_dist = rel_dist.clamp(min = 0)

        bias = 0.

        if self.distance_basis:
            valid_mask = valid_mask & (rel_dist < self.max_dist)
            clamped_dist = clamped_dist.clamp(max = self.max_dist - 1)

            curves = einsum(self.to_head_weights, self.distance_bank, 'h b, b r -> h r')
            bias = bias + curves[:, clamped_dist]

        if self.learned_alibi:
            bias = bias - rearrange(self.to_alibi_slopes, 'h -> h 1 1') * clamped_dist

        return bias.masked_fill(~valid_mask, 0.)

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        attn_gate = True,
        gate_low_rank = 32,
        use_value_mlp = True,
        value_mlp_expansion = 2.,
        block_size = 1,
        rel_pos_bias = True,
        rel_pos_bias_kwargs: dict = dict()
    ):
        super().__init__()
        self.scale = inv_sqrt(dim_head)
        self.block_size = block_size

        self.norm = RMSNorm(dim)
        dim_inner = dim_head * heads

        self.to_queries = LinearNoBias(dim, dim_inner)
        self.to_keys_values = LinearNoBias(dim, dim_inner * 2)

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # projection out

        self.to_out = LinearNoBias(dim_inner, dim)

        # attention gating - Jumper et al. AF2

        self.attn_gate = LoRA(dim, dim_inner, low_rank = gate_low_rank) if attn_gate else None

        # value mlp - post project but before aggregation + residual - shown to work well in a iclr 2026 paper for image restoration and apt for this setting

        self.value_mlp = MLP(dim_head, int(dim_head * value_mlp_expansion), dim_head) if use_value_mlp else None

        # relative positional bias

        self.rel_pos_bias = RelativePositionBias(heads = heads, **rel_pos_bias_kwargs) if rel_pos_bias else None

    def forward(
        self,
        tokens,
        memory: tuple[Tensor, Tensor] | None = None,
        return_memory = False,
        return_recurr_memory = False
    ):
        device = tokens.device

        residual = tokens

        tokens = self.norm(tokens)

        q = self.to_queries(tokens)
        k, v = self.to_keys_values(tokens).chunk(2, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        # qk rmsnorm

        q = self.q_norm(q)
        k = self.k_norm(k)

        # maybe residual value mlp (nonlinearity)

        if exists(self.value_mlp):
            v = v + self.value_mlp(v)

        # key value memories

        if exists(memory):
            mk, mv = memory
            k = cat((mk, k), dim = -2)
            v = cat((mv, v), dim = -2)

        # attention

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        i, j = sim.shape[-2:]

        if exists(self.rel_pos_bias):
            sim = sim + self.rel_pos_bias(i, j)

        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge heads

        agg = self.merge_heads(out)

        # maybe attention gate (nonlinearity)

        if exists(self.attn_gate):
            agg = agg * self.attn_gate(tokens).sigmoid()

        attn_out = self.to_out(agg)

        assert not (return_memory and return_recurr_memory)

        if not (return_memory or return_recurr_memory):
            return attn_out

        if return_memory:
            return attn_out, (k, v)

        # add the output to the residual and then reproject for the 'persistent' key value, key value derived from the output fed back in

        next_token = self.norm(attn_out + residual)

        next_k, next_v = self.to_keys_values(next_token).chunk(2, dim = -1)
        next_k, next_v = map(self.split_heads, (next_k, next_v))

        next_k = self.k_norm(next_k)

        if exists(self.value_mlp):
            next_v = next_v + self.value_mlp(next_v)

        if exists(memory):
            mk, mv = memory
            next_k = cat((mk, next_k), dim = -2)
            next_v = cat((mv, next_v), dim = -2)

        return attn_out, (next_k, next_v)

    def forward_naive_recurrent(
        self,
        tokens,
        return_memory = False,
    ):
        outs = []

        memory = None

        for block in tokens.split(self.block_size, dim = -2):
            out, memory = self(
                block,
                memory = memory,
                return_recurr_memory = True
            )

            outs.append(out)

        outs = cat(outs, dim = -2)

        if not return_memory:
            return outs

        return outs, memory

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
        ff_expansion = 4.,
        recurrent = False,
        block_size = 1,
        attn_gate = True,
        gate_low_rank = 32,
        use_value_mlp = True,
        rel_pos_bias = True,
        rel_pos_bias_kwargs: dict = dict()
    ):
        super().__init__()

        assert not recurrent or block_size >= 1

        self.recurrent = recurrent

        # embed

        self.token_emb = nn.Embedding(num_tokens, dim)

        # layers

        layers = ModuleList([])

        for layer_index in range(depth):
            layer_depth = layer_index + 1

            attn = Attention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                block_size = block_size,
                attn_gate = attn_gate,
                gate_low_rank = gate_low_rank,
                use_value_mlp = use_value_mlp,
                rel_pos_bias = rel_pos_bias,
                rel_pos_bias_kwargs = rel_pos_bias_kwargs
            )

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
        return_loss = False,
        labels = None
    ):
        if exists(labels):
            return_loss = True

        if return_loss and not exists(labels):
            ids, labels = ids[:, :-1], ids[:, 1:]

        # embed

        tokens = self.token_emb(ids)

        # layers

        for attn, ff in self.layers:

            forward_attn = attn.forward_naive_recurrent if self.recurrent else attn
            tokens = forward_attn(tokens) + tokens
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
