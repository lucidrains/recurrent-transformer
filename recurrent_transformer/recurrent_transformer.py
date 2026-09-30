from __future__ import annotations
from functools import partial
from math import ceil

import torch
from torch import nn, Tensor, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm, Linear, Sequential

import einx
from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from torch_einops_utils import clamp, pack_with_inverse, pad_at_dim_to_multiple, temp_eval
from torch_einops_utils.shape import assert_shape, shape

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

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, eps = 1e-10):
    if temperature == 0.:
        return t.argmax(dim = dim)

    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, num_kept: int | None = None, frac_num_tokens = 0.1):
    num_tokens = shape(logits, '... l').l

    num_kept = default(num_kept, ceil(frac_num_tokens * num_tokens))

    threshold = logits.topk(num_kept, dim = -1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float('-inf'))

# intra block rnn

class IntraBlockRNN(Module):
    def __init__(
        self,
        dim,
        block_size,
        dim_hidden = None,
        num_layers = 1,
        rnn_type = nn.GRU,
        rnn_kwargs: dict = dict()
    ):
        super().__init__()
        self.block_size = block_size

        dim_hidden = default(dim_hidden, dim)
        self.rnn = rnn_type(dim, dim_hidden, num_layers = num_layers, batch_first = True, **rnn_kwargs)

        self.to_out = LinearNoBias(dim_hidden, dim) if dim_hidden != dim else nn.Identity()

    def forward(self, tokens):
        block_size = self.block_size
        batch = shape(tokens, 'b ...').b

        # pad to multiple of block size

        tokens, inverse_pad = pad_at_dim_to_multiple(tokens, block_size, dim = -2)

        # run the rnn causally over each block of tokens in parallel

        tokens = rearrange(tokens, 'b (n s) d -> (b n) s d', s = block_size)
        hiddens, _ = self.rnn(tokens)
        hiddens = rearrange(hiddens, '(b n) s d -> b (n s) d', b = batch)

        return inverse_pad(self.to_out(hiddens))

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
        use_intra_block_rnn = True,
        intra_block_rnn_kwargs: dict = dict(),
        rel_pos_bias = True,
        rel_pos_bias_kwargs: dict = dict()
    ):
        super().__init__()
        self.scale = inv_sqrt(dim_head)
        self.block_size = block_size
        self.heads = heads
        self.dim_head = dim_head

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

        self.attn_gate = LoRA(dim_inner, dim_inner, low_rank = gate_low_rank) if attn_gate else None

        # value mlp - only for persistent key values

        self.value_mlp = MLP(dim_head, int(dim_head * value_mlp_expansion), dim_head) if use_value_mlp else None

        # intra block rnn - gives the later tokens of a block a causal recurrent summary of the earlier ones, defaults to a gru

        self.intra_block_rnn = IntraBlockRNN(dim, block_size, **intra_block_rnn_kwargs) if use_intra_block_rnn and block_size > 1 else None

        # relative positional bias

        self.rel_pos_bias = RelativePositionBias(heads = heads, **rel_pos_bias_kwargs) if rel_pos_bias else None

    # helper functions for queries, keys, values

    def process_queries(
        self,
        tokens
    ):
        q = self.to_queries(tokens)
        q = self.split_heads(q)
        q = self.q_norm(q)
        return q

    def process_key_values(
        self,
        tokens,
        persistent = False
    ):
        k, v = self.to_keys_values(tokens).chunk(2, dim = -1)

        k, v = map(self.split_heads, (k, v))

        # rmsnorm

        k = self.k_norm(k)

        # maybe residual value mlp (nonlinearity), only for the persistent values

        if persistent and exists(self.value_mlp):
            v = v + self.value_mlp(v)

        return k, v

    def merge_and_combine_heads(
        self,
        attend_out
    ):
        agg = self.merge_heads(attend_out)

        # maybe attention gate (nonlinearity)

        if exists(self.attn_gate):
            agg = agg * self.attn_gate(agg).sigmoid()

        attn_out = self.to_out(agg)

        return attn_out

    # the proposed tiling approach, based on the author's previous flash inference paper

    def forward_tiled_recurrent(
        self,
        tokens,
        return_memory = False
    ):
        block_size = self.block_size

        # pad to multiple of block size

        tokens, inverse_pad = pad_at_dim_to_multiple(tokens, block_size, dim = -2)

        seq_len = shape(tokens, 'b n d').n

        residual = tokens

        tokens = self.norm(tokens)

        if exists(self.intra_block_rnn):
            tokens = tokens + self.intra_block_rnn(tokens)

        # queries and temporary key / values

        q = self.process_queries(tokens)
        k, v = self.process_key_values(tokens)

        # the queries and temporary keys and values can be processed all at once for the initial partial online row outputs, causal within each block

        q_blk = rearrange(q, 'b h (n s) d -> b h n s d', s = block_size)
        k_blk = rearrange(k, 'b h (n s) d -> b h n s d', s = block_size)
        v_blk = rearrange(v, 'b h (n s) d -> b h n s d', s = block_size)

        row_sim = einsum(q_blk, k_blk, 'b h n i d, b h n j d -> b h n i j') * self.scale

        if exists(self.rel_pos_bias):
            row_sim = row_sim + rearrange(self.rel_pos_bias(block_size, block_size), 'h i j -> 1 h 1 i j')

        if block_size > 1:
            causal_mask = torch.ones((block_size, block_size), dtype = torch.bool, device = row_sim.device).triu(1)
            row_sim = row_sim.masked_fill(causal_mask, -torch.finfo(row_sim.dtype).max)

        row_max = row_sim.amax(dim = -1, keepdim = True)
        row_exp = (row_sim - row_max).exp()

        row_sums = row_exp.sum(dim = -1, keepdim = True)
        row_nums = einsum(row_exp, v_blk, 'b h n i j, b h n j d -> b h n i d')

        # flatten back to the sequence, the row max is cloned as it is updated in place below

        row_max = rearrange(row_max, 'b h n s 1 -> b h (n s) 1').clone()
        row_sums = rearrange(row_sums, 'b h n s 1 -> b h (n s) 1')
        row_nums = rearrange(row_nums, 'b h n s d -> b h (n s) d')

        # accumulate output

        attn_outs = []
        persist_ks = []
        persist_vs = []

        def get_attn_out(block_index):
            token_slice = slice(block_index * block_size, (block_index + 1) * block_size)

            token_num = row_nums[..., token_slice, :].clone()
            token_sums = row_sums[..., token_slice, :].clone()

            token_out = token_num / token_sums
            return self.merge_and_combine_heads(token_out)

        # now process the tiles, one block at a time

        num_blocks = seq_len // block_size

        for block_index in range(num_blocks):

            token_slice = slice(block_index * block_size, (block_index + 1) * block_size)

            # calculate persistent key value

            persist_token_out = get_attn_out(block_index)

            # get the next persistent / recurrent - key value

            normed = self.norm(persist_token_out + residual[:, token_slice])

            next_persist_k, next_persist_v = self.process_key_values(normed, persistent = True)

            persist_ks.append(next_persist_k)
            persist_vs.append(next_persist_v)

            attn_outs.append(persist_token_out)

            # if last block, nothing online left to update

            i = block_index + 1

            if i == num_blocks:
                continue

            # trick for getting the tile size to be processed, taught to me by gemini

            tile_size = i & -i

            # tile slices

            q_tile_slice = slice(i * block_size, clamp((i + tile_size) * block_size, hi = seq_len))
            kv_tile_slice = slice(i - tile_size, i)

            # get the q, k, v for the tile

            tq = q[..., q_tile_slice, :]
            tk = cat(persist_ks[kv_tile_slice], dim = -2)
            tv = cat(persist_vs[kv_tile_slice], dim = -2)

            # calculate tile

            tile_sim = einsum(tq, tk, 'b h i d, b h j d -> b h i j') * self.scale

            if exists(self.rel_pos_bias):
                q_tile_len, = shape(tq, 'b h [i] d')
                kv_tile_len, = shape(tk, 'b h [j] d')
                tile_sim = tile_sim + self.rel_pos_bias(q_tile_len, q_tile_len + kv_tile_len)[..., :kv_tile_len]

            tile_max = tile_sim.amax(dim = -1, keepdim = True)

            # clone the old partials

            old_row_max = row_max[..., q_tile_slice, :].clone()
            old_row_sums = row_sums[..., q_tile_slice, :].clone()
            old_row_nums = row_nums[..., q_tile_slice, :].clone()

            next_row_max = torch.maximum(old_row_max, tile_max)

            tile_sim_exp = (tile_sim - next_row_max).exp()

            tile_num = einsum(tile_sim_exp, tv, 'b h i j, b h j d -> b h i d')

            tile_sums = tile_sim_exp.sum(dim = -1, keepdim = True)

            # update partials in place, as the tile of queries is a contiguous slice

            renorm = (old_row_max - next_row_max).exp()

            row_sums[..., q_tile_slice, :] = old_row_sums * renorm + tile_sums
            row_nums[..., q_tile_slice, :] = old_row_nums * renorm + tile_num
            row_max[..., q_tile_slice, :] = next_row_max

        # return

        attn_outs = inverse_pad(cat(attn_outs, dim = -2))

        if not return_memory:
            return attn_outs

        persist_k = cat(persist_ks, dim = -2)
        persist_v = cat(persist_vs, dim = -2)

        memory = (inverse_pad(persist_k), inverse_pad(persist_v))

        return attn_outs, memory

    def forward_naive_recurrent(
        self,
        tokens,
        memory: tuple[Tensor, Tensor] | None = None,
        return_memory = False
    ):

        # accumulate output

        outs = []

        # one block at a time, the memory returned is always the 'persistent' / recurrent key values

        for block in tokens.split(self.block_size, dim = -2):
            out, memory = self(
                block,
                memory = memory,
                return_recurr_memory = True
            )

            outs.append(out)

        # post process

        outs = cat(outs, dim = -2)

        # return

        if not return_memory:
            return outs

        return outs, memory

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

        if exists(self.intra_block_rnn):
            tokens = tokens + self.intra_block_rnn(tokens)

        # queries

        q = self.process_queries(tokens)

        # key values

        k, v = self.process_key_values(tokens)

        # key value memories

        if exists(memory):
            mk, mv = memory

            assert_shape([(tokens, 'b n d'), (mk, 'b h m dh'), (mv, 'b h m dh')], h = self.heads, dh = self.dim_head)

            k = cat((mk, k), dim = -2)
            v = cat((mv, v), dim = -2)

        # attention

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        i, j = shape(sim, 'b h [i j]')

        if exists(self.rel_pos_bias):
            sim = sim + self.rel_pos_bias(i, j)

        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        attend_out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge heads and combine

        attn_out = self.merge_and_combine_heads(attend_out)

        assert not (return_memory and return_recurr_memory)

        if not (return_memory or return_recurr_memory):
            return attn_out

        if return_memory:
            return attn_out, (k, v)

        # add the output to the residual and then reproject for the 'persistent' key value, key value derived from the output fed back in

        next_token = self.norm(attn_out + residual)

        next_k, next_v = self.process_key_values(next_token, persistent = True)

        if exists(memory):
            mk, mv = memory
            next_k = cat((mk, next_k), dim = -2)
            next_v = cat((mv, next_v), dim = -2)

        return attn_out, (next_k, next_v)

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
        recurrent_mode = 'tiled',
        block_size = 1,
        attn_gate = True,
        gate_low_rank = 32,
        use_value_mlp = True,
        use_intra_block_rnn = True,
        intra_block_rnn_kwargs: dict = dict(),
        rel_pos_bias = True,
        rel_pos_bias_kwargs: dict = dict()
    ):
        super().__init__()

        assert block_size >= 1
        assert recurrent_mode in ('naive', 'tiled')

        self.recurrent = recurrent
        self.recurrent_mode = recurrent_mode

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
                use_intra_block_rnn = use_intra_block_rnn,
                intra_block_rnn_kwargs = intra_block_rnn_kwargs,
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
            RMSNorm(dim),
            LinearNoBias(dim, num_tokens)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @assert_shape({'ids': 'b n', 'labels': 'b n'})
    def forward(
        self,
        ids,
        return_loss = False,
        labels = None,
        recurrent_mode = None,
        memories = None,
        return_memories = False
    ):
        recurrent_mode = default(recurrent_mode, self.recurrent_mode)

        assert recurrent_mode in ('naive', 'tiled')

        # decoding with cached persistent key values is only possible under the naive recurrent mode

        if exists(memories):
            assert self.recurrent and recurrent_mode == 'naive', 'decoding requires recurrent = True and recurrent_mode = "naive"'
            assert not return_loss

        if exists(labels):
            return_loss = True

        if return_loss and not exists(labels):
            ids, labels = ids[:, :-1], ids[:, 1:]

        # embed

        tokens = self.token_emb(ids)

        # layers

        next_memories = []

        for layer_index, (attn, ff) in enumerate(self.layers):
            memory = memories[layer_index] if exists(memories) else None

            if self.recurrent and recurrent_mode == 'naive':
                attn_out, next_memory = attn.forward_naive_recurrent(tokens, memory = memory, return_memory = True)
            elif self.recurrent:
                attn_out, next_memory = attn.forward_tiled_recurrent(tokens, return_memory = True)
            else:
                attn_out, next_memory = attn(tokens), None

            next_memories.append(next_memory)

            tokens = attn_out + tokens
            tokens = ff(tokens) + tokens

        # unembed

        logits = self.to_logits(tokens)

        if return_memories:
            return logits, next_memories

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels
        )

        return loss

    @temp_eval
    @torch.no_grad()
    @assert_shape('... n')
    def generate(
        self,
        prompt,
        seq_len,
        temperature = 1.,
        filter_fn = top_k,
        filter_kwargs = dict(frac_num_tokens = 0.1)
    ):
        assert self.recurrent, 'recurrent must be enabled to decode'

        prompt, inverse_pack = pack_with_inverse(prompt, '* n')

        prompt = prompt.to(self.device)

        # prefill prompt, tiled by default

        logits, memories = self.forward(prompt, return_memories = True)

        # decode one token at a time, naive recurrent with persistent key value memories

        out = []

        for _ in range(seq_len):
            filtered_logits = filter_fn(logits[:, -1], **filter_kwargs)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            out.append(sampled)

            logits, memories = self.forward(
                sampled,
                recurrent_mode = 'naive',
                memories = memories,
                return_memories = True
            )

        return inverse_pack(cat(out, dim = -1))
