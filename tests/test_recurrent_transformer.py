import torch
import pytest
from torch import nn
param = pytest.mark.parametrize

from recurrent_transformer.recurrent_transformer import Attention, GatedTransition, RecurrentTransformer

def get_state_transition(kind, dim):
    if kind == 'gru':
        return nn.GRU(dim, dim, batch_first = True)

    if kind == 'gated':
        return GatedTransition(dim)

def assert_memory_close(naive_memory, tiled_memory):
    assert len(naive_memory) == len(tiled_memory)

    for naive_item, tiled_item in zip(naive_memory, tiled_memory):
        if naive_item is None or tiled_item is None:
            assert naive_item is None and tiled_item is None
            continue

        assert naive_item.shape == tiled_item.shape
        assert torch.allclose(naive_item, tiled_item, atol = 1e-5)

@param('recurrent', (False, True))
@param('block_size', (1, 2, 4))
@param('rel_pos_bias_kwargs', (
    dict(),
    dict(learned_alibi = False, distance_basis = True),
    dict(learned_alibi = True, distance_basis = True)
))
def test_recurrent_transformer(recurrent, block_size, rel_pos_bias_kwargs):

    model = RecurrentTransformer(
        num_tokens = 256,
        dim = 128,
        depth = 2,
        dim_head = 64,
        heads = 2,
        recurrent = recurrent,
        block_size = block_size,
        rel_pos_bias_kwargs = rel_pos_bias_kwargs
    )

    ids = torch.randint(0, 256, (2, 16))

    logits = model(ids)

    assert logits.shape == (2, 16, 256)

    loss = model(ids, return_loss = True)

    assert loss.numel() == 1

    loss.backward()

@param('batch', (1, 2))
@param('seq_len', (16, 12, 7))
@param('block_size', (1, 2, 4, 8))
@param('intra_block_rnn_kwargs', (
    dict(),
    dict(dim_hidden = 64, num_layers = 2)
))
@param('rel_pos_bias_kwargs', (
    dict(),
    dict(learned_alibi = False, distance_basis = True),
    dict(learned_alibi = True, distance_basis = True)
))
@param('transition_kind', ('none', 'gated', 'gru'))
def test_naive_vs_tiled_recurrent(batch, seq_len, block_size, intra_block_rnn_kwargs, rel_pos_bias_kwargs, transition_kind):

    model = RecurrentTransformer(
        num_tokens = 256,
        dim = 128,
        depth = 2,
        dim_head = 64,
        heads = 2,
        recurrent = True,
        block_size = block_size,
        intra_block_rnn_kwargs = intra_block_rnn_kwargs,
        rel_pos_bias_kwargs = rel_pos_bias_kwargs,
        state_transition = get_state_transition(transition_kind, 128)
    )

    ids = torch.randint(0, 256, (batch, seq_len))

    with torch.no_grad():
        naive_logits, naive_memories = model(ids, recurrent_mode = 'naive', return_memories = True)
        tiled_logits, tiled_memories = model(ids, recurrent_mode = 'tiled', return_memories = True)

    assert torch.allclose(naive_logits, tiled_logits, atol = 1e-5)

    for naive_memory, tiled_memory in zip(naive_memories, tiled_memories):
        assert_memory_close(naive_memory, tiled_memory)

    model(ids, return_loss = True, recurrent_mode = 'tiled').backward()

@param('batch', (1, 2))
@param('seq_len', (16, 12, 7))
@param('block_size', (1, 2, 4, 8))
@param('intra_block_rnn_kwargs', (
    dict(),
    dict(dim_hidden = 64, num_layers = 2)
))
@param('transition_kind', ('none', 'gated', 'gru'))
def test_naive_vs_tiled_recurrent_memory(batch, seq_len, block_size, intra_block_rnn_kwargs, transition_kind):

    attn = Attention(
        dim = 128,
        dim_head = 64,
        heads = 2,
        block_size = block_size,
        intra_block_rnn_kwargs = intra_block_rnn_kwargs,
        state_transition = get_state_transition(transition_kind, 128)
    )

    tokens = torch.randn(batch, seq_len, 128)

    with torch.no_grad():
        naive_out, naive_memory = attn.forward_naive_recurrent(tokens, return_memory = True)
        tiled_out, tiled_memory = attn.forward_tiled_recurrent(tokens, return_memory = True)

    assert torch.allclose(naive_out, tiled_out, atol = 1e-5)

    assert_memory_close(naive_memory, tiled_memory)

def test_attention_dim_inner_differs_from_dim():
    model = RecurrentTransformer(
        num_tokens = 256,
        dim = 64,
        depth = 2,
        dim_head = 32,
        heads = 4,
        recurrent = True
    )

    ids = torch.randint(0, 256, (2, 16))

    loss = model(ids, return_loss = True)
    loss.backward()

def test_shape_contracts():
    model = RecurrentTransformer(
        num_tokens = 256,
        dim = 128,
        depth = 2,
        dim_head = 64,
        heads = 2,
        recurrent = True
    )

    ids = torch.randint(0, 256, (2, 16))

    with pytest.raises(AssertionError):
        model(ids[0])

    with pytest.raises(AssertionError):
        model(ids, labels = torch.randint(0, 256, (2, 8)))

    attn = Attention(dim = 128, dim_head = 64, heads = 2)
    tokens = torch.randn(2, 5, 128)
    memory = (torch.randn(2, 2, 5, 16), torch.randn(2, 2, 5, 16), None)

    with pytest.raises(AssertionError):
        attn(tokens, memory = memory)

@param('block_size', (1, 2))
def test_generate(block_size):
    model = RecurrentTransformer(
        num_tokens = 256,
        dim = 128,
        depth = 2,
        dim_head = 64,
        heads = 2,
        recurrent = True,
        block_size = block_size
    )

    prompt = torch.randint(0, 256, (2, 8))

    sampled = model.generate(prompt, seq_len = 4, temperature = 0.)

    assert sampled.shape == (2, 4)

    sampled = model.generate(prompt[0], seq_len = 4, temperature = 0.)

    assert sampled.shape == (4,)

def test_generate_requires_recurrent():
    model = RecurrentTransformer(
        num_tokens = 256,
        dim = 128,
        depth = 2,
        dim_head = 64,
        heads = 2
    )

    with pytest.raises(AssertionError):
        model.generate(torch.randint(0, 256, (1, 8)), seq_len = 4)

@param('transition_kind', ('none', 'gated', 'gru'))
def test_generate_matches_full_recompute(transition_kind):
    torch.manual_seed(42)

    dim = 64

    model = RecurrentTransformer(
        num_tokens = 64,
        dim = dim,
        depth = 2,
        dim_head = 32,
        heads = 2,
        recurrent = True,
        state_transition = get_state_transition(transition_kind, dim)
    )

    prompt = torch.randint(0, 64, (2, 8))

    sampled = model.generate(prompt, seq_len = 4, temperature = 0.)

    # greedy reference recomputes the naive recurrent forward over the full prefix at each step

    ids = prompt

    with torch.no_grad():
        for _ in range(4):
            logits = model(ids, recurrent_mode = 'naive')
            next_token = logits[:, -1].argmax(dim = -1, keepdim = True)
            ids = torch.cat((ids, next_token), dim = -1)

    assert torch.equal(sampled, ids[:, prompt.shape[-1]:])
