import pytest
import torch
from torch import nn
from einops import einsum

from recurrent_transformer.recurrent_transformer import GatedTransition, RecurrentTransformer
from recurrent_transformer.triton_flash_tiled import TRITON_AVAILABLE, can_use_triton, tile_attn_update

param = pytest.mark.parametrize

pytestmark = pytest.mark.skipif(
    not TRITON_AVAILABLE or not torch.cuda.is_available(),
    reason = 'triton and cuda required'
)

def get_state_transition(kind, dim):
    if kind == 'gru':
        return nn.GRU(dim, dim, batch_first = True)

    if kind == 'gated':
        return GatedTransition(dim)

def reference_tile_update(q, k, v, m_old, l_old, o_old, scale, causal, q_offset, kv_offset, slopes = None, curves = None, max_dist = 0):
    seq_len_q, seq_len_k = q.shape[-2], k.shape[-2]
    heads = q.shape[1]

    rel_dist = (q_offset + torch.arange(seq_len_q, device = q.device))[:, None] - (kv_offset + torch.arange(seq_len_k, device = q.device))[None, :]

    valid = rel_dist >= 0
    bias = torch.zeros(heads, seq_len_q, seq_len_k, device = q.device, dtype = q.dtype)

    if curves is not None:
        valid = valid & (rel_dist < max_dist)
        bias = bias + curves[:, rel_dist.clamp(0, max_dist - 1)]

    if slopes is not None:
        bias = bias - slopes[:, None, None] * rel_dist.clamp(min = 0)

    bias = bias.masked_fill(~valid, 0.)

    sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * scale + bias

    if causal:
        sim = sim.masked_fill(rel_dist < 0, float('-inf'))

    m_new = torch.maximum(m_old, sim.amax(dim = -1, keepdim = True))
    alpha = (m_old - m_new).exp()
    p = (sim - m_new).exp()

    l_new = l_old * alpha + p.sum(dim = -1, keepdim = True)
    o_new = o_old * alpha + einsum(p, v, 'b h i j, b h j d -> b h i d')

    return m_new, l_new, o_new

def get_model(block_size, transition_kind, rel_pos_bias_kwargs = dict()):
    return RecurrentTransformer(
        num_tokens = 64,
        dim = 64,
        depth = 2,
        dim_head = 32,
        heads = 2,
        recurrent = True,
        block_size = block_size,
        rel_pos_bias_kwargs = rel_pos_bias_kwargs,
        state_transition = get_state_transition(transition_kind, 64)
    )

@param('causal', (False, True))
@param('bias_kind', ('none', 'both'))
def test_tile_attention_update_matches_reference(causal, bias_kind):
    torch.manual_seed(0)

    batch, heads, seq_len, dim = 2, 3, 8, 16
    scale = dim ** -0.5

    q = torch.randn(batch, heads, seq_len, dim, device = 'cuda', requires_grad = True)
    k = torch.randn(batch, heads, seq_len, dim, device = 'cuda', requires_grad = True)
    v = torch.randn(batch, heads, seq_len, dim, device = 'cuda', requires_grad = True)
    m_old = torch.randn(batch, heads, seq_len, 1, device = 'cuda', requires_grad = True)
    l_old = torch.rand(batch, heads, seq_len, 1, device = 'cuda') + 0.1
    o_old = torch.randn(batch, heads, seq_len, dim, device = 'cuda', requires_grad = True)

    slopes = curves = None
    max_dist = 0

    if bias_kind in ('alibi', 'both'):
        slopes = (torch.randn(heads, device = 'cuda') * 0.1).requires_grad_()

    if bias_kind in ('curves', 'both'):
        max_dist = 32
        curves = torch.randn(heads, max_dist, device = 'cuda', requires_grad = True)

    q_offset = seq_len + 3 if causal else seq_len
    kv_offset = 3 if causal else 0

    inputs = [t for t in (q, k, v, m_old, o_old, slopes, curves) if t is not None]

    reference_out = reference_tile_update(q, k, v, m_old, l_old, o_old, scale, causal, q_offset, kv_offset, slopes, curves, max_dist)
    reference_grads = torch.autograd.grad(sum(t.pow(2).sum() for t in reference_out), inputs)

    fused_out = tile_attn_update(q, k, v, m_old, l_old, o_old, scale = scale, causal = causal, q_offset = q_offset, kv_offset = kv_offset, slopes = slopes, curves = curves, max_dist = max_dist)
    fused_grads = torch.autograd.grad(sum(t.pow(2).sum() for t in fused_out), inputs)

    for reference_t, fused_t in zip(reference_out, fused_out):
        assert torch.allclose(reference_t, fused_t, atol = 1e-5)

    for reference_g, fused_g in zip(reference_grads, fused_grads):
        assert torch.allclose(reference_g, fused_g, rtol = 1e-4, atol = 1e-4)

def test_tile_attention_update_multiple_key_blocks():
    torch.manual_seed(0)

    batch, heads, seq_len, dim = 1, 2, 96, 32
    scale = dim ** -0.5

    q = torch.randn(batch, heads, seq_len, dim, device = 'cuda', requires_grad = True)
    k = torch.randn(batch, heads, seq_len, dim, device = 'cuda', requires_grad = True)
    v = torch.randn(batch, heads, seq_len, dim, device = 'cuda', requires_grad = True)
    m_old = torch.randn(batch, heads, seq_len, 1, device = 'cuda', requires_grad = True)
    l_old = torch.rand(batch, heads, seq_len, 1, device = 'cuda') + 0.1
    o_old = torch.randn(batch, heads, seq_len, dim, device = 'cuda', requires_grad = True)

    inputs = (q, k, v, m_old, o_old)

    reference_out = reference_tile_update(q, k, v, m_old, l_old, o_old, scale, False, seq_len, 0)
    reference_grads = torch.autograd.grad(sum(t.pow(2).sum() for t in reference_out), inputs)

    fused_out = tile_attn_update(q, k, v, m_old, l_old, o_old, scale = scale)
    fused_grads = torch.autograd.grad(sum(t.pow(2).sum() for t in fused_out), inputs)

    for reference_t, fused_t in zip(reference_out, fused_out):
        assert torch.allclose(reference_t, fused_t, atol = 1e-5)

    for reference_g, fused_g in zip(reference_grads, fused_grads):
        assert torch.allclose(reference_g, fused_g, rtol = 1e-4, atol = 1e-4)

@param('block_size', (1, 4))
def test_triton_tiled_matches_cpu_pathway(block_size):
    torch.manual_seed(42)

    model = get_model(block_size, 'none')
    ids = torch.randint(0, 64, (2, 8))

    cpu_loss = model(ids, return_loss = True)
    cpu_grads = torch.autograd.grad(cpu_loss, tuple(model.parameters()))

    model = model.cuda()
    model.zero_grad()

    cuda_loss = model(ids.cuda(), return_loss = True)
    cuda_grads = torch.autograd.grad(cuda_loss, tuple(model.parameters()))

    assert torch.allclose(cpu_loss, cuda_loss.cpu(), atol = 1e-4)

    for cpu_grad, cuda_grad in zip(cpu_grads, cuda_grads):
        assert torch.allclose(cpu_grad, cuda_grad.cpu(), rtol = 1e-3, atol = 1e-3)

@param('block_size', (1, 4))
def test_triton_tiled_matches_naive_pathway(block_size):
    torch.manual_seed(42)

    model = get_model(block_size, 'gru', dict(learned_alibi = True, distance_basis = True)).cuda()
    ids = torch.randint(0, 64, (2, 8), device = 'cuda')

    naive_loss = model(ids, return_loss = True, recurrent_mode = 'naive')
    naive_grads = torch.autograd.grad(naive_loss, tuple(model.parameters()))

    model.zero_grad()

    tiled_loss = model(ids, return_loss = True, recurrent_mode = 'tiled')
    tiled_grads = torch.autograd.grad(tiled_loss, tuple(model.parameters()))

    assert torch.allclose(naive_loss, tiled_loss, atol = 1e-5)

    for naive_grad, tiled_grad in zip(naive_grads, tiled_grads):
        assert torch.allclose(naive_grad, tiled_grad, rtol = 1e-4, atol = 1e-4)

def test_fused_tile_path_is_taken(monkeypatch):
    import recurrent_transformer.recurrent_transformer as rt

    calls = []
    original = rt.tile_attn_update

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(rt, 'tile_attn_update', spy)

    model = get_model(1, 'none').cuda()
    ids = torch.randint(0, 64, (1, 4), device = 'cuda')

    model(ids)

    assert len(calls) > 0

def test_can_use_triton_requires_cuda():
    assert can_use_triton(torch.zeros(1).cuda())
    assert not can_use_triton(torch.zeros(1))
