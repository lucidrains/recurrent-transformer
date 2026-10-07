"""
fused triton kernel for the exact tiled recurrent attention update (Algorithm 2 of the recurrent transformer paper)

a newly revealed persistent key / value tile is folded into the online softmax running statistics (m, l, o)
of an already available block of queries, without materializing the score matrix - the backward is fused as well
"""

from __future__ import annotations

import torch
from torch import Tensor

from torch_einops_utils.shape import shape, size

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None

    TRITON_AVAILABLE = False

TRITON_ENABLED = True

def exists(v):
    return v is not None

def triton_available() -> bool:
    return TRITON_AVAILABLE and TRITON_ENABLED

def can_use_triton(t: Tensor) -> bool:
    return triton_available() and t.is_cuda

# kernels

if TRITON_AVAILABLE:

    NEG_INF = tl.constexpr(float("-inf"))

    @triton.jit
    def _tile_logits(
        q, k_tile, q_pos, kv_pos, mask_n,
        slope, max_dist, CURVES, h, scale,
        CAUSAL: tl.constexpr,
        HAS_ALIBI: tl.constexpr,
        HAS_CURVES: tl.constexpr,
        USE_DOT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        if USE_DOT:
            sim = tl.dot(q, tl.trans(k_tile), input_precision = "ieee") * scale
        else:
            sim = tl.sum(q[:, None, :] * k_tile[None, :, :], axis = 2) * scale

        if CAUSAL:
            valid = (q_pos[:, None] >= kv_pos[None, :]) & mask_n[None, :]
        else:
            valid = tl.broadcast_to(mask_n[None, :], (BLOCK_M, BLOCK_N))

        dist = q_pos[:, None] - kv_pos[None, :]

        if HAS_ALIBI or HAS_CURVES:
            bias_valid = valid & (dist >= 0)
            bias = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

            if HAS_CURVES:
                bias_valid = bias_valid & (dist < max_dist)
                cdist = tl.minimum(tl.maximum(dist, 0), max_dist - 1)
                bias += tl.load(CURVES + h * max_dist + cdist, mask = bias_valid, other = 0.0)

            if HAS_ALIBI:
                bias -= slope * tl.maximum(dist, 0).to(tl.float32)

            sim = sim + tl.where(bias_valid, bias, 0.0)

        return tl.where(valid, sim, NEG_INF), valid, dist

    @triton.jit
    def _tile_attn_fwd_kernel(
        Q, K, V, MO, LO, OO, MN, LN, ON,
        SLOPES, CURVES,
        scale, q_offset, kv_offset, max_dist,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_stat_b, stride_stat_h, stride_stat_m,
        stride_o_b, stride_o_h, stride_o_m, stride_o_d,
        Lq, Lk, D, H,
        CAUSAL: tl.constexpr,
        HAS_ALIBI: tl.constexpr,
        HAS_CURVES: tl.constexpr,
        USE_DOT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        mask_m = offs_m < Lq
        mask_d = offs_d < D
        mask_md = mask_m[:, None] & mask_d[None, :]

        # queries

        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask = mask_md, other = 0.0)

        # running statistics

        m = tl.load(MO + b * stride_stat_b + h * stride_stat_h + offs_m * stride_stat_m, mask = mask_m, other = NEG_INF)
        l = tl.load(LO + b * stride_stat_b + h * stride_stat_h + offs_m * stride_stat_m, mask = mask_m, other = 0.0)
        acc = tl.load(OO + b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d, mask = mask_md, other = 0.0)

        if HAS_ALIBI:
            slope = tl.load(SLOPES + h)
        else:
            slope = 0.0

        q_pos = q_offset + offs_m

        for n0 in range(0, Lk, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < Lk
            mask_nd = mask_n[:, None] & mask_d[None, :]

            k_tile = tl.load(K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask = mask_nd, other = 0.0)
            v_tile = tl.load(V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask = mask_nd, other = 0.0)

            sim, _, _ = _tile_logits(q, k_tile, q_pos, kv_offset + offs_n, mask_n, slope, max_dist, CURVES, h, scale, CAUSAL, HAS_ALIBI, HAS_CURVES, USE_DOT, BLOCK_M, BLOCK_N)

            # online softmax merge

            m_tile = tl.max(sim, axis = 1)
            m_new = tl.maximum(m, m_tile)
            m_safe = tl.where(m_new == NEG_INF, 0.0, m_new)

            alpha = tl.exp(m - m_safe)
            p = tl.exp(sim - m_safe[:, None])

            l = l * alpha + tl.sum(p, axis = 1)

            if USE_DOT:
                acc = acc * alpha[:, None] + tl.dot(p, v_tile, input_precision = "ieee")
            else:
                acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v_tile[None, :, :], axis = 1)

            m = m_new

        tl.store(MN + b * stride_stat_b + h * stride_stat_h + offs_m * stride_stat_m, m, mask = mask_m)
        tl.store(LN + b * stride_stat_b + h * stride_stat_h + offs_m * stride_stat_m, l, mask = mask_m)
        tl.store(ON + b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d, acc, mask = mask_md)

    @triton.jit
    def _tile_attn_bwd_kernel(
        Q, K, V, MO, LO, OO,
        DO, DL, DM,
        DQ, DK, DV, DMO, DLO, DOO,
        SLOPES, CURVES, DSLOPES, DCURVES,
        scale, q_offset, kv_offset, max_dist,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_stat_b, stride_stat_h, stride_stat_m,
        stride_o_b, stride_o_h, stride_o_m, stride_o_d,
        Lq, Lk, D, H,
        CAUSAL: tl.constexpr,
        HAS_ALIBI: tl.constexpr,
        HAS_CURVES: tl.constexpr,
        USE_DOT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_am_lane = tl.arange(0, BLOCK_N)

        mask_m = offs_m < Lq
        mask_d = offs_d < D
        mask_md = mask_m[:, None] & mask_d[None, :]

        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask = mask_md, other = 0.0)
        m = tl.load(MO + b * stride_stat_b + h * stride_stat_h + offs_m * stride_stat_m, mask = mask_m, other = NEG_INF)
        l = tl.load(LO + b * stride_stat_b + h * stride_stat_h + offs_m * stride_stat_m, mask = mask_m, other = 0.0)
        o_old = tl.load(OO + b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d, mask = mask_md, other = 0.0)

        d_m = tl.load(DM + b * stride_stat_b + h * stride_stat_h + offs_m * stride_stat_m, mask = mask_m, other = 0.0)
        d_l = tl.load(DL + b * stride_stat_b + h * stride_stat_h + offs_m * stride_stat_m, mask = mask_m, other = 0.0)
        d_o = tl.load(DO + b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d, mask = mask_md, other = 0.0)

        if HAS_ALIBI:
            slope = tl.load(SLOPES + h)
        else:
            slope = 0.0

        q_pos = q_offset + offs_m

        # first pass - final tile max over the whole key / value tile

        m_tile = tl.full((BLOCK_M,), NEG_INF, dtype=tl.float32)

        for n0 in range(0, Lk, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < Lk
            mask_nd = mask_n[:, None] & mask_d[None, :]

            k_tile = tl.load(K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask = mask_nd, other = 0.0)

            sim, _, _ = _tile_logits(q, k_tile, q_pos, kv_offset + offs_n, mask_n, slope, max_dist, CURVES, h, scale, CAUSAL, HAS_ALIBI, HAS_CURVES, USE_DOT, BLOCK_M, BLOCK_N)
            m_tile = tl.maximum(m_tile, tl.max(sim, axis = 1))

        m_new = tl.maximum(m, m_tile)
        m_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
        alpha = tl.exp(m - m_safe)

        take_old = m >= m_tile

        # second pass - recompute relative to the final max, accumulating gradients

        dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        sum_p_dp = tl.zeros((BLOCK_M,), dtype=tl.float32)

        has_argmax = tl.zeros((BLOCK_M,), dtype=tl.int1)
        argmax_index = tl.zeros((BLOCK_M,), dtype=tl.int32)

        for n0 in range(0, Lk, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < Lk
            mask_nd = mask_n[:, None] & mask_d[None, :]

            k_tile = tl.load(K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask = mask_nd, other = 0.0)
            v_tile = tl.load(V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask = mask_nd, other = 0.0)

            sim, valid, dist = _tile_logits(q, k_tile, q_pos, kv_offset + offs_n, mask_n, slope, max_dist, CURVES, h, scale, CAUSAL, HAS_ALIBI, HAS_CURVES, USE_DOT, BLOCK_M, BLOCK_N)

            p = tl.where(mask_m[:, None], tl.exp(sim - m_safe[:, None]), 0.0)

            d_p = tl.sum(d_o[:, None, :] * v_tile[None, :, :], axis = 2) + d_l[:, None]
            d_sim = p * d_p

            sum_p_dp += tl.sum(p * d_p, axis = 1)
            d_sim = tl.where(mask_m[:, None], d_sim, 0.0)

            # gradient path through the tile max - remember the argmax position per row

            chunk_max = tl.max(sim, axis = 1)
            local_argmax = tl.argmax(sim, axis = 1)
            is_argmax = (chunk_max == m_tile) & (~has_argmax) & mask_m

            argmax_index = tl.where(is_argmax, tl.sum(tl.where(offs_am_lane[None, :] == local_argmax[:, None], offs_n[None, :], 0), axis = 1), argmax_index)
            has_argmax = has_argmax | is_argmax

            # query / key / value gradients

            if USE_DOT:
                dq += tl.dot(d_sim, k_tile, input_precision = "ieee") * scale
                dk = tl.dot(tl.trans(d_sim), q, input_precision = "ieee") * scale
                dv = tl.dot(tl.trans(p), d_o, input_precision = "ieee")
            else:
                dq += tl.sum(d_sim[:, :, None] * k_tile[None, :, :], axis = 1) * scale
                dk = tl.sum(d_sim[:, :, None] * q[:, None, :], axis = 0) * scale
                dv = tl.sum(p[:, :, None] * d_o[:, None, :], axis = 0)

            # relative positional bias parameter gradients

            if HAS_ALIBI:
                dslope = -tl.sum(tl.sum(d_sim * tl.maximum(dist, 0).to(tl.float32), axis = 1), axis = 0)
                tl.atomic_add(DSLOPES + h, dslope)

            if HAS_CURVES:
                cdist = tl.minimum(tl.maximum(dist, 0), max_dist - 1)
                tl.atomic_add(DCURVES + h * max_dist + cdist, d_sim, mask = valid & (dist >= 0) & (dist < max_dist))

            tl.atomic_add(DK + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, dk, mask = mask_nd)
            tl.atomic_add(DV + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, dv, mask = mask_nd)

        d_renorm = tl.sum(d_o * o_old, axis = 1) + d_l * l

        d_m_old = d_renorm * alpha + tl.where(take_old, d_m - d_renorm * alpha - sum_p_dp, 0.0)
        d_m_tile = tl.where(take_old, 0.0, d_m - d_renorm * alpha - sum_p_dp)

        tl.store(DMO + b * stride_stat_b + h * stride_stat_h + offs_m * stride_stat_m, d_m_old, mask = mask_m)
        tl.store(DLO + b * stride_stat_b + h * stride_stat_h + offs_m * stride_stat_m, d_l * alpha, mask = mask_m)
        tl.store(DOO + b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d, d_o * alpha[:, None], mask = mask_md)

        # route the tile max gradient to the argmax position

        argmax_mask = has_argmax & mask_m
        k_argmax = tl.load(K + b * stride_kb + h * stride_kh + argmax_index[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask = argmax_mask[:, None] & mask_d[None, :], other = 0.0)
        dq += tl.where(argmax_mask[:, None], d_m_tile[:, None] * k_argmax * scale, 0.0)

        dk_argmax = tl.where(argmax_mask[:, None], d_m_tile[:, None] * q * scale, 0.0)
        tl.atomic_add(DK + b * stride_kb + h * stride_kh + argmax_index[:, None] * stride_kn + offs_d[None, :] * stride_kd, dk_argmax, mask = argmax_mask[:, None] & mask_d[None, :])

        if HAS_ALIBI or HAS_CURVES:
            dist_argmax = q_pos - (kv_offset + argmax_index)

            if HAS_ALIBI:
                dslope = -tl.sum(tl.where(argmax_mask, d_m_tile * tl.maximum(dist_argmax, 0).to(tl.float32), 0.0), axis = 0)
                tl.atomic_add(DSLOPES + h, dslope)

            if HAS_CURVES:
                cdist_argmax = tl.minimum(tl.maximum(dist_argmax, 0), max_dist - 1)
                tl.atomic_add(DCURVES + h * max_dist + cdist_argmax, d_m_tile, mask = argmax_mask & (dist_argmax >= 0) & (dist_argmax < max_dist))

        tl.store(DQ + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, dq, mask = mask_md)

# autograd function

class TileAttentionUpdate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        m_old: Tensor,
        l_old: Tensor,
        o_old: Tensor,
        scale: float,
        causal: bool,
        q_offset: int,
        kv_offset: int,
        slopes: Tensor | None,
        curves: Tensor | None,
        max_dist: int,
    ):
        B, H, Lq, D = shape(q, 'b h i d')
        Lk = size(k, 'b h [j] d')

        m_new = torch.empty_like(m_old)
        l_new = torch.empty_like(l_old)
        o_new = torch.empty_like(o_old)

        block_m = min(64, triton.next_power_of_2(Lq))
        block_n = min(64, triton.next_power_of_2(Lk))
        block_d = triton.next_power_of_2(D)

        use_dot = block_m >= 16 and block_n >= 16 and block_d >= 16

        dummy = q
        slopes_arg = slopes if exists(slopes) else dummy
        curves_arg = curves if exists(curves) else dummy

        grid = (triton.cdiv(Lq, block_m), B * H)

        _tile_attn_fwd_kernel[grid](
            q, k, v, m_old, l_old, o_old, m_new, l_new, o_new,
            slopes_arg, curves_arg,
            scale, q_offset, kv_offset, max_dist,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            m_old.stride(0), m_old.stride(1), m_old.stride(2),
            o_old.stride(0), o_old.stride(1), o_old.stride(2), o_old.stride(3),
            Lq, Lk, D, H,
            CAUSAL = causal,
            HAS_ALIBI = exists(slopes),
            HAS_CURVES = exists(curves),
            USE_DOT = use_dot,
            BLOCK_M = block_m,
            BLOCK_N = block_n,
            BLOCK_D = block_d,
            num_warps = 4 if use_dot else 1,
            num_stages = 1,
        )

        ctx.save_for_backward(q, k, v, m_old, l_old, o_old)
        ctx.scale = scale
        ctx.causal = causal
        ctx.q_offset = q_offset
        ctx.kv_offset = kv_offset
        ctx.slopes = slopes
        ctx.curves = curves
        ctx.max_dist = max_dist
        ctx.block_sizes = (block_m, block_n, block_d)

        return m_new, l_new, o_new

    @staticmethod
    def backward(ctx, dm_new, dl_new, do_new):
        q, k, v, m_old, l_old, o_old = ctx.saved_tensors

        if dm_new is None:
            dm_new = torch.zeros_like(m_old)
        if dl_new is None:
            dl_new = torch.zeros_like(l_old)
        if do_new is None:
            do_new = torch.zeros_like(o_old)

        dm_new = dm_new.contiguous()
        dl_new = dl_new.contiguous()
        do_new = do_new.contiguous()

        slopes, curves = ctx.slopes, ctx.curves

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dm_old = torch.empty_like(m_old)
        dl_old = torch.empty_like(l_old)
        do_old = torch.empty_like(o_old)

        dslopes = torch.zeros_like(slopes) if exists(slopes) else q
        dcurves = torch.zeros_like(curves) if exists(curves) else q

        B, H, Lq, D = shape(q, 'b h i d')
        Lk = size(k, 'b h [j] d')
        block_m, block_n, block_d = ctx.block_sizes

        use_dot = block_m >= 16 and block_n >= 16 and block_d >= 16

        grid = (triton.cdiv(Lq, block_m), B * H)

        _tile_attn_bwd_kernel[grid](
            q, k, v, m_old, l_old, o_old,
            do_new, dl_new, dm_new,
            dq, dk, dv, dm_old, dl_old, do_old,
            slopes if exists(slopes) else q,
            curves if exists(curves) else q,
            dslopes, dcurves,
            ctx.scale, ctx.q_offset, ctx.kv_offset, ctx.max_dist,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            m_old.stride(0), m_old.stride(1), m_old.stride(2),
            o_old.stride(0), o_old.stride(1), o_old.stride(2), o_old.stride(3),
            Lq, Lk, D, H,
            CAUSAL = ctx.causal,
            HAS_ALIBI = exists(slopes),
            HAS_CURVES = exists(curves),
            USE_DOT = use_dot,
            BLOCK_M = block_m,
            BLOCK_N = block_n,
            BLOCK_D = block_d,
            num_warps = 4 if use_dot else 1,
            num_stages = 1,
        )

        return (
            dq, dk, dv, dm_old, dl_old, do_old,
            None, None, None, None,
            dslopes if exists(slopes) else None,
            dcurves if exists(curves) else None,
            None,
        )

def tile_attn_update(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    m_old: Tensor,
    l_old: Tensor,
    o_old: Tensor,
    *,
    scale: float,
    causal: bool = False,
    q_offset: int = 0,
    kv_offset: int = 0,
    slopes: Tensor | None = None,
    curves: Tensor | None = None,
    max_dist: int = 0,
):
    """
    online softmax update of the running attention statistics (m, l, o) of a query
    tile with a newly revealed key / value tile, the fused UpdateTile of the paper

    shapes
        q:      (b h i d)
        k, v:   (b h j d)
        m_old:  (b h i 1)
        l_old:  (b h i 1)
        o_old:  (b h i d)
        slopes: (h)          optional learned alibi slopes
        curves: (h max_dist) optional distance basis curves
    """

    assert triton_available(), "triton is not available"
    assert q.is_cuda, "triton kernel requires cuda tensors"

    q = q.to(torch.float32).contiguous()
    k = k.to(torch.float32).contiguous()
    v = v.to(torch.float32).contiguous()
    m_old = m_old.to(torch.float32).contiguous()
    l_old = l_old.to(torch.float32).contiguous()
    o_old = o_old.to(torch.float32).contiguous()

    if exists(slopes):
        slopes = slopes.to(torch.float32).contiguous()

    if exists(curves):
        curves = curves.to(torch.float32).contiguous()

    return TileAttentionUpdate.apply(
        q, k, v, m_old, l_old, o_old,
        float(scale), bool(causal), int(q_offset), int(kv_offset),
        slopes, curves, int(max_dist),
    )
