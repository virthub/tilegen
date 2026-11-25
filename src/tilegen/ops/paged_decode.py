# src/tilegen/ops/paged_decode.py
#
# Batched paged-KV decode kernel

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    Q, K_cache, V_cache, block_tables, seq_lens, O, sm_scale,
    NUM_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    r = tl.program_id(0)                       # one program per request
    seq = tl.load(seq_lens + r)
    num_blocks = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE

    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, BLOCK_SIZE)
    offs_g = tl.arange(0, RATIO)               # group-local head index

    q_base = Q + r * (NUM_HEADS * HEAD_DIM)
    o_base = O + r * (NUM_HEADS * HEAD_DIM)

    for hkv in range(0, NUM_KV_HEADS):
        # Load this KV head's assigned Q group [RATIO, D].
        q = tl.load(q_base + (hkv * RATIO + offs_g)[:, None] * HEAD_DIM + offs_d[None, :])

        m = tl.full([RATIO], float("-inf"), dtype=tl.float32)
        l = tl.zeros([RATIO], dtype=tl.float32)
        acc = tl.zeros([RATIO, HEAD_DIM], dtype=tl.float32)

        for blk in range(0, num_blocks):
            bid = tl.load(block_tables + r * MAX_BLOCKS + blk)
            mask = blk * BLOCK_SIZE + offs_s < seq

            k_base = K_cache + bid * (BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM)
            k = tl.load(
                k_base + offs_s[:, None] * (NUM_KV_HEADS * HEAD_DIM) + hkv * HEAD_DIM + offs_d[None, :],
                mask=mask[:, None], other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * sm_scale          # [RATIO, BLOCK_SIZE]
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m - m_ij)
            l = l * alpha + l_ij
            acc = acc * alpha[:, None]

            v_base = V_cache + bid * (BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM)
            v = tl.load(
                v_base + offs_s[:, None] * (NUM_KV_HEADS * HEAD_DIM) + hkv * HEAD_DIM + offs_d[None, :],
                mask=mask[:, None], other=0.0,
            )
            acc += tl.dot(p.to(v.dtype), v)
            m = m_ij

        l = 1.0 / l
        acc = acc * l[:, None]
        if IS_BF16:
            tl.store(o_base + (hkv * RATIO + offs_g)[:, None] * HEAD_DIM + offs_d[None, :],
                     acc.to(tl.bfloat16))
        else:
            tl.store(o_base + (hkv * RATIO + offs_g)[:, None] * HEAD_DIM + offs_d[None, :],
                     acc.to(tl.float16))


def paged_decode_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_layout,
    sm_scale=None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Batched paged decode: one launch for all requests.

    Args:
        q: ``[num_reqs, num_heads, D]`` (1 query token per request).
        k_cache/v_cache: ``[num_blocks, block_size, num_kv_heads, D]``.
        block_tables: ``[num_reqs, max_blocks]`` int32.
        seq_lens: ``[num_reqs]`` int32.
        kv_layout: :class:`PagedKVLayout` (block_size, num_kv_heads, head_dim).
    """
    num_reqs, num_heads, D = q.shape
    num_kv_heads = kv_layout.num_kv_heads
    block_size = kv_layout.block_size
    assert num_heads % num_kv_heads == 0
    ratio = num_heads // num_kv_heads
    max_blocks = block_tables.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    q = q.contiguous()
    o = torch.empty_like(q)
    is_bf16 = q.dtype == torch.bfloat16

    _paged_decode_kernel[(num_reqs,)](
        q, k_cache, v_cache, block_tables, seq_lens, o, sm_scale,
        NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, RATIO=ratio,
        HEAD_DIM=D, BLOCK_SIZE=block_size, MAX_BLOCKS=max_blocks,
        IS_BF16=is_bf16, num_warps=num_warps, num_stages=num_stages,
    )
    return o


@triton.jit
def _paged_decode_int4_kernel(
    Q, K_cache, V_cache, K_scale, V_scale, block_tables, seq_lens, O, sm_scale,
    NUM_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr, IS_BF16: tl.constexpr,
):
    """Split-dot INT4 decode (NG == 1): avoids ``tl.interleave`` (Triton 3.5
    miscompiles ``interleave`` on the loaded uint8 nibbles when followed by MMA),
    splitting the QK^T/PV into even/odd channel halves."""
    r = tl.program_id(0)
    seq = tl.load(seq_lens + r)
    num_blocks = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE

    offs_d2 = tl.arange(0, HEAD_DIM // 2)
    offs_s = tl.arange(0, BLOCK_SIZE)
    offs_g = tl.arange(0, RATIO)

    D2: tl.constexpr = HEAD_DIM // 2
    NG: tl.constexpr = HEAD_DIM // GROUP_SIZE             # groups per head
    GS2: tl.constexpr = GROUP_SIZE // 2                   # group size in packed dim
    offs_group = offs_d2 // GS2                           # [D2] group index (packed)

    for hkv in range(0, NUM_KV_HEADS):
        qb = Q + r * (NUM_HEADS * HEAD_DIM) + (hkv * RATIO + offs_g)[:, None] * HEAD_DIM
        q_even = tl.load(qb + offs_d2[None, :] * 2)        # [RATIO, D//2] even
        q_odd = tl.load(qb + offs_d2[None, :] * 2 + 1)     # [RATIO, D//2] odd
        m = tl.full([RATIO], float("-inf"), dtype=tl.float32)
        l = tl.zeros([RATIO], dtype=tl.float32)
        acc_even = tl.zeros([RATIO, D2], dtype=tl.float32)
        acc_odd = tl.zeros([RATIO, D2], dtype=tl.float32)

        for blk in range(0, num_blocks):
            bid = tl.load(block_tables + r * MAX_BLOCKS + blk)
            mask = blk * BLOCK_SIZE + offs_s < seq

            kb = K_cache + bid * (BLOCK_SIZE * NUM_KV_HEADS * D2) \
                + offs_s[:, None] * (NUM_KV_HEADS * D2) + hkv * D2
            kp = tl.load(kb + offs_d2[None, :], mask=mask[:, None], other=0)
            k_lo = (kp & 0x0F).to(tl.float16)
            k_hi = ((kp >> 4) & 0x0F).to(tl.float16)
            sc = tl.load(
                K_scale + bid * (BLOCK_SIZE * NUM_KV_HEADS * NG)
                + offs_s[:, None] * (NUM_KV_HEADS * NG) + hkv * NG + offs_group[None, :],
                mask=mask[:, None], other=0.0,
            ).to(tl.float32)                                # [BLOCK_SIZE, D2] gather
            k_lo_dq = ((k_lo.to(tl.float32) - 8.0) * sc).to(tl.float16)
            k_hi_dq = ((k_hi.to(tl.float32) - 8.0) * sc).to(tl.float16)
            qk = (tl.dot(q_even, tl.trans(k_lo_dq)) + tl.dot(q_odd, tl.trans(k_hi_dq))) * sm_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m - m_ij)
            l = l * alpha + l_ij
            acc_even = acc_even * alpha[:, None]
            acc_odd = acc_odd * alpha[:, None]

            vb = V_cache + bid * (BLOCK_SIZE * NUM_KV_HEADS * D2) \
                + offs_s[:, None] * (NUM_KV_HEADS * D2) + hkv * D2
            vp = tl.load(vb + offs_d2[None, :], mask=mask[:, None], other=0)
            v_lo = (vp & 0x0F).to(tl.float32)
            v_hi = ((vp >> 4) & 0x0F).to(tl.float32)
            vsc = tl.load(
                V_scale + bid * (BLOCK_SIZE * NUM_KV_HEADS * NG)
                + offs_s[:, None] * (NUM_KV_HEADS * NG) + hkv * NG + offs_group[None, :],
                mask=mask[:, None], other=0.0,
            ).to(tl.float32)
            v_lo_dq = ((v_lo - 8.0) * vsc).to(tl.float16)
            v_hi_dq = ((v_hi - 8.0) * vsc).to(tl.float16)
            acc_even += tl.dot(p.to(tl.float16), v_lo_dq)
            acc_odd += tl.dot(p.to(tl.float16), v_hi_dq)
            m = m_ij

        l = 1.0 / l
        acc_even = acc_even * l[:, None]
        acc_odd = acc_odd * l[:, None]
        ob = O + r * (NUM_HEADS * HEAD_DIM) + (hkv * RATIO + offs_g)[:, None] * HEAD_DIM
        if IS_BF16:
            tl.store(ob + offs_d2[None, :] * 2, acc_even.to(tl.bfloat16))
            tl.store(ob + offs_d2[None, :] * 2 + 1, acc_odd.to(tl.bfloat16))
        else:
            tl.store(ob + offs_d2[None, :] * 2, acc_even.to(tl.float16))
            tl.store(ob + offs_d2[None, :] * 2 + 1, acc_odd.to(tl.float16))


def paged_decode_int4_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_layout,
    group_size: int = 64,
    sm_scale=None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Batched paged decode with in-kernel INT4 dequant (packed KV + scales).

    Uses a split-dot formulation (even/odd channel halves) with a per-group scale
    gather, avoiding ``tl.interleave`` (miscompiled by Triton 3.5 before MMA).
    """
    num_reqs, num_heads, D = q.shape
    num_kv_heads = kv_layout.num_kv_heads
    block_size = kv_layout.block_size
    assert num_heads % num_kv_heads == 0
    assert D % group_size == 0
    ratio = num_heads // num_kv_heads
    max_blocks = block_tables.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    q = q.contiguous()
    o = torch.empty_like(q)
    is_bf16 = q.dtype == torch.bfloat16

    _paged_decode_int4_kernel[(num_reqs,)](
        q, k_cache, v_cache, k_scale, v_scale, block_tables, seq_lens, o, sm_scale,
        NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, RATIO=ratio,
        HEAD_DIM=D, BLOCK_SIZE=block_size, MAX_BLOCKS=max_blocks, GROUP_SIZE=group_size,
        IS_BF16=is_bf16, num_warps=num_warps, num_stages=num_stages,
    )
    return o


@triton.jit
def _paged_decode_tokens_kernel(
    Q, K_pool, V_pool, req_to_token, seq_lens, O, sm_scale,
    NUM_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_CTX: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """Batched decode with token-indexed (radix-tree) KV gather."""
    r = tl.program_id(0)
    seq = tl.load(seq_lens + r)
    num_blocks = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE

    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, BLOCK_SIZE)
    offs_g = tl.arange(0, RATIO)

    q_base = Q + r * (NUM_HEADS * HEAD_DIM)
    o_base = O + r * (NUM_HEADS * HEAD_DIM)

    for hkv in range(0, NUM_KV_HEADS):
        q = tl.load(q_base + (hkv * RATIO + offs_g)[:, None] * HEAD_DIM + offs_d[None, :])
        m = tl.full([RATIO], float("-inf"), dtype=tl.float32)
        l = tl.zeros([RATIO], dtype=tl.float32)
        acc = tl.zeros([RATIO, HEAD_DIM], dtype=tl.float32)

        for blk in range(0, num_blocks):
            mask = blk * BLOCK_SIZE + offs_s < seq
            toks = tl.load(
                req_to_token + r * MAX_CTX + blk * BLOCK_SIZE + offs_s,
                mask=mask, other=0,
            )                                             # [BLOCK_SIZE] token indices

            k = tl.load(
                K_pool + toks[:, None] * (NUM_KV_HEADS * HEAD_DIM) + hkv * HEAD_DIM + offs_d[None, :],
                mask=mask[:, None], other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m - m_ij)
            l = l * alpha + l_ij
            acc = acc * alpha[:, None]

            v = tl.load(
                V_pool + toks[:, None] * (NUM_KV_HEADS * HEAD_DIM) + hkv * HEAD_DIM + offs_d[None, :],
                mask=mask[:, None], other=0.0,
            )
            acc += tl.dot(p.to(v.dtype), v)
            m = m_ij

        l = 1.0 / l
        acc = acc * l[:, None]
        if IS_BF16:
            tl.store(o_base + (hkv * RATIO + offs_g)[:, None] * HEAD_DIM + offs_d[None, :],
                     acc.to(tl.bfloat16))
        else:
            tl.store(o_base + (hkv * RATIO + offs_g)[:, None] * HEAD_DIM + offs_d[None, :],
                     acc.to(tl.float16))


def paged_decode_tokens_forward(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale=None,
    block_size: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Batched token-indexed decode (SGLang radix-tree KV, single launch).

    Args:
        q: ``[num_reqs, num_heads, D]`` (1 query token per request).
        k_pool/v_pool: ``[num_tokens, num_kv_heads, D]`` (flat token pool).
        req_to_token: ``[num_reqs, max_ctx]`` int32 token indices.
        seq_lens: ``[num_reqs]`` int32.
    """
    num_reqs, num_heads, D = q.shape
    num_kv_heads = k_pool.shape[1]
    assert num_heads % num_kv_heads == 0
    ratio = num_heads // num_kv_heads
    max_ctx = req_to_token.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    q = q.contiguous()
    o = torch.empty_like(q)
    is_bf16 = q.dtype == torch.bfloat16

    _paged_decode_tokens_kernel[(num_reqs,)](
        q, k_pool, v_pool, req_to_token, seq_lens, o, sm_scale,
        NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, RATIO=ratio,
        HEAD_DIM=D, BLOCK_SIZE=block_size, MAX_CTX=max_ctx, IS_BF16=is_bf16,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


__all__ = ["paged_decode_forward", "paged_decode_int4_forward", "paged_decode_tokens_forward"]
