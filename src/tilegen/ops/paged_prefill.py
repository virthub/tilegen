# src/tilegen/ops/paged_prefill.py
#
# Chunked paged-KV prefill kernel

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_prefill_kernel(
    Q, K_cache, V_cache, block_tables, seq_lens, O, sm_scale,
    NUM_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr, Q_LEN: tl.constexpr, IS_CAUSAL: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid_m = tl.program_id(0)   # query block
    r = tl.program_id(1)       # request
    seq = tl.load(seq_lens + r)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query positions [BLOCK_M]
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_g = tl.arange(0, RATIO)

    q_base = Q + r * (Q_LEN * NUM_HEADS * HEAD_DIM)
    o_base = O + r * (Q_LEN * NUM_HEADS * HEAD_DIM)

    for hkv in range(0, NUM_KV_HEADS):
        q = tl.load(
            q_base + offs_m[:, None, None] * (NUM_HEADS * HEAD_DIM)
            + (hkv * RATIO + offs_g)[None, :, None] * HEAD_DIM
            + offs_d[None, None, :],
            mask=offs_m[:, None, None] < Q_LEN, other=0.0,
        )
        q_flat = tl.reshape(q, (BLOCK_M * RATIO, HEAD_DIM))
        qpos = tl.reshape(offs_m[:, None] + tl.zeros([1, RATIO], dtype=tl.int32),
                          (BLOCK_M * RATIO,))

        m = tl.full([BLOCK_M * RATIO], float("-inf"), dtype=tl.float32)
        l = tl.zeros([BLOCK_M * RATIO], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M * RATIO, HEAD_DIM], dtype=tl.float32)

        if IS_CAUSAL:
            hi = tl.minimum((pid_m + 1) * BLOCK_M, seq)
        else:
            hi = seq
        nblk = (hi + BLOCK_SIZE - 1) // BLOCK_SIZE

        for blk in range(0, nblk):
            bid = tl.load(block_tables + r * MAX_BLOCKS + blk)
            mask = blk * BLOCK_SIZE + offs_n < seq
            k = tl.load(
                K_cache + bid * (BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM)
                + offs_n[:, None] * (NUM_KV_HEADS * HEAD_DIM)
                + hkv * HEAD_DIM + offs_d[None, :],
                mask=mask[:, None], other=0.0,
            )
            qk = tl.dot(q_flat, tl.trans(k)) * sm_scale   # [BLOCK_M*RATIO, BLOCK_SIZE]
            if IS_CAUSAL:
                kvpos = blk * BLOCK_SIZE + offs_n
                qk = tl.where(qpos[:, None] >= kvpos[None, :], qk, float("-inf"))
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m - m_ij)
            l = l * alpha + l_ij
            acc = acc * alpha[:, None]

            v = tl.load(
                V_cache + bid * (BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM)
                + offs_n[:, None] * (NUM_KV_HEADS * HEAD_DIM)
                + hkv * HEAD_DIM + offs_d[None, :],
                mask=mask[:, None], other=0.0,
            )
            acc += tl.dot(p.to(v.dtype), v)
            m = m_ij

        l = 1.0 / l
        acc = acc * l[:, None]
        acc3 = tl.reshape(acc, (BLOCK_M, RATIO, HEAD_DIM))
        if IS_BF16:
            tl.store(
                o_base + offs_m[:, None, None] * (NUM_HEADS * HEAD_DIM)
                + (hkv * RATIO + offs_g)[None, :, None] * HEAD_DIM
                + offs_d[None, None, :],
                acc3.to(tl.bfloat16), mask=offs_m[:, None, None] < Q_LEN,
            )
        else:
            tl.store(
                o_base + offs_m[:, None, None] * (NUM_HEADS * HEAD_DIM)
                + (hkv * RATIO + offs_g)[None, :, None] * HEAD_DIM
                + offs_d[None, None, :],
                acc3.to(tl.float16), mask=offs_m[:, None, None] < Q_LEN,
            )


def paged_prefill_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_layout,
    is_causal: bool = True,
    sm_scale=None,
    block_m: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Chunked paged prefill: one launch, ``q`` is ``[num_reqs, q_len, H, D]``."""
    num_reqs, q_len, num_heads, D = q.shape
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

    grid = (triton.cdiv(q_len, block_m), num_reqs)
    _paged_prefill_kernel[grid](
        q, k_cache, v_cache, block_tables, seq_lens, o, sm_scale,
        NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, RATIO=ratio,
        HEAD_DIM=D, BLOCK_SIZE=block_size, MAX_BLOCKS=max_blocks,
        BLOCK_M=block_m, Q_LEN=q_len, IS_CAUSAL=is_causal, IS_BF16=is_bf16,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


__all__ = ["paged_prefill_forward"]
