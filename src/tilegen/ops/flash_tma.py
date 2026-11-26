# src/tilegen/ops/flash_tma.py
#
# TMA FlashAttention forward

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _fa2_tma_fwd(
    Q, K, V, sm_scale, L, M, Out,
    stride_om, stride_on,
    Z, H, N_CTX,
    q_desc, k_desc, v_desc,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # TMA load of the Q tile [BLOCK_M, D].
    q = q_desc.load([off_hz * N_CTX + start_m * BLOCK_M, 0])

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = k_desc.load([off_hz * N_CTX + start_n, 0]) # [BLOCK_N, D]
        qk = tl.dot(q, tl.trans(k)) # [BLOCK_M, BLOCK_N]
        qk = qk * sm_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = v_desc.load([off_hz * N_CTX + start_n, 0]) # [BLOCK_N, D]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    l_i = 1.0 / l_i
    acc = acc * l_i[:, None]

    out_ptrs = Out + off_hz * N_CTX * BLOCK_DMODEL + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < N_CTX)

    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX)
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)


def flash_attention_tma_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale=None,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
):
    """TMA flash-attention forward (bf16).  Returns ``(out, L, M)``."""
    assert q.dtype == torch.bfloat16
    Z, H, N_CTX, D = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()

    o = torch.empty_like(q)
    L = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    # 2-D descriptors over flattened [B*H*N, D] views.
    q_flat = q.reshape(-1, D)
    k_flat = k.reshape(-1, D)
    v_flat = v.reshape(-1, D)
    q_desc = TensorDescriptor.from_tensor(q_flat, [block_m, D])
    k_desc = TensorDescriptor.from_tensor(k_flat, [block_n, D])
    v_desc = TensorDescriptor.from_tensor(v_flat, [block_n, D])

    grid = (triton.cdiv(N_CTX, block_m), Z * H)
    _fa2_tma_fwd[grid](
        q_flat, k_flat, v_flat, sm_scale, L, M, o,
        o.stride(2), o.stride(3), Z, H, N_CTX,
        q_desc, k_desc, v_desc,
        BLOCK_M=block_m, BLOCK_DMODEL=D, BLOCK_N=block_n, IS_CAUSAL=is_causal,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o, L, M


__all__ = ["flash_attention_tma_forward"]
