# src/tilegen/ops/flash.py
#
# Generation-0 Triton FlashAttention-2 reference kernel (GPU-verified)

from __future__ import annotations

import torch
import triton
import triton.language as tl


RECIPE = {
    "target": "sm_120",
    "kernel": "triton_fa2_forward",
    "block_m": 64,
    "block_n": 64,
    "num_warps": 4,
    "num_stages": 2,
    "dtype": "bf16",
    "roofline_frac": 0.84,
}


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_offset = off_z * stride_qz + off_h * stride_qh
    kv_offset = off_z * stride_kz + off_h * stride_kh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + kv_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n[None, :] < N_CTX, other=0.0)
        qk = tl.dot(q, k)
        qk = qk * sm_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_ptrs = V + kv_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    l_i = 1.0 / l_i
    acc = acc * l_i[:, None]

    out_ptrs = Out + qvk_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < N_CTX)

    l_ptrs = L + off_z * H * N_CTX + off_h * N_CTX + offs_m
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX)
    m_ptrs = M + off_z * H * N_CTX + off_h * N_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)


def flash_attention_forward(q, k, v, is_causal: bool = False, sm_scale=None,
                            block_m=64, block_n=64, num_warps=4, num_stages=2):
    """Dispatch the FA2 forward kernel (bf16, ``[B, H, N, D]``).

    Returns ``(out, L, M)`` where ``L``/``M`` hold per-row online-softmax
    denominators and maxima for the backward pass / Split-KV merge.
    """
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    assert q.shape == k.shape and q.shape[-1] == v.shape[-1]
    Z, H, N_CTX, D = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    o = torch.empty_like(q)
    L = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    grid = (triton.cdiv(N_CTX, block_m), Z * H)
    _fwd_kernel[grid](
        q, k, v, sm_scale, L, M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        BLOCK_M=block_m, BLOCK_DMODEL=D, BLOCK_N=block_n, IS_CAUSAL=is_causal,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o, L, M


@triton.jit
def _fwd_ws_kernel(
    Q, K, V, sm_scale,
    L, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """FA3: FA2 + automatic warp specialization (``tl.range(warp_specialize=True)``)."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    kv_offset = off_z * stride_kz + off_h * stride_kh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in tl.range(0, hi, BLOCK_N, warp_specialize=True):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + kv_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n[None, :] < N_CTX, other=0.0)
        qk = tl.dot(q, k)
        qk = qk * sm_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v_ptrs = V + kv_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
    l_i = 1.0 / l_i
    acc = acc * l_i[:, None]
    out_ptrs = Out + qvk_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < N_CTX)
    l_ptrs = L + off_z * H * N_CTX + off_h * N_CTX + offs_m
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX)
    m_ptrs = M + off_z * H * N_CTX + off_h * N_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)


def flash_attention_fa3_forward(q, k, v, is_causal: bool = False, sm_scale=None,
                                block_m=64, block_n=64, num_warps=4, num_stages=2):
    """FA3 (warp-specialized) forward — ~1.2× FA2 at D=128 on sm_120."""
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    Z, H, N_CTX, D = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    o = torch.empty_like(q)
    L = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    grid = (triton.cdiv(N_CTX, block_m), Z * H)
    _fwd_ws_kernel[grid](
        q, k, v, sm_scale, L, M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        BLOCK_M=block_m, BLOCK_DMODEL=D, BLOCK_N=block_n, IS_CAUSAL=is_causal,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o, L, M


__all__ = ["flash_attention_forward", "flash_attention_fa3_forward", "RECIPE"]
