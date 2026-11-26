# src/tilegen/ops/quant_gemm.py
#
# Quantized GEMM operators

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0 # E4M3 finite max magnitude


@triton.jit
def _fp8_gemm_kernel(
    A, B, A_scale, B_scale, C,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_scale = tl.load(A_scale)
    b_scale = tl.load(B_scale)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs) # fp8e4nv
        b = tl.load(b_ptrs) # fp8e4nv
        acc = tl.dot(a, b, acc) # native FP8 MMA -> fp32 accumulator
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    acc = acc * (a_scale * b_scale)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16))


def quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to FP8 E4M3 with a per-tensor scale (returns ``(x_q, scale)``)."""
    scale = (x.abs().max() / FP8_MAX).clamp(min=1e-12)
    x_q = (x / scale).to(torch.float8_e4m3fn)
    return x_q, scale


def quant_gemm_forward(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP8 GEMM: ``C = (A * a_scale) @ (B * b_scale)``.

    ``a``/``b`` are ``[M, K]``/``[K, N]`` FP8 tensors with per-tensor scales.
    Returns ``[M, N]`` in ``out_dtype``.
    """
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fp8_gemm_kernel[grid](
        a, b, a_scale, b_scale, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return c


@triton.jit
def _w4a16_kernel(
    A, W, W_scale, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn, # W packed [K, N//2]
    stride_sg, stride_sn, # scale [K//group, N]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_nh = tl.arange(0, BLOCK_N // 2) # packed half-width

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptrs = A + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        a = tl.load(a_ptrs) # [BLOCK_M, BLOCK_K] fp16

        w_ptrs = W + (k + offs_k)[:, None] * stride_wk + \
            (pid_n * (BLOCK_N // 2) + offs_nh)[None, :] * stride_wn
        w_packed = tl.load(w_ptrs) # uint8 [BLOCK_K, BLOCK_N//2]
        w_lo = (w_packed & 0x0F).to(tl.float16)
        w_hi = ((w_packed >> 4) & 0x0F).to(tl.float16)
        w_int = tl.interleave(w_lo, w_hi) - 8.0 # signed int4 [BLOCK_K, BLOCK_N]

        group = k // BLOCK_K # one scale group per K-tile (group_size == BLOCK_K)
        s_ptrs = W_scale + group * stride_sg + offs_n * stride_sn
        s = tl.load(s_ptrs) # [BLOCK_N] fp16
        w = w_int * s[None, :] # fused dequant

        acc = tl.dot(a, w, acc)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


def quantize_w4(w: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[K, N]`` to symmetric signed INT4 with per-group scales.

    Returns ``(packed_uint8, scale)`` where ``packed_uint8`` is ``[K, N//2]``
    (two nibbles per byte, low nibble = even column) and ``scale`` is
    ``[K // group_size, N]``.
    """
    K, N = w.shape
    assert K % group_size == 0 and N % 2 == 0
    wf = w.float().reshape(K // group_size, group_size, N)
    w_max = wf.abs().amax(dim=1, keepdim=True) # [G, 1, N]
    scale = (w_max / 8.0).clamp(min=1e-6) # [G, 1, N]
    w_q = (wf / scale).round().clamp(-8, 7).to(torch.int8) # [G, group_size, N]
    nibble = (w_q + 8).to(torch.uint8) # 0..15
    even = nibble[..., 0::2]
    odd = nibble[..., 1::2]
    packed = (even | (odd << 4)).reshape(K, N // 2).contiguous()
    scale = scale.squeeze(1).to(w.dtype) # [G, N]
    return packed, scale


def dequant_w4(packed: torch.Tensor, scale: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Reference dequantization of packed INT4 weights to ``[K, N]``."""
    K, Nh = packed.shape
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    w_int = torch.stack([lo, hi], dim=-1).reshape(K, Nh * 2) - 8
    scale_exp = scale.repeat_interleave(group_size, dim=0)
    return w_int.float() * scale_exp


def w4a16_gemm_forward(
    a: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """W4A16 GEMM: ``C = A @ dequant(w_packed)`` (A fp16, W INT4)."""
    M, K = a.shape
    K2, Nh = w_packed.shape
    assert K == K2 and K % group_size == 0
    N = Nh * 2
    a = a.contiguous()
    w_packed = w_packed.contiguous()
    w_scale = w_scale.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _w4a16_kernel[grid](
        a, w_packed, w_scale, c, M, N, K,
        a.stride(0), a.stride(1), w_packed.stride(0), w_packed.stride(1),
        w_scale.stride(0), w_scale.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return c


@triton.jit
def _w8a8_kernel(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs) # int8
        b = tl.load(b_ptrs) # int8
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def w8a8_gemm_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """INT8 GEMM: ``C = A @ B`` with int32 accumulation (native int8 MMA)."""
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _w8a8_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return c


# OCP MXFP4 / NVIDIA NVFP4 E2M1 magnitudes (1 sign, 2 exp, 1 mantissa).
_FP4_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_E2M1_VALUES = _FP4_E2M1_MAG + [-v for v in _FP4_E2M1_MAG]


def _fp4_lut(dtype: torch.dtype, device) -> torch.Tensor:
    return torch.tensor(FP4_E2M1_VALUES, dtype=dtype, device=device)


def quantize_fp4(x: torch.Tensor, block_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[K, N]`` to FP4 E2M1 with per-block scales.

    Returns ``(packed_uint8 [K, N//2], scale [K//block_size, N])`` (low nibble =
    even column), matching ``QuantSpec.nvfp4()`` (block=16, pack=2).
    """
    K, N = x.shape
    assert K % block_size == 0 and N % 2 == 0
    xf = x.float().reshape(K // block_size, block_size, N)
    scale = (xf.abs().amax(dim=1, keepdim=True) / 6.0).clamp(min=1e-6) # [G, 1, N]
    q = xf / scale
    sign = (q < 0).to(torch.uint8) << 3
    mags = torch.tensor(_FP4_E2M1_MAG, device=x.device)
    idx = (q.abs().unsqueeze(-1) - mags).abs().argmin(dim=-1).to(torch.uint8) # 0..7
    nibble = idx | sign # 0..15
    even = nibble[..., 0::2]
    odd = nibble[..., 1::2]
    packed = (even | (odd << 4)).reshape(K, N // 2).contiguous()
    return packed, scale.squeeze(1).to(x.dtype)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Reference dequantization of packed FP4 E2M1 weights to ``[K, N]``."""
    K, Nh = packed.shape
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    idx = torch.stack([lo, hi], dim=-1).reshape(K, Nh * 2) # [K, N] 0..15
    lut = _fp4_lut(torch.float32, packed.device)
    w = lut[idx.long()] # [K, N] fp32
    scale_exp = scale.repeat_interleave(block_size, dim=0)
    return (w * scale_exp.float()).to(scale.dtype)


def fp4_gemm_forward(
    a: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Emulated FP4 GEMM: dequantize weights to FP16/BF16, then ``A @ W``."""
    w = dequant_fp4(w_packed, w_scale, block_size)
    return (a.float() @ w.float()).to(a.dtype)


__all__ = [
    "quant_gemm_forward",
    "quantize_fp8",
    "quantize_w4",
    "dequant_w4",
    "w4a16_gemm_forward",
    "w8a8_gemm_forward",
    "quantize_fp4",
    "dequant_fp4",
    "fp4_gemm_forward",
    "FP8_MAX",
    "FP4_E2M1_VALUES",
]
