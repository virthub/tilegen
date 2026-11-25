# src/tilegen/ops/fused.py
#
# Fused elementwise operators

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(X, W, Y, eps, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x_ptrs = X + row * H + offs
    x = tl.load(x_ptrs, mask=offs < H, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    y = x * rstd
    w = tl.load(W + offs, mask=offs < H, other=0.0).to(tl.float32)
    y = (y * w).to(tl.float16)
    tl.store(Y + row * H + offs, y, mask=offs < H)


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise RMSNorm: ``y = x / sqrt(mean(x^2) + eps) * weight``."""
    assert x.dim() == 2 and x.shape[-1] == weight.shape[-1]
    T, H = x.shape
    x = x.contiguous()
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(H)
    _rms_norm_kernel[(T,)](x, weight, y, eps, H, BLOCK=BLOCK, num_warps=4)
    return y


@triton.jit
def _rope_kernel(X, COS, SIN, Y, H, half, stride_xt, stride_xh, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H
    offs = tl.arange(0, BLOCK)
    base = t * stride_xt + h * stride_xh
    x1 = tl.load(X + base + offs).to(tl.float32)
    x2 = tl.load(X + base + (offs + BLOCK)).to(tl.float32)
    c = tl.load(COS + t * BLOCK + offs).to(tl.float32)
    s = tl.load(SIN + t * BLOCK + offs).to(tl.float32)
    y1 = (x1 * c - x2 * s).to(tl.float16)
    y2 = (x2 * c + x1 * s).to(tl.float16)
    tl.store(Y + base + offs, y1)
    tl.store(Y + base + (offs + BLOCK), y2)


def rope_forward(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """GPT-NeoX RoPE for ``x`` of shape ``[T, H, D]`` with ``cos``/``sin`` ``[T, D//2]``.

    ``cos``/``sin`` are broadcast across heads.
    """
    assert x.dim() == 3 and x.shape[-1] % 2 == 0
    T, H, D = x.shape
    half = D // 2
    assert cos.shape == (T, half) and sin.shape == (T, half)
    x = x.contiguous()
    y = torch.empty_like(x)
    _rope_kernel[(T * H,)](x, cos, sin, y, H, half,
                           x.stride(0), x.stride(1), BLOCK=half, num_warps=4)
    return y


@triton.jit
def _swiglu_kernel(G, U, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    # silu(g) = g * sigmoid(g)
    silu = g / (1.0 + tl.exp(-g))
    y = (silu * u).to(tl.float16)
    tl.store(Y + offs, y, mask=mask)


def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU: ``y = silu(gate) * up`` (elementwise)."""
    assert gate.shape == up.shape
    orig_shape = gate.shape
    N = gate.numel()
    gate = gate.contiguous().flatten()
    up = up.contiguous().flatten()
    y = torch.empty_like(gate)
    BLOCK = 1024
    _swiglu_kernel[(triton.cdiv(N, BLOCK),)](gate, up, y, N, BLOCK=BLOCK)
    return y.reshape(orig_shape)


@triton.jit
def _layer_norm_kernel(X, W, B, Y, eps, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + offs, mask=offs < H, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / H
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xc * rstd
    w = tl.load(W + offs, mask=offs < H, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=offs < H, other=0.0).to(tl.float32)
    y = (y * w + b).to(tl.float16)
    tl.store(Y + row * H + offs, y, mask=offs < H)


def layer_norm_forward(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Row-wise LayerNorm: ``y = (x - mean) / sqrt(var + eps) * w + b``."""
    assert x.dim() == 2 and x.shape[-1] == weight.shape[-1] == bias.shape[-1]
    T, H = x.shape
    x = x.contiguous()
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(H)
    _layer_norm_kernel[(T,)](x, weight, bias, y, eps, H, BLOCK=BLOCK, num_warps=4)
    return y


@triton.jit
def _gelu_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # tanh approximation: 0.5*x*(1 + tanh(0.7978845608 * (x + 0.044715 x^3)))
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    e2 = tl.exp(2.0 * inner)
    tanh_inner = (e2 - 1.0) / (e2 + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner)
    tl.store(Y + offs, y.to(tl.float16), mask=mask)


@triton.jit
def _geglu_kernel(G, U, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608 * (g + 0.044715 * g * g * g)
    e2 = tl.exp(2.0 * inner)
    gelu = 0.5 * g * (1.0 + (e2 - 1.0) / (e2 + 1.0))
    y = (gelu * u).to(tl.float16)
    tl.store(Y + offs, y, mask=mask)


def gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """GELU activation (tanh approximation), elementwise."""
    orig_shape = x.shape
    N = x.numel()
    xf = x.contiguous().flatten()
    y = torch.empty_like(xf)
    BLOCK = 1024
    _gelu_kernel[(triton.cdiv(N, BLOCK),)](xf, y, N, BLOCK=BLOCK)
    return y.reshape(orig_shape)


def geglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """GeGLU: ``y = gelu(gate) * up`` (elementwise)."""
    assert gate.shape == up.shape
    orig_shape = gate.shape
    N = gate.numel()
    g = gate.contiguous().flatten()
    u = up.contiguous().flatten()
    y = torch.empty_like(g)
    BLOCK = 1024
    _geglu_kernel[(triton.cdiv(N, BLOCK),)](g, u, y, N, BLOCK=BLOCK)
    return y.reshape(orig_shape)


def moe_topk_gate(
    logits: torch.Tensor, k: int = 2, normalize: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k expert routing: returns ``(weights, expert_indices)``.

    ``logits`` is ``[num_tokens, num_experts]``; ``weights`` are the (optional
    softmax-normalized) top-k logits and ``expert_indices`` the ``[num_tokens, k]``
    selected experts.
    """
    topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
    weights = torch.softmax(topk_logits, dim=-1) if normalize else topk_logits
    return weights, topk_idx


__all__ = [
    "rms_norm_forward",
    "layer_norm_forward",
    "rope_forward",
    "swiglu_forward",
    "gelu_forward",
    "geglu_forward",
    "moe_topk_gate",
]
