# src/tilegen/ops/quant_int4.py
#
# Symmetric group-quantized INT4 KV

from __future__ import annotations

import torch

__all__ = ["quantize_int4_kv", "dequant_int4_kv", "PACKED_BYTES"]

PACKED_BYTES = 0.5 # packed bytes per element (2 int4 per byte)


def quantize_int4_kv(
    x: torch.Tensor,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., D]`` to symmetric group INT4.

    Returns ``(packed [..., D//2] uint8, scale [..., D//group_size] fp16)``.
    """
    assert x.shape[-1] % group_size == 0, "head_dim must divide group_size"
    xf = x.float()
    groups = x.shape[-1] // group_size
    xg = xf.reshape(*x.shape[:-1], groups, group_size)
    scale = (xg.abs().max(dim=-1, keepdim=True).values / 8.0).clamp(min=1e-6)
    q = torch.clamp(torch.round(xg / scale) + 8.0, 0.0, 15.0) # [.., groups, gs]
    q = q.reshape(*x.shape[:-1], x.shape[-1]) # [.., D]
    lo = q[..., 0::2].to(torch.uint8)
    hi = q[..., 1::2].to(torch.uint8)
    packed = lo | (hi << 4)
    return packed, scale.squeeze(-1).to(torch.float16)


def dequant_int4_kv(
    packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """Dequant ``[..., D//2] uint8`` + ``[..., D//group_size]`` scale → fp16."""
    lo = (packed & 0x0F).to(torch.float32)
    hi = ((packed >> 4) & 0x0F).to(torch.float32)
    w = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    scale = scale.float().repeat_interleave(group_size, dim=-1) # [.., groups] -> [.., D]
    return ((w - 8.0) * scale).to(torch.float16)
