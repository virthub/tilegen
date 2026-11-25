# src/tilegen/attention.py
#
# Canonical attention kernel dispatch

from __future__ import annotations

import torch

__all__ = [
    "dense_attention",
    "paged_decode",
    "paged_prefill",
    "paged_decode_int4",
    "tilelang_available",
]


def tilelang_available() -> bool:
    """True if the TileLang FA4 backend is importable."""
    try:
        import tilelang  # noqa: F401

        return True
    except Exception:
        return False


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    is_causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """Best dense attention forward.

    Cascade: FA4 (TileLang tcgen05, ~1.02-1.05x native) → FA3/FA2 (Triton) →
    native ``F.scaled_dot_product_attention`` (fp16 fallback).  Returns the
    output tensor ``[B, H, N, D]``.

    Prefix-cached / cross attention (``q.shape[2] != k.shape[2]``) uses native
    SDPA — the Triton FA2/FA3 kernels assume ``q_len == kv_len``, and the
    TileLang FA4 ``past_len`` path is not yet validated for this case.
    """
    prefix_case = q.dim() == 4 and q.shape[2] != k.shape[2]

    if prefix_case:
        import torch.nn.functional as F

        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

    try:
        from tilegen.ops.flash_tilelang import flash_attention_tilelang_forward

        return flash_attention_tilelang_forward(q, k, v, is_causal=is_causal)[0]
    except Exception:
        pass

    if q.dtype == torch.bfloat16:
        from tilegen.ops.flash import flash_attention_fa3_forward, flash_attention_forward

        fn = flash_attention_fa3_forward if q.shape[-1] >= 128 else flash_attention_forward
        out, _, _ = fn(q, k, v, is_causal=is_causal, sm_scale=scale)
        return out

    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)


def paged_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                 block_table: torch.Tensor, seq_lens: torch.Tensor, kv_layout,
                 sm_scale: float | None = None) -> torch.Tensor:
    """Batched paged decode (single launch, ~25x the per-request loop)."""
    from tilegen.ops.paged_decode import paged_decode_forward

    return paged_decode_forward(
        q, k_cache, v_cache, block_table, seq_lens, kv_layout, sm_scale=sm_scale
    )


def paged_prefill(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                  block_table: torch.Tensor, seq_lens: torch.Tensor, kv_layout,
                  is_causal: bool = True, sm_scale: float | None = None) -> torch.Tensor:
    """Chunked paged prefill (flash-attention over paged K/V)."""
    from tilegen.ops.paged_prefill import paged_prefill_forward

    return paged_prefill_forward(
        q, k_cache, v_cache, block_table, seq_lens, kv_layout,
        is_causal=is_causal, sm_scale=sm_scale,
    )


def paged_decode_int4(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                      k_scale: torch.Tensor, v_scale: torch.Tensor,
                      block_table: torch.Tensor, seq_lens: torch.Tensor, kv_layout,
                      group_size: int = 64, sm_scale: float | None = None) -> torch.Tensor:
    """Batched paged decode with in-kernel INT4 dequant (split-dot)."""
    from tilegen.ops.paged_decode import paged_decode_int4_forward

    return paged_decode_int4_forward(
        q, k_cache, v_cache, k_scale, v_scale, block_table, seq_lens, kv_layout,
        group_size=group_size, sm_scale=sm_scale,
    )
