# src/tilegen/runtime/sdpa.py
#
# PyTorch SDPA dispatch & custom-op registration

from __future__ import annotations

import contextlib
from typing import Iterator

_REGISTERED = False
_DISPATCH_COUNT = 0


def _register_ops() -> None:
    """Register ``tilegen::sdpa`` (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    def sdpa_meta(q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False,
                  scale: float | None = None) -> Tensor:
        return torch.empty_like(q)

    def sdpa_cuda(q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False,
                  scale: float | None = None) -> Tensor:
        global _DISPATCH_COUNT
        # Route only dense, 4-D bf16 to the TileGen kernels.
        if (
            q.dtype == torch.bfloat16
            and q.is_cuda
            and q.dim() == 4
            and q.shape == k.shape
            and q.shape[-1] == v.shape[-1]
        ):
            from tilegen.attention import dense_attention

            out = dense_attention(q, k, v, is_causal=is_causal, scale=scale)
            _DISPATCH_COUNT += 1
            return out
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

    def sdpa_cpu(q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False,
                 scale: float | None = None) -> Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

    torch.library.define(
        "tilegen::sdpa",
        "(Tensor q, Tensor k, Tensor v, bool is_causal=False, float? scale=None) -> Tensor",
    )
    torch.library.impl("tilegen::sdpa", "Meta", sdpa_meta)
    torch.library.impl("tilegen::sdpa", "CUDA", sdpa_cuda)
    torch.library.impl("tilegen::sdpa", "CPU", sdpa_cpu)
    _REGISTERED = True


@contextlib.contextmanager
def sdpa_kernel(*args, **kwargs) -> Iterator[None]:
    """Route ``F.scaled_dot_product_attention`` through ``tilegen::sdpa``.

    Within this context, dense CUDA bf16 attention calls dispatch to the
    TileGen FA2 kernel; all other shapes/dtypes delegate to the native PyTorch
    SDPA backends unchanged.
    """
    import torch
    import torch.nn.functional as F

    _register_ops()
    original = F.scaled_dot_product_attention

    def routed(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        del kw
        if (
            attn_mask is None
            and dropout_p == 0.0
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and q.dim() == 4
        ):
            return torch.ops.tilegen.sdpa(q, k, v, is_causal, scale)
        return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, scale=scale)

    F.scaled_dot_product_attention = routed
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


def dispatch_count() -> int:
    """Number of CUDA dispatches into the TileGen FA2 kernel (for tests)."""
    return _DISPATCH_COUNT


__all__ = ["sdpa_kernel", "dispatch_count", "_register_ops"]
