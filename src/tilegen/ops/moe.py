# src/tilegen/ops/moe.py
#
# Mixture-of-Experts forward

from __future__ import annotations

import torch
import torch.nn.functional as F
from tilegen.ops.fused import moe_topk_gate


def moe_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gate_logits: torch.Tensor,
    top_k: int = 2,
) -> torch.Tensor:
    """Mixture-of-Experts forward.

    Args:
        x: ``[T, H]`` input tokens.
        w1: ``[E, 2*I, H]`` concatenated gate/up expert weights.
        w2: ``[E, H, I]`` down-projection expert weights.
        gate_logits: ``[T, E]`` router logits.
        top_k: number of experts per token.

    Returns:
        ``[T, H]`` the gating-weighted sum of expert outputs.
    """
    T, H = x.shape
    E, inter2, _ = w1.shape
    I = inter2 // 2

    weights, indices = moe_topk_gate(gate_logits, top_k) # [T, k], [T, k]
    out = torch.zeros(T, H, device=x.device, dtype=x.dtype)

    for e in range(E):
        token_mask = indices == e # [T, k]
        tw = (weights * token_mask).sum(dim=1) # [T] weight for expert e
        sel = tw > 0
        if not sel.any():
            continue
        xe = x[sel] # [n, H]
        gate_up = xe @ w1[e].t() # [n, 2I]
        h = F.silu(gate_up[:, :I]) * gate_up[:, I:] # SwiGLU [n, I]
        y = h @ w2[e].t() # [n, H]
        out[sel] += y * tw[sel, None]
    return out


__all__ = ["moe_forward"]
