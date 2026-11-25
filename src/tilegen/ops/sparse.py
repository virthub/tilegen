# src/tilegen/ops/sparse.py
#
# Sparse MLA — top-k sparse attention

from __future__ import annotations

import torch


def sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Top-k sparse attention.

    Args:
        q: ``[B, H, N, D]`` queries.
        k: ``[B, H, M, D]`` keys.
        v: ``[B, H, M, D]`` values.
        top_k: number of keys to retain per query.
        scale: softmax scale (defaults to ``1/sqrt(D)``).

    Returns:
        ``[B, H, N, D]`` the sparse attention output.
    """
    B, H, N, D = q.shape
    M = k.shape[2]
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    assert top_k <= M

    scores = torch.einsum("bhnd,bhmd->bhnm", q, k) * scale      # [B, H, N, M]
    topk_scores, topk_idx = torch.topk(scores, top_k, dim=-1)   # [B, H, N, k]
    topk_weights = torch.softmax(topk_scores, dim=-1)           # [B, H, N, k]

    # Gather the selected values and weight-sum.
    v_exp = v.unsqueeze(2).expand(B, H, N, M, D)                # [B, H, N, M, D]
    idx_exp = topk_idx.unsqueeze(-1).expand(B, H, N, top_k, D)  # [B, H, N, k, D]
    topk_v = torch.gather(v_exp, dim=3, index=idx_exp)          # [B, H, N, k, D]
    return torch.einsum("bhnk,bhnkd->bhnd", topk_weights, topk_v)


__all__ = ["sparse_attention_forward"]
