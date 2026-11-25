# src/tilegen/ops/gdn.py
#
# Gated Delta Network (GDN) linear attention

from __future__ import annotations

import torch


def gdn_forward(
    x: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wq: torch.Tensor,
    wb: torch.Tensor,
    wo: torch.Tensor,
) -> torch.Tensor:
    """Gated delta-rule linear attention.

    Args:
        x: ``[T, D]`` input sequence.
        wk/wv/wq: ``[D, d]`` key / value / query projections.
        wb: ``[D, 1]`` gate projection (sigmoid -> ``beta``).
        wo: ``[d, D]`` output projection.

    Returns:
        ``[T, D]`` the gated-memory readout.
    """
    T, D = x.shape
    d = wk.shape[1]

    k = (x @ wk).float()        # [T, d]
    v = (x @ wv).float()        # [T, d]
    q = (x @ wq).float()        # [T, d]
    beta = torch.sigmoid(x @ wb).float()  # [T, 1]

    S = torch.zeros(d, d, device=x.device, dtype=torch.float32)
    o = torch.zeros(T, d, device=x.device, dtype=torch.float32)
    for t in range(T):
        # Delta rule: gated fast-weight update.
        S = (1.0 - beta[t]) * S + beta[t] * torch.outer(k[t], v[t])
        o[t] = q[t] @ S
    return (o @ wo.float()).to(x.dtype)


__all__ = ["gdn_forward"]
