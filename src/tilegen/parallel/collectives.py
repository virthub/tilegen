# src/tilegen/parallel/collectives.py
#
# Collective primitives

from __future__ import annotations

import torch
import torch.distributed as dist
from typing import Optional


def _group(group):
    if group is not None:
        return group
    return dist.group.WORLD if dist.is_available() and dist.is_initialized() else None


def is_initialized(group=None) -> bool:
    return _group(group) is not None


def all_gather(t: torch.Tensor, dim: int = 0, group=None) -> torch.Tensor:
    """All-gather ``t`` along ``dim`` (the Approach-A CP K/V visibility step)."""
    g = _group(group)
    if g is None:
        return t
    world = dist.get_world_size(g)
    chunks = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(chunks, t, group=g)
    return torch.cat(chunks, dim=dim)


def reduce_scatter(t: torch.Tensor, dim: int = 0, group=None, op: str = "sum") -> torch.Tensor:
    """Reduce-scatter ``t`` along ``dim`` back to the owning rank."""
    g = _group(group)
    if g is None:
        return t
    reduce_op = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}[op]
    out = torch.empty_like(t).narrow(dim, 0, t.shape[dim] // dist.get_world_size(g))
    dist.reduce_scatter_tensor(out, t, op=reduce_op, group=g)
    return out


def all_reduce(t: torch.Tensor, op: str = "max", group=None) -> torch.Tensor:
    """All-reduce ``t`` (softmax statistics) across ranks."""
    g = _group(group)
    if g is None:
        return t
    reduce_op = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}[op]
    out = t.clone()
    dist.all_reduce(out, op=reduce_op, group=g)
    return out


def ring_step(send: torch.Tensor, recv: torch.Tensor, send_rank: int, recv_rank: int, group=None) -> None:
    """One P2P ring step (Approach B): send ``send`` to ``send_rank``, receive into ``recv``.

    Not used by the default all-gather executor; provided for the ring pattern.
    """
    g = _group(group)
    if g is None:
        return
    # send/recv pair (use a combined op to avoid deadlock on the ring).
    ops = [dist.P2POp(dist.isend, send, send_rank, group=g),
           dist.P2POp(dist.irecv, recv, recv_rank, group=g)]
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()


__all__ = ["is_initialized", "all_gather", "reduce_scatter", "all_reduce", "ring_step"]
