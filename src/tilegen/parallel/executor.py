# src/tilegen/parallel/executor.py
#
# Parallel executor

from __future__ import annotations

import torch
from typing import Callable, Optional
from tilegen.parallel import collectives
from tilegen.parallel.spec import CommPattern, ParallelismSpec

__all__ = ["ParallelExecutor"]


class ParallelExecutor:
    """Runs a local attention kernel under a :class:`ParallelismSpec`."""

    def __init__(self, spec: ParallelismSpec, group=None, local_forward: Optional[Callable] = None):
        self.spec = spec
        self.group = group
        self._local = local_forward

    @property
    def _forward(self):
        if self._local is not None:
            return self._local
        from tilegen.attention import dense_attention

        return dense_attention

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  is_causal: bool = False, scale: Optional[float] = None) -> torch.Tensor:
        """Dense attention with context parallelism (Approach A).

        Each rank owns a sequence shard of Q (and its K/V); the executor gathers
        K/V, runs the local kernel, and returns the rank's Q-shard output.
        """
        if not self.spec.is_sharded or self.spec.comm_pattern != CommPattern.ALL_GATHER:
            return self._forward(q, k, v, is_causal=is_causal, scale=scale)

        k_full = collectives.all_gather(k, dim=2, group=self.group)
        v_full = collectives.all_gather(v, dim=2, group=self.group)
        return self._forward(q, k_full, v_full, is_causal=is_causal, scale=scale)

    def paged_decode(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     block_table: torch.Tensor, seq_lens: torch.Tensor, kv_layout,
                     sm_scale: Optional[float] = None) -> torch.Tensor:
        """Paged decode under context parallelism.

        The KV cache is sharded across ranks by *block range*; each rank gathers
        the other ranks' cache blocks before running the local batched decode.
        """
        from tilegen.attention import paged_decode

        if not self.spec.is_sharded or self.spec.comm_pattern != CommPattern.ALL_GATHER:
            return paged_decode(q, k_cache, v_cache, block_table, seq_lens, kv_layout, sm_scale=sm_scale)

        k_full = collectives.all_gather(k_cache, dim=0, group=self.group)
        v_full = collectives.all_gather(v_cache, dim=0, group=self.group)
        return paged_decode(q, k_full, v_full, block_table, seq_lens, kv_layout, sm_scale=sm_scale)
