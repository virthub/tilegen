# src/tilegen/parallel/__init__.py
#
# Parallelism layer

"""The generic CP/SP (context/sequence parallelism) flow for inference:

* :class:`ParallelismSpec` — device-agnostic cp/sp/tp layout.
* :mod:`collectives` — ``torch.distributed`` (NCCL/RCCL/HCCL) wrappers.
* :class:`ParallelExecutor` — wraps an *unchanged* FA2/FA3/FA4 kernel with
  AllGather(K/V) → local attention → ReduceScatter(O).

The IR also gains collective ops (``AllGatherOp``/``ReduceScatterOp``/
``AllReduceOp``/``RingStepOp``) and an ``OnlineSoftmaxOp.global_reduce`` flag so
the synthesis engine can reason about the cross-rank reduction — even though
execution is host-side orchestration in the default all-gather pattern.
"""

from tilegen.parallel.spec import CommPattern, ParallelismSpec, parallelism_from_cp
from tilegen.parallel.collectives import all_gather, all_reduce, reduce_scatter, ring_step
from tilegen.parallel.executor import ParallelExecutor

__all__ = [
    "CommPattern",
    "ParallelismSpec",
    "parallelism_from_cp",
    "all_gather",
    "all_reduce",
    "reduce_scatter",
    "ring_step",
    "ParallelExecutor",
]
