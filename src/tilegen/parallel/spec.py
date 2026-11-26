# src/tilegen/parallel/spec.py
#
# Parallelism spec

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CommPattern(Enum):
    ALL_GATHER = "all_gather" # gather K/V -> local attention -> (optional) reduce-scatter
    RING = "ring" # in-kernel P2P ring (RingAttention; not yet lowered)
    TREE = "tree" # tree reduce for softmax statistics


@dataclass(frozen=True)
class ParallelismSpec:
    """Parallel layout for a VTM attention kernel.

    ``cp_degree`` shards the sequence for *attention*; ``sp_degree`` shards the
    sequence for the *whole block* (attention is the CP part, the rest is
    embarrassingly parallel); ``tp_degree`` shards heads/weights.  ``devices``
    optionally names the per-rank device backend for heterogeneous multi-DSA
    (e.g. ``("cuda", "cuda", "rocm")``); empty means homogeneous.
    """

    cp_degree: int = 1
    sp_degree: int = 1
    tp_degree: int = 1
    comm_pattern: CommPattern = CommPattern.ALL_GATHER
    shard_seq: bool = True
    devices: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name, val in (("cp_degree", self.cp_degree), ("sp_degree", self.sp_degree), ("tp_degree", self.tp_degree)):
            if val <= 0:
                raise ValueError(f"{name} must be positive")
        if self.comm_pattern == CommPattern.RING and self.cp_degree <= 1:
            raise ValueError("RING comm pattern requires cp_degree >= 2")
        if self.devices and len(self.devices) != self.world_size:
            raise ValueError(f"devices length {len(self.devices)} != world_size {self.world_size}")

    @property
    def world_size(self) -> int:
        return self.cp_degree * self.sp_degree * self.tp_degree

    @property
    def is_sharded(self) -> bool:
        return self.world_size > 1

    def to_dict(self) -> dict:
        return {
            "cp_degree": self.cp_degree,
            "sp_degree": self.sp_degree,
            "tp_degree": self.tp_degree,
            "comm_pattern": self.comm_pattern.value,
            "shard_seq": self.shard_seq,
            "devices": list(self.devices),
        }


def parallelism_from_cp(
    cp_size: int,
    tp_size: int = 1,
    comm_pattern: CommPattern = CommPattern.ALL_GATHER,
) -> ParallelismSpec:
    """Build a :class:`ParallelismSpec` from serving-framework CP metadata.

    vLLM exposes ``parallel_config.context_parallel_size``; SGLang exposes its
    attention DP/CP size.  This maps that single value onto TileGen's spec.
    """
    return ParallelismSpec(cp_degree=cp_size, tp_degree=tp_size, comm_pattern=comm_pattern)


__all__ = ["CommPattern", "ParallelismSpec", "parallelism_from_cp"]
