# src/tilegen/ir/topology.py
#
# Evolvable topology & tactical IR primitives

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class GridMode(Enum):
    DATA_PARALLEL = "data_parallel"
    SPLIT_K = "split_k"
    SPLIT_KV = "split_kv"
    PERSISTENT_SM = "persistent_sm"


class RasterOrder(Enum):
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    SWIZZLED_L2 = "swizzled_l2"


class PhaseMode(Enum):
    SERIAL = "serial"
    PING_PONG = "ping_pong"


class Materialize(Enum):
    REGISTER_FUSED = "register_fused"
    SCRATCHPAD_CACHED = "scratchpad_cached"


@dataclass(frozen=True)
class GridTopology:
    """$M_0$ structural lever: how CTAs cover the output grid."""

    mode: GridMode = GridMode.DATA_PARALLEL
    raster: RasterOrder = RasterOrder.ROW_MAJOR
    split: int = 1 # SPLIT_K / SPLIT_KV reduction factor
    cta_group: Literal[1, 2] = 1 # 2 => Blackwell 2-CTA cooperative MMA

    def __post_init__(self) -> None:
        if self.split <= 0:
            raise ValueError("split must be positive")
        if self.cta_group not in (1, 2):
            raise ValueError("cta_group must be 1 or 2")
        if self.mode in (GridMode.SPLIT_K, GridMode.SPLIT_KV) and self.split < 1:
            raise ValueError("SPLIT_K/SPLIT_KV require split >= 1")

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "raster": self.raster.value,
            "split": self.split,
            "cta_group": self.cta_group,
        }


@dataclass(frozen=True)
class ExecutionRoles:
    """$M_0$ concurrency lever: producer/consumer warpgroup partition."""

    producer_warpgroups: int = 1
    consumer_warpgroups: int = 1
    phase_mode: PhaseMode = PhaseMode.SERIAL

    def __post_init__(self) -> None:
        if self.producer_warpgroups <= 0 or self.consumer_warpgroups <= 0:
            raise ValueError("warpgroup counts must be positive")

    def to_dict(self) -> dict:
        return {
            "producer_warpgroups": self.producer_warpgroups,
            "consumer_warpgroups": self.consumer_warpgroups,
            "phase_mode": self.phase_mode.value,
        }


@dataclass(frozen=True)
class RecomputeBoundary:
    """$M_0$ memory-compute trade-off: cache vs. recompute intermediates."""

    default: Materialize = Materialize.REGISTER_FUSED
    overrides: dict[str, Materialize] = field(default_factory=dict)

    def for_tensor(self, name: str) -> Materialize:
        return self.overrides.get(name, self.default)

    def to_dict(self) -> dict:
        return {
            "default": self.default.value,
            "overrides": {k: v.value for k, v in self.overrides.items()},
        }


@dataclass(frozen=True)
class TacticalKnobs:
    """$M_1 \\dots M_k$ fine-tuning levers, decoupled from $M_0$ topology."""

    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64
    pipeline_stages: int = 2
    swizzle_mode: str = "NONE" # NONE / B32 / B64 / B128
    async_wait_depth: int = 0
    max_registers_per_thread: int = 168
    exp_emulation_ratio: float = 0.0 # hybrid FMA/SFU exp2 split (FA4)
    rescale_threshold: float = 0.0 # conditional online-softmax rescaling

    def __post_init__(self) -> None:
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            raise ValueError("tile dims must be positive")
        if self.pipeline_stages <= 0:
            raise ValueError("pipeline_stages must be positive")
        if self.swizzle_mode not in ("NONE", "B32", "B64", "B128"):
            raise ValueError(f"invalid swizzle_mode {self.swizzle_mode!r}")
        if not (0.0 <= self.exp_emulation_ratio <= 1.0):
            raise ValueError("exp_emulation_ratio must be in [0, 1]")
        if self.rescale_threshold < 0.0:
            raise ValueError("rescale_threshold must be non-negative")

    def to_dict(self) -> dict:
        return {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "pipeline_stages": self.pipeline_stages,
            "swizzle_mode": self.swizzle_mode,
            "async_wait_depth": self.async_wait_depth,
            "max_registers_per_thread": self.max_registers_per_thread,
            "exp_emulation_ratio": self.exp_emulation_ratio,
            "rescale_threshold": self.rescale_threshold,
        }


@dataclass(frozen=True)
class TopologySpec:
    """Bundle of the $M_0$ topology decisions used by a VTM kernel."""

    grid: GridTopology = field(default_factory=GridTopology)
    roles: ExecutionRoles = field(default_factory=ExecutionRoles)
    recompute: RecomputeBoundary = field(default_factory=RecomputeBoundary)
    knobs: TacticalKnobs = field(default_factory=TacticalKnobs)

    def to_dict(self) -> dict:
        return {
            "grid": self.grid.to_dict(),
            "roles": self.roles.to_dict(),
            "recompute": self.recompute.to_dict(),
            "knobs": self.knobs.to_dict(),
        }


__all__ = [
    "GridMode",
    "RasterOrder",
    "PhaseMode",
    "Materialize",
    "GridTopology",
    "ExecutionRoles",
    "RecomputeBoundary",
    "TacticalKnobs",
    "TopologySpec",
]
