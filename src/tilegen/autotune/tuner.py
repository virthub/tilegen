# src/tilegen/autotune/tuner.py
#
# Constraint-pruned offline autotuner

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence
from tilegen.layout.quant import QuantSpec
from tilegen.ops.guards import check_quant_alignment
from tilegen.target.spec import TargetSpec


@dataclass(frozen=True)
class TuneConfig:
    """One point in the autotuning search space."""

    tile_m: int
    tile_n: int
    tile_k: int
    pipeline_stages: int = 2
    swizzle_mode: str = "NONE"
    num_warps: int = 4

    def to_dict(self) -> dict:
        return {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "pipeline_stages": self.pipeline_stages,
            "swizzle_mode": self.swizzle_mode,
            "num_warps": self.num_warps,
        }


@dataclass
class TuneResult:
    config: TuneConfig
    device_time_us: float
    correct: bool = True
    max_rel_error: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "device_time_us": self.device_time_us,
            "correct": self.correct,
            "max_rel_error": self.max_rel_error,
            "notes": self.notes,
        }


@dataclass
class SearchSpace:
    tile_m: Sequence[int] = (64, 128, 256)
    tile_n: Sequence[int] = (64, 128, 256)
    tile_k: Sequence[int] = (32, 64, 128)
    pipeline_stages: Sequence[int] = (1, 2, 3, 4)
    swizzle_mode: Sequence[str] = ("NONE", "B32", "B64", "B128")
    num_warps: Sequence[int] = (4, 8)

    @classmethod
    def from_dict(cls, data: dict) -> "SearchSpace":
        known = {f: getattr(cls, f) for f in cls.__dataclass_fields__}
        merged = {k: data.get(k, v) for k, v in known.items()}
        return cls(**merged)

    def expand(self) -> list[TuneConfig]:
        return [
            TuneConfig(m, n, k, s, w, warps)
            for m, n, k, s, w, warps in product(
                self.tile_m, self.tile_n, self.tile_k,
                self.pipeline_stages, self.swizzle_mode, self.num_warps,
            )
        ]


def estimate_smem_bytes(config: TuneConfig, head_dim: int, dtype_bytes: int = 2) -> int:
    """Approximate shared-memory footprint of a FA2 config.

    Two circular buffers (K and V) of ``stages * tile_n * head_dim`` elements
    each, plus a Q tile of ``tile_m * head_dim``.
    """
    kv_buffers = 2 * config.pipeline_stages * config.tile_n * head_dim * dtype_bytes
    q_tile = config.tile_m * head_dim * dtype_bytes
    return kv_buffers + q_tile


def estimate_registers(config: TuneConfig, head_dim: int) -> int:
    """Approximate registers-per-thread for a FA2 config.

    The accumulator and Q tile are distributed across ``num_warps * 32``
    threads; add a small bookkeeping allowance.
    """
    threads = config.num_warps * 32
    acc = (config.tile_m * head_dim) // threads
    q_tile = (config.tile_m * head_dim) // threads
    return max(64, acc + q_tile + 32)


def prune_configs(
    configs: Iterable[TuneConfig],
    spec,
    target: TargetSpec,
    quant: Optional[QuantSpec] = None,
) -> list[TuneConfig]:
    """Drop configs that violate hard constraints.

    Constraints:
      * ``tile_k % block_size == 0`` and sub-byte packing alignment
        (``tile_k``/``tile_n`` divisible by ``pack_factor``),
      * shared-memory footprint within ``target.max_smem_bytes``,
      * register footprint within ``target.max_registers_per_thread``.
    """
    head_dim = getattr(spec, "head_dim", 64)
    kept: list[TuneConfig] = []
    for cfg in configs:
        if quant is not None:
            try:
                check_quant_alignment(cfg.tile_k, cfg.tile_n, quant)
            except ValueError:
                continue
        if estimate_smem_bytes(cfg, head_dim) > target.max_smem_bytes:
            continue
        if estimate_registers(cfg, head_dim) > target.max_registers_per_thread:
            continue
        kept.append(cfg)
    return kept


def autotune(
    configs: Sequence[TuneConfig],
    profile_fn: Callable[[TuneConfig], float],
    warmup: int = 3,
    iters: int = 20,
) -> list[TuneResult]:
    """Profile surviving configs on-device and return results sorted by time.

    ``profile_fn(config)`` must return the average device time in **seconds**
    for one forward pass.
    """
    results: list[TuneResult] = []
    for cfg in configs:
        # Warm-up then measure.
        for _ in range(warmup):
            profile_fn(cfg)
        if torch_sync() is not None:
            torch_sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            profile_fn(cfg)
        if torch_sync() is not None:
            torch_sync()
        elapsed = (time.perf_counter() - t0) / iters
        results.append(TuneResult(cfg, device_time_us=elapsed * 1e6))
    results.sort(key=lambda r: r.device_time_us)
    return results


def torch_sync():
    """Synchronize the CUDA device if torch is available; else no-op."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch
    except ImportError: # pragma: no cover
        pass
    return None


def save_recipe(path: Path, target: TargetSpec, best: TuneResult,
                extra: Optional[dict] = None) -> None:
    """Persist the best config as a probe-certified recipe in ``conf/kb/``."""
    data = {
        "target": target.name,
        "arch": target.arch,
        "config": best.config.to_dict(),
        "device_time_us": best.device_time_us,
        "correct": best.correct,
    }
    if extra:
        data.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = [
    "TuneConfig",
    "TuneResult",
    "SearchSpace",
    "estimate_smem_bytes",
    "estimate_registers",
    "prune_configs",
    "autotune",
    "save_recipe",
]
