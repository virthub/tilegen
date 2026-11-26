# src/tilegen/evolve/noise.py
#
# Robust measurement statistics: the noise-floor estimation protocol and
# the direction-correct promotion gate.
#
# The noise floor is the smallest performance difference the measurement
# infrastructure can reliably distinguish from run-to-run variance. GPU
# kernel times are right-skewed, so the primary estimators are the median and
# the Median Absolute Deviation (never mean/std), warm-up runs are excluded,
# and candidate acceptance uses a conservative lower confidence bound on the
# speedup (lower-is-better latency form).

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

DEFAULT_MIN_FLOOR = 0.015 # the protocol mandates at least 1.5%
MAD_SIGMA = 1.4826 # consistent estimator of sigma under normality


@dataclass
class NoiseFloorRecord:
    """A persisted noise-floor estimate (samples, robust stats, floor)."""

    m_r: float
    mad_r: float
    sigma_r: float
    floor: float
    k: float = 2.0
    R: int = 15
    W: int = 2
    cache_state: str = "warm_l2"
    fingerprint: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NoiseFloorRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def robust_stats(samples: Sequence[float]) -> tuple[float, float, float]:
    """Return (median, MAD, sigma = 1.4826 * MAD) for a timing sample."""
    if not samples:
        raise ValueError("robust_stats: empty sample")
    med = statistics.median(samples)
    mad = statistics.median(abs(x - med) for x in samples)
    return med, mad, MAD_SIGMA * mad


def estimate_noise_floor_from_runs(
    runs: Sequence[float],
    W: int = 2,
    k: float = 2.0,
    cache_state: str = "warm_l2",
    min_floor: float = DEFAULT_MIN_FLOOR,
    fingerprint: str = "",
) -> NoiseFloorRecord:
    """Estimate the floor from an already-collected run sample.

    ``runs`` includes warm-up; the first ``W`` runs are excluded.
    """
    usable = list(runs)[W:] if len(runs) > W else list(runs)
    m, mad, sigma = robust_stats(usable)
    floor = max(min_floor, k * sigma / m) if m > 0 else min_floor
    return NoiseFloorRecord(
        m_r=m, mad_r=mad, sigma_r=sigma, floor=floor,
        k=k, R=len(usable), W=W, cache_state=cache_state,
        fingerprint=fingerprint,
    )


def estimate_noise_floor(
    sampling_fn: Callable[[], float],
    R: int = 15,
    W: int = 2,
    k: float = 2.0,
    cache_state: str = "warm_l2",
    min_floor: float = DEFAULT_MIN_FLOOR,
    fingerprint: str = "",
) -> NoiseFloorRecord:
    """Collect ``R + W`` samples via ``sampling_fn`` and estimate the floor."""
    runs = [sampling_fn() for _ in range(R + W)]
    return estimate_noise_floor_from_runs(
        runs, W=W, k=k, cache_state=cache_state, min_floor=min_floor,
        fingerprint=fingerprint,
    )


def lc_bound(
    m_c: float,
    sigma_c: float,
    m_r: float,
    sigma_r: float,
    k_c: float = 1.0,
    k_r: float = 2.0,
) -> Optional[float]:
    """Conservative lower confidence bound on speedup (latency form).

    ``LCB_speedup = (m_r - k_r*sigma_r) / (m_c + k_c*sigma_c)``; the
    candidate's plausible *slowest* must beat the reference's plausible
    *fastest*.  ``k_c = 1`` (not 2) at small candidate samples (R_cand = 5)
    to avoid over-penalizing small-sample MAD.
    """
    if m_c <= 0:
        return None
    denom = m_c + k_c * sigma_c
    if denom <= 0:
        return None
    return (m_r - k_r * sigma_r) / denom


def gate_value(noise_floor: float, bootstrap_multiplier: float) -> float:
    """Unified bootstrap gate: max(1 + noise_floor, bootstrap_multiplier)."""
    return max(1.0 + noise_floor, bootstrap_multiplier)


def campaign_noise(
    evaluator, initial_kernel
) -> tuple[float, Optional[tuple[float, float, float]]]:
    """(noise_floor, ref_stats) for a strategy campaign.

    Uses the evaluator's ``sample_times`` when available (uncached samples,
    warm-up excluded); falls back to the 1.5% lower bound when the evaluator
    cannot sample (mocks, non-GPU paths).
    """
    from tilegen.evolve.evaluator import knobs_to_params

    sampler = getattr(evaluator, "sample_times", None)
    if sampler is None:
        return DEFAULT_MIN_FLOOR, None
    try:
        runs = sampler(
            knobs_to_params(initial_kernel.topology.knobs),
            "fa2_bf16", iters=17,
        )
        rec = estimate_noise_floor_from_runs(runs, W=2)
        return rec.floor, (rec.m_r, rec.mad_r, rec.sigma_r)
    except Exception:
        return DEFAULT_MIN_FLOOR, None


__all__ = [
    "NoiseFloorRecord",
    "campaign_noise",
    "DEFAULT_MIN_FLOOR",
    "robust_stats",
    "estimate_noise_floor",
    "estimate_noise_floor_from_runs",
    "lc_bound",
    "gate_value",
]
