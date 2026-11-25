# src/tilegen/ops/schedules.py
#
# Multi-generation attention schedule generators (FA1–FA4)

from __future__ import annotations

from tilegen.ir.topology import (
    ExecutionRoles,
    GridMode,
    GridTopology,
    PhaseMode,
    RasterOrder,
    TacticalKnobs,
    TopologySpec,
)
from tilegen.target.spec import TargetSpec

VARIANTS = ("fa1", "fa2_prefill", "fa2_decode", "fa3", "fa4")


def resolve_variant(target: TargetSpec, variant: str = "auto") -> str:
    """Pick the generation-optimal FA variant for a target."""
    if variant != "auto":
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        return variant
    if target.supports("tcgen05"):
        return "fa4"
    if target.supports("wgmma"):
        return "fa3"
    if target.supports("mma_sync") or target.supports("mfma"):
        return "fa2_prefill"
    return "fa1"


def attention_topology(variant: str, target: TargetSpec) -> TopologySpec:
    """Return the $M_0$ topology + $M_1..M_k$ knobs for a FA variant."""
    v = resolve_variant(target, variant)

    if v == "fa1":
        # Low-SRAM reference: outer-KV / inner-Q tiled online softmax.
        return TopologySpec(
            grid=GridTopology(GridMode.DATA_PARALLEL, RasterOrder.ROW_MAJOR),
            roles=ExecutionRoles(1, 1, PhaseMode.SERIAL),
            knobs=TacticalKnobs(
                tile_m=64, tile_n=64, tile_k=64, pipeline_stages=1,
                swizzle_mode="NONE", max_registers_per_thread=128,
            ),
        )

    if v == "fa2_prefill":
        # Ampere: outer-Q / inner-KV warp-partitioned (Split-Q) prefill.
        return TopologySpec(
            grid=GridTopology(GridMode.DATA_PARALLEL, RasterOrder.ROW_MAJOR),
            roles=ExecutionRoles(1, 1, PhaseMode.SERIAL),
            knobs=TacticalKnobs(
                tile_m=128, tile_n=128, tile_k=64, pipeline_stages=2,
                swizzle_mode="NONE",
            ),
        )

    if v == "fa2_decode":
        # Ampere decode: parallel Split-KV with log-sum-exp (lse) reduction.
        return TopologySpec(
            grid=GridTopology(GridMode.SPLIT_KV, RasterOrder.ROW_MAJOR, split=4),
            roles=ExecutionRoles(1, 1, PhaseMode.SERIAL),
            knobs=TacticalKnobs(
                tile_m=128, tile_n=128, tile_k=64, pipeline_stages=2,
                swizzle_mode="NONE",
            ),
        )

    if v == "fa3":
        # Hopper: producer-consumer warp specialization + TMA + PING_PONG.
        return TopologySpec(
            grid=GridTopology(GridMode.PERSISTENT_SM, RasterOrder.SWIZZLED_L2),
            roles=ExecutionRoles(1, 2, PhaseMode.PING_PONG),
            knobs=TacticalKnobs(
                tile_m=128, tile_n=128, tile_k=64, pipeline_stages=3,
                async_wait_depth=1, max_registers_per_thread=168, swizzle_mode="B64",
            ),
        )

    # fa4 — Blackwell: tcgen05 + TMEM + 2-CTA cooperative MMA + hybrid exp2.
    cta_group = 2 if target.supports("2cta") else 1
    return TopologySpec(
        grid=GridTopology(GridMode.PERSISTENT_SM, RasterOrder.SWIZZLED_L2, cta_group=cta_group),
        roles=ExecutionRoles(1, 2, PhaseMode.PING_PONG),
        knobs=TacticalKnobs(
            tile_m=128, tile_n=128, tile_k=64, pipeline_stages=4,
            async_wait_depth=1, max_registers_per_thread=232, swizzle_mode="B128",
            exp_emulation_ratio=0.25, rescale_threshold=0.5,
        ),
    )


__all__ = ["VARIANTS", "resolve_variant", "attention_topology"]
