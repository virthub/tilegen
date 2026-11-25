# src/tilegen/evolve/profiler.py
#
# Unified profiler adapter: normalized telemetry → VTM IR diagnostics

from __future__ import annotations

from typing import Mapping

_CTA_BOUNDARY = "cta_boundary_stall"
_MBARRIER_WAIT = "mbarrier_wait_stall"
_SFU_STALL = "sfu_stall"
_REGISTER_SPILL = "register_spill"
_TENSORCORE_IDLE = "tensorcore_idle"


def diagnose(telemetry: Mapping[str, float]) -> list[dict[str, str]]:
    """Map normalized stall metrics to ``(symptom, action)`` VTM IR diagnoses.

    Returns a list of ``{"symptom": ..., "action": ...}`` records ordered by
    severity (highest stall first).
    """
    out: list[dict[str, str]] = []
    ordered = [
        (_CTA_BOUNDARY, "Reopen GridTopology -> PERSISTENT_SM"),
        (_MBARRIER_WAIT, "Tune pipeline_stages / async_wait_depth / producer_warpgroups"),
        (_SFU_STALL, "Enable PhaseSchedule.PING_PONG or tune exp_emulation_ratio / rescale_threshold"),
        (_REGISTER_SPILL, "Shift RecomputeBoundary -> REGISTER_FUSED or tighten max_registers_per_thread"),
        (_TENSORCORE_IDLE, "Overlap Vector/Cube sub-tiling or increase pipeline_stages"),
    ]
    for key, action in ordered:
        value = float(telemetry.get(key, 0.0))
        if value > 0.0:
            out.append({"symptom": key, "action": action})
    return out


def normalize_ncu_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Map raw NVIDIA Nsight-Compute metric names to normalized symptom keys.

    Accepts the common ``ncu`` metric names; unknown keys are passed through.
    """
    mapping = {
        "sm__warps_active.avg.pct_of_peak_sustained_active": _TENSORCORE_IDLE,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": _MBARRIER_WAIT,
        "smsp__inst_executed_pipe_xu.sum": _SFU_STALL,
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": _REGISTER_SPILL,
    }
    out: dict[str, float] = {}
    for key, value in metrics.items():
        out[mapping.get(key, key)] = float(value)
    return out


# Symptom -> tactical-knob fields a mutation should bias toward.
SYMPTOM_FIELDS: dict[str, list[str]] = {
    _CTA_BOUNDARY: ["grid_mode"],
    _MBARRIER_WAIT: ["pipeline_stages", "async_wait_depth", "producer_warpgroups"],
    _SFU_STALL: ["exp_emulation_ratio", "rescale_threshold", "phase_mode"],
    _REGISTER_SPILL: ["max_registers_per_thread"],
    _TENSORCORE_IDLE: ["pipeline_stages", "tile_m", "tile_n"],
}


def diagnose_fields(telemetry: Mapping[str, float]) -> list[str]:
    """Map telemetry to the knob fields a mutation should prefer.

    Returns the ordered (most-severe-first) list of biased fields, which a
    strategy passes to ``mutate_knobs(..., bias=...)``.  This is the feedback
    edge that turns the evolver from a random walk into a profiler-guided one.
    """
    fields: list[str] = []
    for d in diagnose(telemetry):
        for f in SYMPTOM_FIELDS.get(d["symptom"], []):
            if f not in fields:
                fields.append(f)
    return fields


__all__ = ["diagnose", "normalize_ncu_metrics", "diagnose_fields", "SYMPTOM_FIELDS"]
