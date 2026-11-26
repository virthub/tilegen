# src/tilegen/evolve/pricing.py
#
# Rule-based revision pricing: the ordered list of which structural
# decision to change when a stall/gate trigger reopens the topology, driven
# by profile symptoms (severity-weighted), KB-certified mechanisms (priced
# below profile evidence), and the noise floor (sub-noise suggestions are
# dropped).

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

# Symptom -> structural-decision price table (profile evidence, severity-weighted).
# Prices are per-unit-severity; higher severity raises the price linearly.
_PROFILE_PRICES: dict[str, tuple[str, str, float]] = {
    "cta_boundary_stall": ("grid_mode", "PERSISTENT_SM", 1.0),
    "mbarrier_wait_stall": ("phase_mode", "PING_PONG", 0.8),
    "sfu_stall": ("phase_mode", "PING_PONG", 0.6),
    "register_spill": ("cta_group", "2", 0.5),
    "tensorcore_idle": ("grid_mode", "PERSISTENT_SM", 0.4),
}

_KB_MULTIPLIER = 0.5 # certified mechanisms rank below live profile evidence


@dataclass
class PricingRecord:
    decision: str
    action: str
    price: float
    source: str # "profile" | "kb"


def price_revision(
    telemetry: Mapping[str, float],
    kb: Optional[object] = None,
    noise_floor: float = 0.015,
    curriculum: Optional[object] = None,
) -> list[PricingRecord]:
    """Ordered lever prices for a structural revision; empty when nothing clears."""
    del curriculum
    records: list[PricingRecord] = []
    for symptom, (decision, action, base) in _PROFILE_PRICES.items():
        severity = float(telemetry.get(symptom, 0.0))
        if severity <= noise_floor:
            continue # sub-noise evidence: do not price the lever
        records.append(PricingRecord(
            decision=decision, action=action,
            price=base * severity, source="profile",
        ))
    if kb is not None:
# Certified mechanisms matching an observed symptom are priced as
        # secondary (KB) evidence.
        for symptom, (decision, action, base) in _PROFILE_PRICES.items():
            if float(telemetry.get(symptom, 0.0)) <= noise_floor:
                continue
            try:
                recipes = kb.query_recipes(None, [symptom])
            except Exception: # noqa: BLE001
                recipes = []
            for recipe in recipes or []:
                if isinstance(recipe, dict) and recipe.get("certification_source"):
                    records.append(PricingRecord(
                        decision=decision, action=action,
                        price=base * _KB_MULTIPLIER, source="kb",
                    ))
                    break
    records.sort(key=lambda r: -r.price)
    return records


def apply_priced_revision(kernel, records: Sequence[PricingRecord], rng=None):
    """Apply the top-priced structural decision; fall back to a random reopen."""
    from dataclasses import replace

    from tilegen.evolve.mutation import reopen_m0

    for rec in records:
        if rec.decision == "grid_mode":
            from tilegen.ir.topology import GridMode

            mode = GridMode.PERSISTENT_SM if rec.action == "PERSISTENT_SM" \
                else GridMode.DATA_PARALLEL
            grid = replace(kernel.topology.grid, mode=mode)
            return replace(kernel, topology=replace(kernel.topology, grid=grid))
        if rec.decision == "phase_mode":
            from tilegen.ir.topology import PhaseMode

            phase = PhaseMode.PING_PONG if rec.action == "PING_PONG" \
                else PhaseMode.SERIAL
            roles = replace(kernel.topology.roles, phase_mode=phase)
            return replace(kernel, topology=replace(kernel.topology, roles=roles))
        if rec.decision == "cta_group":
            grid = replace(kernel.topology.grid, cta_group=int(rec.action))
            return replace(kernel, topology=replace(kernel.topology, grid=grid))
    return reopen_m0(kernel, rng)


__all__ = ["PricingRecord", "price_revision", "apply_priced_revision"]
