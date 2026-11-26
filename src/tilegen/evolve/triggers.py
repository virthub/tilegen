# src/tilegen/evolve/triggers.py
#
# Stall/gate trigger detector: canonical trigger enums with per-class
# thresholds, replacing hardcoded plateau counters in the reference-guided
# strategies. A trigger fires when the inner loop stalls or when gate
# failures concentrate on structural decisions; the strategy responds by
# reopening the structural topology.

from __future__ import annotations

from typing import Optional

TRIGGER_STALL = "stall_without_new_best"
TRIGGER_GRAPH = "decision_graph_exhausted"
TRIGGER_GATE = "gate_impasse"
TRIGGERS = (TRIGGER_STALL, TRIGGER_GRAPH, TRIGGER_GATE)


class TriggerDetector:
    """Counts consecutive stalls and gate failures; fires canonical triggers.

    Counters reset on any improvement and re-arm after firing, so a trigger
    re-evaluates from scratch after the strategy responds.
    """

    def __init__(self, thresholds: Optional[dict] = None):
        t = thresholds or {}
        self.n_stall = int(t.get("N_stall", 2))
        self.n_gate = int(t.get("N_gate", 2))
        self.graph_ratio = float(t.get("graph_exhaustion_ratio", 0.9))
        self.concentration = float(
            t.get("failure_concentration_threshold", 0.7))
        self._stalls = 0
        self._gate_failures = 0
        self._gate_failures_m0 = 0

    def update(
        self,
        improved: bool,
        gate_failed: bool = False,
        m0_localized: bool = False,
    ) -> list[str]:
        """Feed one generation's outcome; return the list of fired triggers."""
        fired: list[str] = []
        if improved:
            self._stalls = 0
            self._gate_failures = 0
            self._gate_failures_m0 = 0
            return fired

        self._stalls += 1
        if self._stalls >= self.n_stall:
            fired.append(TRIGGER_STALL)
            self._stalls = 0 # re-arm after firing

        if gate_failed:
            self._gate_failures += 1
            if m0_localized:
                self._gate_failures_m0 += 1
            if (self._gate_failures >= self.n_gate
                    and self._gate_failures_m0
                    / max(self._gate_failures, 1) >= self.concentration):
                fired.append(TRIGGER_GATE)
                self._gate_failures = 0
                self._gate_failures_m0 = 0
        return fired

    def graph_exhausted(self, tried: int, actionable: int) -> bool:
        """True when the fraction of actionable levers tried hits the ratio."""
        if actionable <= 0:
            return False
        return tried / actionable >= self.graph_ratio


__all__ = [
    "TRIGGERS",
    "TRIGGER_STALL",
    "TRIGGER_GRAPH",
    "TRIGGER_GATE",
    "TriggerDetector",
]


class RevisionGuard:
    """Budget-modulated cap on structural revisions.

    Counts *unproductive consecutive* unfreeze cycles since the last verified
    promotion (reset on promotion); a global cap bounds total unfreezes.
    ``K_max_effective = K_max_base + floor((1 - budget_consumed_fraction) *
    K_max_slack)``.  When the fraction rises mid-run and the effective cap
    drops below the counter, further revisions are blocked (not retroactive).
    """

    def __init__(self, k_max_base: int = 2, k_max_slack: int = 2,
                 k_global_cap: int = 8):
        self.base = k_max_base
        self.slack = k_max_slack
        self.global_cap = k_global_cap
        self.consecutive = 0
        self.total = 0

    def on_promotion(self) -> None:
        self.consecutive = 0

    def allow(self, budget_consumed_fraction: float) -> bool:
        if self.total >= self.global_cap:
            return False
        k_max = self.base + int((1.0 - max(0.0, min(1.0, budget_consumed_fraction))) * self.slack)
        return self.consecutive < k_max

    def record_revision(self) -> None:
        self.consecutive += 1
        self.total += 1


__all__ = [
    "TRIGGERS",
    "TRIGGER_STALL",
    "TRIGGER_GRAPH",
    "TRIGGER_GATE",
    "TriggerDetector",
    "RevisionGuard",
]
