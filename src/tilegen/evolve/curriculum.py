# src/tilegen/evolve/curriculum.py
#
# Curriculum representation for the campaign: the serializable, diffable
# record of the structural decisions, per-class trigger thresholds,
# noise-floor record, and bootstrap multiplier that the reference-guided
# strategies carry across a campaign.

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

DEFAULT_THRESHOLDS = {
    "N_stall": 2,
    "N_gate": 2,
    "graph_exhaustion_ratio": 0.9,
    "failure_concentration_threshold": 0.7,
}


@dataclass
class Curriculum:
    """One campaign's curriculum.

    Mirrors the meta-evolver specification's curriculum schema: thresholds
    are per-operator-class configuration parameters, not global defaults.
    """

    operator_class: str = "attention"
    version: str = "gen_000"
    reference_id: str = "ref_000"
    m0_epoch: int = 0
    thresholds: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    noise_floor_record: str = ""
    active_bootstrap_multiplier: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Curriculum":
        valid = {f.name for f in fields(cls)}
        merged = {**DEFAULT_THRESHOLDS, **(data.get("thresholds") or {})}
        kwargs = {k: v for k, v in data.items() if k in valid}
        kwargs["thresholds"] = merged
        return cls(**kwargs)

    @classmethod
    def load(
        cls,
        budget: Optional[dict] = None,
        operator_class: str = "attention",
    ) -> "Curriculum":
        """Build from a budget dict (``budget["curriculum"]`` overrides)."""
        raw = dict((budget or {}).get("curriculum") or {})
        raw.setdefault("operator_class", operator_class)
        return cls.from_dict(raw)

    def to_yaml(self) -> str:
        """Serialize to convention-conforming YAML (the schema-hygiene conventions)."""
        from tilegen.evolve.conventions import dumps

        return dumps(self.to_dict())

    @classmethod
    def from_yaml(cls, text: str) -> "Curriculum":
        """Parse convention-conforming YAML back into a Curriculum."""
        from tilegen.evolve.conventions import loads

        return cls.from_dict(loads(text))


__all__ = ["Curriculum", "DEFAULT_THRESHOLDS"]
