# src/tilegen/evolve/kb_manager.py
#
# File-backed knowledge store

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional
from tilegen.evolve.interface import EvalResult, KernelCandidate
from tilegen.evolve.profiler import SYMPTOM_FIELDS
from tilegen.runtime.hash import semantic_hash
from tilegen.target.spec import TargetSpec


class FileKnowledgeStore:
    """A ``KnowledgeStore`` backed by JSON files under a directory."""

    def __init__(self, root: str | Path = "conf/kb"):
        self.root = Path(root)

    def query_recipes(self, target: TargetSpec, symptoms: list[str]) -> list[dict[str, Any]]:
        """Return certified recipes for ``target``.

        Without symptoms the previous ordering is preserved.  With symptoms,
        recipes whose ``config`` touches the knob fields associated with those
        symptoms (``profiler.SYMPTOM_FIELDS``) rank first, so agent evolvers
        get a symptom-directed warm start.
        """
        if not self.root.exists():
            return []
        symptom_fields: set[str] = set()
        for s in symptoms or []:
            symptom_fields.update(SYMPTOM_FIELDS.get(s, []))
        pairs: list[tuple[int, dict]] = []
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("target") == target.name or data.get("target") == target.sm:
                score = 0
                config = data.get("config") or {}
                if symptom_fields and isinstance(config, dict):
                    score = len(symptom_fields.intersection(config.keys()))
                pairs.append((score, data))
        if symptom_fields:
            pairs.sort(key=lambda pair: -pair[0])  # stable: higher score first
        return [data for _, data in pairs]

    def record_certified(self, recipe: dict[str, Any], result: EvalResult) -> None:
        """Write a probe-certified recipe file."""
        self.root.mkdir(parents=True, exist_ok=True)
        sid = semantic_hash({"recipe": recipe, "target": recipe.get("target", "")})[:12]
        payload = {
            "certified": True,
            "id": sid,
            "target": recipe.get("target"),
            "config": recipe.get("config"),
            "device_time_us": result.device_time_us,
            "speedup_vs_ref": result.speedup_vs_ref,
            "recorded_at": time.time(),
        }
        (self.root / f"recipe_{sid}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def promote_reference(self, candidate: KernelCandidate, result: EvalResult) -> str:
        """Promote a candidate as the sealed reference; return the reference id."""
        self.root.mkdir(parents=True, exist_ok=True)
        knobs = candidate.vtm_ir.topology.knobs.to_dict()
        sid = semantic_hash({"ref": candidate.vtm_ir.name, "knobs": knobs})[:12]
        payload = {
            "reference": True,
            "id": sid,
            "kernel": candidate.vtm_ir.name,
            "knobs": knobs,
            "speedup_vs_ref": result.speedup_vs_ref,
            "device_time_us": result.device_time_us,
            "promoted_at": time.time(),
        }
        (self.root / f"reference_{sid}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return sid


__all__ = ["FileKnowledgeStore"]
