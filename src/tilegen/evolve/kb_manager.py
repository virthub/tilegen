# src/tilegen/evolve/kb_manager.py
#
# File-backed knowledge store

from __future__ import annotations

import json
import os
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
            if not isinstance(data, dict): # skip arrays (rejections/metrics)
                continue
            if data.get("target") == target.name or data.get("target") == target.sm:
                score = 0
                config = data.get("config") or {}
                if symptom_fields and isinstance(config, dict):
                    score = len(symptom_fields.intersection(config.keys()))
                pairs.append((score, data))
        if symptom_fields:
            pairs.sort(key=lambda pair: -pair[0]) # stable: higher score first
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
            # Provenance line: the certifying probe id (never a location).
            "certification_source": recipe.get("Certification source"),
            "device_time_us": result.device_time_us,
            "speedup_vs_ref": result.speedup_vs_ref,
            "recorded_at": time.time(),
        }
        (self.root / f"recipe_{sid}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    # -- compare-and-swap promotion registry (atomic rename, POSIX) --------

    def _registry_path(self) -> Path:
        return self.root / "current_reference.json"

    def _read_current(self) -> Optional[str]:
        try:
            data = json.loads(self._registry_path().read_text(encoding="utf-8"))
            return data.get("reference_id")
        except (json.JSONDecodeError, OSError):
            return None

    def _write_current(self, reference_id: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"reference_id": reference_id}, indent=2)
        tmp = self._registry_path().with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self._registry_path())

    def _record_rejection(self, candidate_id: str, expected: str,
                          current: str, result: EvalResult) -> None:
        path = self.root / "rejections.json"
        entries = []
        if path.exists():
            try:
                entries = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                entries = []
        entries.append({
            "candidate_id": candidate_id,
            "expected_reference_id": expected,
            "current_reference_id_at_cas": current,
            "decision": "rejected_stale",
            "speedup_vs_ref": result.speedup_vs_ref,
            "ts": time.time(),
        })
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def promote_reference(
        self,
        candidate: KernelCandidate,
        result: EvalResult,
        expected_reference_id: Optional[str] = None,
    ) -> str:
        """Promote a candidate as the sealed reference; return the reference id.

        Compare-and-swap: when
        ``expected_reference_id`` is given and does not match the current
        registry entry, the promotion is rejected as stale (recorded in
        ``rejections.json``) and the empty string is returned.  The registry
        update is an atomic rename, so concurrent readers always see a
        complete record.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        current = self._read_current()
        if (expected_reference_id is not None and current is not None
                and expected_reference_id != current):
            self._record_rejection(
                candidate.candidate_id, expected_reference_id, current, result)
            return ""
        knobs = candidate.vtm_ir.topology.knobs.to_dict()
        sid = semantic_hash({"ref": candidate.vtm_ir.name, "knobs": knobs})[:12]
        payload = {
            "reference": True,
            "id": sid,
            "decision": "accepted",
            "kernel": candidate.vtm_ir.name,
            "knobs": knobs,
            "speedup_vs_ref": result.speedup_vs_ref,
            "device_time_us": result.device_time_us,
            "current_reference_id_at_cas": current,
            "becomes_reference_for": "next_campaign",
            # Confirmation block (populated when robust stats were measured).
            "confirmation": {
                "m_c": result.m_c,
                "sigma_c": result.sigma_c,
                "lcb_speedup": result.lcb_speedup,
            },
            "promoted_at": time.time(),
        }
        (self.root / f"reference_{sid}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        self._write_current(sid)
        return sid


__all__ = ["FileKnowledgeStore"]
