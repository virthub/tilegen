# src/tilegen/runtime/cache.py
#
# Python-side AOT artifact cache

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from tilegen.runtime.hash import semantic_hash

DEFAULT_CACHE_DIR = ".tilegen_cache"


@dataclass
class ArtifactRecord:
    """A single compiled-kernel manifest entry."""

    semantic_id: str
    kernel_name: str
    backend: str
    target: str
    spec: dict
    artifact: Optional[str] = None # path to .cubin/.hsaco/.so relative to cache dir
    meta: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "semantic_id": self.semantic_id,
            "kernel_name": self.kernel_name,
            "backend": self.backend,
            "target": self.target,
            "spec": self.spec,
            "artifact": self.artifact,
            "meta": self.meta,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArtifactRecord":
        return cls(
            semantic_id=data["semantic_id"],
            kernel_name=data["kernel_name"],
            backend=data["backend"],
            target=data["target"],
            spec=data.get("spec", {}),
            artifact=data.get("artifact"),
            meta=data.get("meta", {}),
            created_at=data.get("created_at", 0.0),
        )


class AOTArtifactCache:
    """Content-addressed on-disk cache keyed by semantic hash."""

    def __init__(self, root: Optional[Path | str] = None) -> None:
        self.root = Path(root) if root else Path(os.environ.get("TILEGEN_CACHE_DIR", DEFAULT_CACHE_DIR))

    def _record_path(self, semantic_id: str) -> Path:
        return self.root / "manifests" / f"{semantic_id}.json"

    def lookup(self, semantic_id: str) -> Optional[ArtifactRecord]:
        p = self._record_path(semantic_id)
        if not p.exists():
            return None
        return ArtifactRecord.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def store(self, record: ArtifactRecord) -> ArtifactRecord:
        p = self._record_path(record.semantic_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return record

    def record_for(self, kernel: Any, target: Any, backend: str) -> ArtifactRecord:
        spec = kernel.to_dict() if hasattr(kernel, "to_dict") else dict(kernel)
        target_dict = target.to_dict() if hasattr(target, "to_dict") else dict(target)
        full_spec = {"kernel": spec, "target": target_dict, "backend": backend}
        sid = semantic_hash(full_spec)
        return ArtifactRecord(
            semantic_id=sid,
            kernel_name=spec.get("name", "kernel"),
            backend=backend,
            target=target_dict.get("name", target_dict.get("sm", "unknown")),
            spec=full_spec,
        )


__all__ = ["AOTArtifactCache", "ArtifactRecord", "DEFAULT_CACHE_DIR"]
