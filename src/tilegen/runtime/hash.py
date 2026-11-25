# src/tilegen/runtime/hash.py
#
# Deterministic semantic hashing for AOT artifact dispatch

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any


def canonicalize(obj: Any) -> Any:
    """Recursively convert arbitrary IR objects to JSON-serializable primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (tuple, list)):
        return [canonicalize(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [canonicalize(x) for x in sorted(obj, key=lambda x: repr(x))]
    if isinstance(obj, dict):
        return {str(k): canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if hasattr(obj, "to_dict"):
        return canonicalize(obj.to_dict())
    if hasattr(obj, "__dataclass_fields__"):
        return canonicalize(
            {k: getattr(obj, k) for k in sorted(obj.__dataclass_fields__)}
        )
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Canonical, stable JSON serialization (sorted keys, no whitespace)."""
    return json.dumps(
        canonicalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def semantic_hash(obj: Any, length: int = 40) -> str:
    """Return a deterministic semantic hash over ``obj``'s canonical JSON."""
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest[:length]


__all__ = ["canonicalize", "canonical_json", "semantic_hash"]
