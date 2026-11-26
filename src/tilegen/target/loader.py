# src/tilegen/target/loader.py
#
# TargetSpec loaders from YAML/JSON profiles (conf/target/*.yaml)

from __future__ import annotations

import json
from pathlib import Path
from typing import Union
from tilegen.target.spec import TargetSpec


def _load_yaml(path: Path) -> dict:
    try:
        import yaml # type: ignore
    except ImportError as exc: # pragma: no cover - pyyaml is a core dependency
        raise RuntimeError("PyYAML is required to load YAML profiles") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"profile {path} must contain a mapping at top level")
    return data


def load_target_file(path: Union[str, Path]) -> TargetSpec:
    """Load a single ``TargetSpec`` from a YAML or JSON file."""
    p = Path(path)
    if p.suffix.lower() in (".yaml", ".yml"):
        data = _load_yaml(p)
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported profile extension: {p.suffix!r}")
    return TargetSpec.from_dict(data)


def load_target(name_or_path: Union[str, Path]) -> TargetSpec:
    """Resolve ``name_or_path`` to a ``TargetSpec``.

    A bare name (``sm_90a``) resolves against the built-in registry; a path
    resolves against a YAML/JSON profile file.
    """
    from tilegen.target.spec import BUILTIN_TARGETS

    s = str(name_or_path)
    if s in BUILTIN_TARGETS:
        return BUILTIN_TARGETS[s]
    if Path(s).exists():
        return load_target_file(s)
    raise KeyError(f"unknown target {s!r}")


__all__ = ["load_target", "load_target_file"]
