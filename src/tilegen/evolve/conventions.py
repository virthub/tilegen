# src/tilegen/evolve/conventions.py
#
# YAML schema layer for curriculum / probe configuration interchange: the
# YAML hygiene conventions (2-space sequence indent under the parent
# key, 4-space mapping continuation, no inline comments inside mappings, no
# tabs) are enforced by a line-based validator so agent-generated configs
# fail pre-parse with actionable messages instead of misparsing.

from __future__ import annotations

from typing import Any, Optional


def validate_yaml_text(text: str) -> list[str]:
    """Return a list of style violations (empty means the text conforms)."""
    violations: list[str] = []
    last_key_indent = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if "\t" in line:
            violations.append(f"line {lineno}: tab character (use spaces)")
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent % 2 != 0:
            violations.append(
                f"line {lineno}: indent {indent} is not a multiple of 2")
        # Sequence items sit 2 spaces under their parent key.
        if stripped.startswith("- "):
            if indent != last_key_indent + 2:
                violations.append(
                    f"line {lineno}: sequence item indented {indent} spaces; "
                    f"the convention is {last_key_indent + 2} (2 spaces "
                    "under the parent key)")
            continue
        # No inline comments inside mappings (comments precede blocks).
        if ":" in line and "#" in line:
            before_comment = line.split("#", 1)[0]
            if ":" in before_comment:
                violations.append(
                    f"line {lineno}: inline comment inside a mapping; "
                    "place comments on their own line above the block")
        if stripped.endswith(":"):
            last_key_indent = indent
    return violations


def dumps(obj: Any) -> str:
    """Serialize a schema object to convention-conforming YAML text."""
    import yaml # type: ignore

    text = yaml.safe_dump(obj, sort_keys=False, default_flow_style=False)
    violations = validate_yaml_text(text)
    if violations:
        raise ValueError(f"generated YAML violates the schema conventions: "
                         f"{violations}")
    return text


def loads(text: str) -> Any:
    """Parse convention-conforming YAML text into a dict."""
    import yaml # type: ignore

    violations = validate_yaml_text(text)
    if violations:
        raise ValueError(
            "YAML violates the schema conventions: " + "; ".join(violations))
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("YAML must be a mapping at the top level")
    return data


__all__ = ["validate_yaml_text", "dumps", "loads"]
