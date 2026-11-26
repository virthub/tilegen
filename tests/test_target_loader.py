# tests/test_target_loader.py
#
# Fixture-driven tests for conf-based target resolution (coverage gap
# §24.2.3 of tilegen.md).

import json

import pytest

from tilegen.target.loader import load_target, load_target_file
from tilegen.target.spec import BUILTIN_TARGETS, TargetSpec


class TestLoadTargetFile:
    def test_yaml_roundtrip(self, tmp_path):
        p = tmp_path / "t.yaml"
        p.write_text(
            "name: sm_90a\n"
            "arch: hopper\n"
            "sm: 90\n", encoding="utf-8")
        t = load_target_file(p)
        assert isinstance(t, TargetSpec)
        assert t.name == "sm_90a"

    def test_json_roundtrip(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text(json.dumps({"name": "sm_90a", "arch": "hopper"}),
                     encoding="utf-8")
        t = load_target_file(p)
        assert t.name == "sm_90a"

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported profile extension"):
            load_target_file(p)

    def test_non_mapping_yaml_raises(self, tmp_path):
        p = tmp_path / "t.yaml"
        p.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_target_file(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_target_file(tmp_path / "missing.yaml")


class TestLoadTarget:
    def test_builtin_name(self):
        assert load_target("sm_90a") is BUILTIN_TARGETS["sm_90a"]

    def test_path_fallback(self, tmp_path):
        p = tmp_path / "t.yaml"
        p.write_text("name: sm_90a\narch: hopper\n", encoding="utf-8")
        assert load_target(str(p)).name == "sm_90a"

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="unknown target"):
            load_target("definitely_not_a_target")
