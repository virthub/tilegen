# tests/test_conventions.py
#
# Tests for the YAML schema layer: hygiene validator and the
# curriculum/probe YAML round-trips.

import pytest

from tilegen.evolve.curriculum import Curriculum
from tilegen.evolve.probe import ProbeSpec
from tilegen.evolve.conventions import dumps, loads, validate_yaml_text

CONFORMING = (
    "curriculum:\n"
    "  operator_class: attention\n"
    "  thresholds:\n"
    "    N_stall: 2\n"
    "    N_gate: 2\n"
    "  mechanism_lines:\n"
    "    - line_1\n"
    "    - line_2\n"
)


class TestValidator:
    def test_conforming_text(self):
        assert validate_yaml_text(CONFORMING) == []

    def test_rejects_tabs(self):
        assert any("tab" in v for v in validate_yaml_text("x:\n\t- 1\n"))

    def test_rejects_odd_indent(self):
        assert any("multiple of 2" in v
                   for v in validate_yaml_text("x:\n   y: 1\n"))

    def test_rejects_sequence_indent(self):
        assert any("2 spaces" in v
                   for v in validate_yaml_text("x:\n    - 1\n"))

    def test_rejects_inline_mapping_comment(self):
        assert any("inline comment" in v
                   for v in validate_yaml_text("x: 1  # note\n"))


class TestRoundTrips:
    def test_curriculum_yaml_roundtrip(self):
        c = Curriculum(operator_class="dense_gemm", m0_epoch=2,
                       thresholds={"N_stall": 5})
        c2 = Curriculum.from_yaml(c.to_yaml())
        assert c2.operator_class == "dense_gemm"
        assert c2.m0_epoch == 2
        assert c2.thresholds["N_stall"] == 5

    def test_probe_yaml_roundtrip(self):
        spec = ProbeSpec(id="probe_1", treatment="set_pipeline_stages",
                         treatment_params={"stages": 4},
                         verdict_threshold=0.1, certification=True)
        spec2 = ProbeSpec.from_yaml(spec.to_yaml())
        assert spec2.id == "probe_1"
        assert spec2.treatment_params == {"stages": 4}
        assert spec2.certification is True

    def test_loads_rejects_non_conforming(self):
# A top-level sequence violates the 2-space sequence-indent rule, so
        # loads() rejects it before parsing.
        with pytest.raises(ValueError, match="conventions"):
            loads("- item\n")

    def test_generated_yaml_conforms(self):
        text = dumps(Curriculum().to_dict())
        assert validate_yaml_text(text) == []
