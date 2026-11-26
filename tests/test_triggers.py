# tests/test_triggers.py
#
# Tests for Phase-2 curriculum + trigger detector.

import pytest

from tilegen.evolve.curriculum import Curriculum, DEFAULT_THRESHOLDS
from tilegen.evolve.triggers import (
    TRIGGER_GATE,
    TRIGGER_STALL,
    TriggerDetector,
)


class TestCurriculum:
    def test_defaults(self):
        c = Curriculum.load(None, operator_class="dense_gemm")
        assert c.operator_class == "dense_gemm"
        assert c.thresholds == DEFAULT_THRESHOLDS
        assert c.m0_epoch == 0

    def test_roundtrip(self):
        c = Curriculum(operator_class="x", version="v2", m0_epoch=3,
                       thresholds={"N_stall": 5})
        d = c.to_dict()
        c2 = Curriculum.from_dict(d)
        assert c2.m0_epoch == 3
        assert c2.thresholds["N_stall"] == 5
        # Missing threshold keys fall back to defaults.
        assert c2.thresholds["N_gate"] == 2

    def test_load_budget_override(self):
        c = Curriculum.load(
            {"curriculum": {"thresholds": {"N_stall": 9},
                            "active_bootstrap_multiplier": 1.05}})
        assert c.thresholds["N_stall"] == 9
        assert c.active_bootstrap_multiplier == 1.05


class TestTriggerDetector:
    def test_stall_fires_and_rearms(self):
        d = TriggerDetector({"N_stall": 2})
        assert d.update(improved=False) == []
        assert d.update(improved=False) == [TRIGGER_STALL]
        # Re-armed: one more stall does not fire again immediately.
        assert d.update(improved=False) == []

    def test_improvement_resets_counters(self):
        d = TriggerDetector({"N_stall": 2, "N_gate": 2})
        assert d.update(improved=False) == []
        assert d.update(improved=True) == []
        assert d.update(improved=False) == []

    def test_gate_impasse_requires_concentration(self):
        d = TriggerDetector({"N_stall": 9, "N_gate": 2,
                             "failure_concentration_threshold": 0.6})
        # Two gate failures, but only one localized to M0 (0.5 < 0.6).
        assert d.update(improved=False, gate_failed=True, m0_localized=True) == []
        assert d.update(improved=False, gate_failed=True, m0_localized=False) == []
        # Third failure localized -> concentration 2/3 >= 0.6 and N_gate met.
        assert d.update(improved=False, gate_failed=True,
                        m0_localized=True) == [TRIGGER_GATE]

    def test_graph_exhaustion(self):
        d = TriggerDetector({})
        assert not d.graph_exhausted(5, 10)
        assert d.graph_exhausted(9, 10)
        assert not d.graph_exhausted(3, 0)


class TestCasPromotion:
    def _cand(self, name="k", cid="c1"):
        from dataclasses import replace

        from tilegen.evolve.interface import EvalResult, KernelCandidate

        kernel = replace(_kernel(), name=name)
        return KernelCandidate(cid, [], kernel), EvalResult(
            cid, True, True, device_time_us=90.0, speedup_vs_ref=1.1)

    def test_cas_accept_then_stale_reject(self, tmp_path):
        from tilegen.evolve.kb_manager import FileKnowledgeStore

        kb = FileKnowledgeStore(tmp_path)
        cand, res = self._cand()
        sid1 = kb.promote_reference(cand, res) # first promotion, no expectation
        assert sid1
        registry = __import__("json").loads(
            (tmp_path / "current_reference.json").read_text(encoding="utf-8"))
        assert registry["reference_id"] == sid1
# Second promotion (a different candidate) with the correct
        # expectation succeeds.
        cand2, res2 = self._cand(name="k2", cid="c2")
        sid2 = kb.promote_reference(cand2, res2, expected_reference_id=sid1)
        assert sid2 and sid2 != sid1
        # A stale async verifier (old expectation) is rejected and logged.
        assert kb.promote_reference(cand, res, expected_reference_id=sid1) == ""
        rejects = __import__("json").loads(
            (tmp_path / "rejections.json").read_text(encoding="utf-8"))
        assert rejects[-1]["decision"] == "rejected_stale"
        assert kb._read_current() == sid2 # registry unchanged by the stale call

    def test_record_schema_fields(self, tmp_path):
        from tilegen.evolve.kb_manager import FileKnowledgeStore

        kb = FileKnowledgeStore(tmp_path)
        cand, res = self._cand()
        sid = kb.promote_reference(cand, res)
        import json

        rec = json.loads(
            (tmp_path / f"reference_{sid}.json").read_text(encoding="utf-8"))
        assert rec["decision"] == "accepted"
        assert "current_reference_id_at_cas" in rec
        assert rec["confirmation"]["lcb_speedup"] is None


def _kernel():
    from tilegen.layout.quant import QuantSpec
    from tilegen.ops.attention import AttentionSpec, build_attention_vtm
    from tilegen.target.spec import get_target

    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    return build_attention_vtm(spec, get_target("sm_90a"), variant="fa2_prefill")
