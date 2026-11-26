# tests/test_probe.py
#
# Tests for the probe channel: closed treatment registry, typed parameters,
# promotion-free schema, comparative verdicts, and mechanism certification.

import pytest

from tilegen.evolve.interface import EvalResult, KernelCandidate
from tilegen.evolve.kb_manager import FileKnowledgeStore
from tilegen.evolve.probe import (
    TREATMENTS,
    ProbeResult,
    ProbeSpec,
    certify,
    run_probe,
    validate_spec,
)
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec, build_attention_vtm
from tilegen.target.spec import get_target


def _kernel(pipeline_stages=2):
    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    k = build_attention_vtm(spec, get_target("sm_90a"), variant="fa2_prefill")
    from dataclasses import replace

    knobs = replace(k.topology.knobs, pipeline_stages=pipeline_stages)
    return replace(k, topology=replace(k.topology, knobs=knobs))


class _StageEvaluator:
    """Time depends on the candidate's pipeline_stages knob."""

    def profile_reference(self, sealed_ref_id, target):
        del sealed_ref_id, target
        return {"device_time_us": 100.0}

    def evaluate(self, candidates, target, sealed_ref_id=None):
        del target, sealed_ref_id
        out = []
        for c in candidates:
            stages = c.vtm_ir.topology.knobs.pipeline_stages
            t = 90.0 if stages <= 2 else 70.0
            out.append(EvalResult(
                c.candidate_id, True, True, device_time_us=t,
                speedup_vs_ref=100.0 / t, max_rel_error=0.0,
            ))
        return out


class TestValidation:
    def test_valid_spec(self):
        spec = ProbeSpec(id="p1", treatment="set_pipeline_stages",
                         treatment_params={"stages": 4})
        assert validate_spec(spec) == []

    def test_rejects_unknown_treatment(self):
        spec = ProbeSpec(id="p1", treatment="patch_binary")
        assert any("not in registry" in e for e in validate_spec(spec))

    def test_rejects_untyped_params(self):
        spec = ProbeSpec(id="p1", treatment_params={"stages": [4]})
        assert any("untyped" in e for e in validate_spec(spec))

    def test_rejects_missing_ids(self):
        assert validate_spec(ProbeSpec(id="", treatment="set_tile")) != []
        assert validate_spec(
            ProbeSpec(id="p1", target_kernel_id="")) != []


class TestRunProbe:
    def test_confirmed_verdict(self):
        spec = ProbeSpec(id="p1", hypothesis="deeper pipeline wins",
                         treatment="set_pipeline_stages",
                         treatment_params={"stages": 4},
                         verdict_threshold=0.1)
        result = run_probe(spec, _kernel(2), _StageEvaluator(),
                           get_target("sm_90a"))
        assert result.applied and result.verdict == "confirmed"
        assert result.delta == pytest.approx((90.0 - 70.0) / 90.0)

    def test_not_confirmed_below_threshold(self):
        spec = ProbeSpec(id="p1", treatment="set_pipeline_stages",
                         treatment_params={"stages": 4},
                         verdict_threshold=0.5)
        result = run_probe(spec, _kernel(2), _StageEvaluator(),
                           get_target("sm_90a"))
        assert result.verdict == "not_confirmed"

    def test_invalid_spec_returns_unapplied(self):
        result = run_probe(
            ProbeSpec(id="p1", treatment="nope"), _kernel(2),
            _StageEvaluator(), get_target("sm_90a"))
        assert not result.applied and result.errors

    def test_reopen_m0_toggles_structure(self):
        from tilegen.ir.topology import GridMode

        kernel = _kernel()
        before = kernel.topology.grid.mode
        treated = TREATMENTS["reopen_m0"](kernel, {})
        assert treated.topology.grid.mode != before
        assert {before, treated.topology.grid.mode} <= {
            GridMode.DATA_PARALLEL, GridMode.PERSISTENT_SM}

    def test_result_has_no_promotion_fields(self):
        result = ProbeResult(probe_id="p1", applied=False)
        assert not hasattr(result, "becomes_reference")
        assert not hasattr(result, "promotion_decision")
        assert not hasattr(result, "decision")


class TestCertification:
    def test_certify_writes_provenance(self, tmp_path):
        from tilegen.evolve.kb_manager import FileKnowledgeStore as KB

        kb = KB(tmp_path)
        spec = ProbeSpec(id="probe_017", mechanism_target="pipeline_depth",
                         treatment="set_pipeline_stages",
                         treatment_params={"stages": 4},
                         verdict_threshold=0.1, certification=True)
        result = run_probe(spec, _kernel(2), _StageEvaluator(),
                           get_target("sm_90a"))
        ok = certify(spec, result, kb, get_target("sm_90a"),
                     EvalResult("x", True, True))
        assert ok
        import json

        recipes = [json.loads(p.read_text(encoding="utf-8"))
                   for p in tmp_path.glob("recipe_*.json")]
        assert recipes and recipes[0]["certified"] is True
        assert recipes[0]["config"] == {"stages": 4}
        assert recipes[0]["certification_source"] == "probe_017"

    def test_no_certification_without_flag(self, tmp_path):
        kb = FileKnowledgeStore(tmp_path)
        spec = ProbeSpec(id="p1", treatment="set_pipeline_stages",
                         treatment_params={"stages": 4},
                         verdict_threshold=0.1, certification=False)
        result = run_probe(spec, _kernel(2), _StageEvaluator(),
                           get_target("sm_90a"))
        assert not certify(spec, result, kb, get_target("sm_90a"),
                           EvalResult("x", True, True))
        assert not list(tmp_path.glob("recipe_*.json"))
