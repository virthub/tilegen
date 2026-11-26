# tests/test_adapter.py
#
# Unit tests for the evolver adaptation layer (adapter, from_dict, KB, registry)

import asyncio
import json
import pytest
from tilegen.evolve.adapter import (
    AsyncRemoteEvaluator,
    RemoteKernelEvaluator,
    candidate_from_dict,
    candidate_to_dict,
    eval_result_from_dict,
    feedback_json,
    format_feedback,
    normalize_variant,
    propose_candidates,
)
from tilegen.evolve.interface import EvalResult, KernelCandidate
from tilegen.evolve.kb_manager import FileKnowledgeStore
from tilegen.evolve.strategies import ENGINES, get_engine, register_engine
from tilegen.ir import vtm_kernel_from_dict, vtm_kernel_from_json
from tilegen.ir.ops import VTMKernel
from tilegen.ops.attention import AttentionSpec, build_decode_vtm, build_dense_attention_vtm
from tilegen.runtime.hash import semantic_hash
from tilegen.target import get_target


def _dense() -> VTMKernel:
    spec = AttentionSpec(batch=2, num_heads=4, head_dim=64, seq_len=256, causal=True)
    return build_dense_attention_vtm(spec, get_target("sm_120"), variant="fa3")


def _decode() -> VTMKernel:
    spec = AttentionSpec(
        batch=4, num_heads=8, num_kv_heads=2, head_dim=64, seq_len=1024,
        page_size=16, mode="decode",
    )
    return build_decode_vtm(spec, get_target("sm_90a"))


def test_dense_round_trip():
    kernel = _dense()
    data = kernel.to_dict()
    rebuilt = vtm_kernel_from_dict(data)
    assert rebuilt.to_dict() == data
    assert semantic_hash(rebuilt) == semantic_hash(kernel)


def test_decode_round_trip():
    kernel = _decode()
    data = kernel.to_dict()
    rebuilt = vtm_kernel_from_dict(data)
    assert rebuilt.to_dict() == data


def test_vtm_kernel_from_json():
    kernel = _dense()
    rebuilt = vtm_kernel_from_json(json.dumps(kernel.to_dict()))
    assert rebuilt.to_dict() == kernel.to_dict()


def test_vtm_kernel_from_dict_classmethod():
    kernel = _dense()
    assert VTMKernel.from_dict(kernel.to_dict()).to_dict() == kernel.to_dict()


def test_from_dict_verifies_invalid():
    data = _dense().to_dict()
    data["body"].append(
        {"op": "matmul", "a": "missing_buffer", "b": "q_tile", "acc": "acc", "role": "matrix_compute"}
    )
    with pytest.raises(ValueError):
        vtm_kernel_from_dict(data)
    vtm_kernel_from_dict(data, verify=False) # verify=False skips structural checks


def test_from_dict_rejects_non_dict():
    with pytest.raises(TypeError):
        vtm_kernel_from_dict(["not", "a", "dict"])


@pytest.mark.parametrize(
    "alias,canon",
    [
        ("sdpa", "sdpa"),
        ("fa1", "fa1"),
        ("fa2_bf16", "fa2_prefill"),
        ("fa2_prefill", "fa2_prefill"),
        ("fa2_decode", "fa2_decode"),
        ("fa3", "fa3"),
        ("fa3_tma", "fa3"),
        ("fa3_ws", "fa3"),
        ("fa4", "fa4"),
        ("fa4_tilelang", "fa4"),
    ],
)
def test_normalize_variant(alias, canon):
    assert normalize_variant(alias) == canon


def test_normalize_variant_unknown():
    with pytest.raises(ValueError):
        normalize_variant("fa5")


def test_candidate_dict_round_trip():
    kernel = _dense()
    cand = KernelCandidate(
        candidate_id="x1", parent_ids=["p0"], vtm_ir=kernel,
        backend="triton", variant="fa3_ws", rationale="test",
    )
    rebuilt = candidate_from_dict(candidate_to_dict(cand))
    assert rebuilt.candidate_id == "x1"
    assert rebuilt.parent_ids == ["p0"]
    assert rebuilt.variant == "fa3"
    assert rebuilt.backend == "triton"
    assert rebuilt.vtm_ir.to_dict() == kernel.to_dict()


def test_propose_candidates_list_and_dict_payloads():
    kernel = _dense()
    item = {"candidate_id": "a", "vtm_ir": kernel.to_dict(), "variant": "fa4_tilelang", "backend": "tilelang"}
    candidates = propose_candidates([item])
    assert len(candidates) == 1
    assert candidates[0].variant == "fa4" and candidates[0].backend == "tilelang"
    assert len(propose_candidates({"candidates": [item]})) == 1


def test_propose_candidates_rejects_invalid():
    kernel = _dense()
    with pytest.raises(ValueError, match="candidate 1 invalid"):
        propose_candidates([{"vtm_ir": kernel.to_dict()}, {"vtm_ir": kernel.to_dict(), "variant": "nope"}])
    with pytest.raises(ValueError):
        propose_candidates({"nope": []})
    with pytest.raises(ValueError):
        propose_candidates([{"vtm_ir": kernel.to_dict(), "backend": "cuda"}])


def test_eval_result_from_dict():
    result = eval_result_from_dict(
        {"candidate_id": "r", "compiled": True, "correct": False, "device_time_us": 7.5, "junk": 1}
    )
    assert result.candidate_id == "r"
    assert result.compiled and not result.correct
    assert result.device_time_us == 7.5


def test_format_feedback_and_feedback_json():
    results = [
        EvalResult("a", True, True, device_time_us=120.0, speedup_vs_ref=1.25, max_rel_error=1e-4),
        EvalResult("b", False, False, error_log="compile failed", telemetry={"sfu_stall": 5.0}),
    ]
    text = format_feedback(results)
    assert "[PASS] a" in text and "1.25x" in text
    assert "[FAIL] b" in text and "compile failed" in text
    assert "sfu_stall" in text
    js = feedback_json(results)
    assert js[0]["candidate_id"] == "a"
    assert js[1]["compiled"] is False
    assert any(d["symptom"] == "sfu_stall" for d in js[1]["diagnoses"])


def test_remote_evaluator():
    def submit(payload, target_name, sealed_ref_id):
        assert target_name == "sm_120"
        assert sealed_ref_id == "gen-0"
        assert payload[0]["candidate_id"] == "r1"
        return [{"candidate_id": "r1", "compiled": True, "correct": True, "device_time_us": 42.0}]

    evaluator = RemoteKernelEvaluator(submit)
    kernel = _dense()
    results = evaluator.evaluate(
        [KernelCandidate("r1", [], vtm_ir=kernel)], get_target("sm_120"), sealed_ref_id="gen-0"
    )
    assert results[0].compiled and results[0].correct
    assert results[0].device_time_us == 42.0


def test_async_remote_evaluator():
    async def submit(payload, target_name, sealed_ref_id):
        return [{"candidate_id": "r1", "compiled": True, "correct": True}]

    evaluator = AsyncRemoteEvaluator(submit)
    kernel = _dense()
    results = asyncio.run(
        evaluator.evaluate([KernelCandidate("r1", [], vtm_ir=kernel)], get_target("sm_120"))
    )
    assert results[0].compiled


def test_register_engine():
    class DummyEngine:
        def run_campaign(self, *args, **kwargs):
            return None

    register_engine("dummy", DummyEngine)
    try:
        assert isinstance(get_engine("dummy"), DummyEngine)
    finally:
        ENGINES.pop("dummy", None)
    with pytest.raises(ValueError):
        register_engine("", DummyEngine)
    with pytest.raises(ValueError):
        register_engine("none", None)


def test_kb_query_recipes_symptom_ranking(tmp_path):
    kb = FileKnowledgeStore(str(tmp_path))
    (tmp_path / "recipe_a.json").write_text(
        json.dumps({"target": "sm_120", "config": {"tile_m": 128}}), encoding="utf-8"
    )
    (tmp_path / "recipe_b.json").write_text(
        json.dumps({"target": "sm_120", "config": {"exp_emulation_ratio": 0.5}}), encoding="utf-8"
    )
    target = get_target("sm_120")
    ranked = kb.query_recipes(target, ["sfu_stall"])
    assert ranked[0]["config"].get("exp_emulation_ratio") == 0.5
    assert len(kb.query_recipes(target, [])) == 2


def test_load_module_from_source():
    from tilegen.evolve.evaluator import _load_module_from_source

    mod = _load_module_from_source("def hello():\n    return 42\n", "adapter_test_mod")
    assert mod.hello() == 42
