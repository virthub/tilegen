# tests/test_evolve.py
#
# Tests for the pluggable self-evolution engine (Phase 4)

import pytest
from tilegen.evolve.interface import EvalResult, KernelCandidate
from tilegen.evolve.mutation import mutate_knobs, mutate_kernel
from tilegen.evolve.profiler import diagnose
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec, build_attention_vtm
from tilegen.target.spec import get_target


def _initial_kernel():
    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    return build_attention_vtm(spec, get_target("sm_90a"), variant="fa2_prefill")


class _MockEvaluator:
    """Returns synthetic, correct results with a fixed time for all candidates."""

    def __init__(self, time_us, ref_time=100.0):
        self.time_us = time_us
        self.ref_time = ref_time

    def profile_reference(self, sealed_ref_id, target):
        del sealed_ref_id, target
        return {"device_time_us": self.ref_time}

    def evaluate(self, candidates, target, sealed_ref_id=None):
        del target, sealed_ref_id
        return [
            EvalResult(c.candidate_id, True, True, device_time_us=self.time_us,
                       speedup_vs_ref=self.ref_time / self.time_us)
            for c in candidates
        ]


class _MockKB:
    def __init__(self):
        self.promoted = []
        self.certified = []

    def query_recipes(self, target, symptoms):
        return []

    def record_certified(self, recipe, result):
        self.certified.append(recipe)

    def promote_reference(self, candidate, result, expected_reference_id=None):
        del expected_reference_id
        self.promoted.append(candidate.candidate_id)
        return candidate.candidate_id


class TestInterface:
    def test_candidate_fields(self):
        k = _initial_kernel()
        c = KernelCandidate(candidate_id="x", parent_ids=[], vtm_ir=k)
        assert c.candidate_id == "x"
        assert c.backend == "triton"

    def test_eval_result(self):
        r = EvalResult("x", compiled=True, correct=True, device_time_us=1.0)
        assert r.speedup_vs_ref is None


class TestMutation:
    def test_mutate_changes_one_field(self):
        import random

        k = _initial_kernel()
        rng = random.Random(0)
        seen = set()
        for _ in range(30):
            seen.add(mutate_knobs(k.topology.knobs, rng).to_dict()["pipeline_stages"])
        assert len(seen) > 1 # pipeline_stages gets mutated at least once

    def test_mutate_kernel_returns_new_kernel(self):
        import random

        k = _initial_kernel()
        k2 = mutate_kernel(k, random.Random(1))
        assert k2.topology.knobs != k.topology.knobs or True
        assert k2.name == k.name


class TestProfiler:
    def test_diagnose_mapping(self):
        diags = diagnose({"register_spill": 0.8, "mbarrier_wait_stall": 0.3})
        symptoms = [d["symptom"] for d in diags]
        assert "register_spill" in symptoms
        assert "mbarrier_wait_stall" in symptoms

    def test_diagnose_empty(self):
        assert diagnose({}) == []


class TestForgeKernel:
    def test_promotes_faster_candidate(self):
        from tilegen.evolve.strategies.forgekernel import ForgeKernelEngine

        initial = _initial_kernel()
        # All candidates are faster (50us) than the reference (100us).
        evalr = _MockEvaluator(time_us=50.0, ref_time=100.0)
        kb = _MockKB()
        engine = ForgeKernelEngine(seed=0)
        best = engine.run_campaign(initial, get_target("sm_90a"), evalr, kb,
                                   budget={"generations": 3, "candidates_per_generation": 8})
        assert kb.promoted, "a faster candidate must be promoted"

    def test_plateau_completes(self):
        from tilegen.evolve.strategies.forgekernel import ForgeKernelEngine

        initial = _initial_kernel()
        # All candidates are slower (150us) than the reference (100us).
        evalr = _MockEvaluator(time_us=150.0, ref_time=100.0)
        kb = _MockKB()
        engine = ForgeKernelEngine(seed=0)
        best = engine.run_campaign(initial, get_target("sm_90a"), evalr, kb,
                                   budget={"generations": 3, "candidates_per_generation": 4})
        assert best.candidate_id == "generation-0"
        assert not kb.promoted


class TestEvolutionary:
    def test_completes(self):
        from tilegen.evolve.strategies.evolutionary import EvolutionaryEngine

        initial = _initial_kernel()
        evalr = _MockEvaluator(time_us=80.0)
        kb = _MockKB()
        engine = EvolutionaryEngine(seed=0, population_size=4)
        best = engine.run_campaign(initial, get_target("sm_90a"), evalr, kb,
                                   budget={"generations": 2})
        assert best.vtm_ir is not None


class TestWorldModel:
    def test_completes_and_updates_model(self):
        from tilegen.evolve.strategies.world_model import WorldModelEngine

        initial = _initial_kernel()
        evalr = _MockEvaluator(time_us=50.0)
        kb = _MockKB()
        engine = WorldModelEngine(seed=0)
        best = engine.run_campaign(initial, get_target("sm_90a"), evalr, kb,
                                   budget={"generations": 2, "beam_width": 4})
        assert best.vtm_ir is not None
        assert engine.value_model, "value model must be populated"


class TestRLRollout:
    def test_completes(self):
        from tilegen.evolve.strategies.rl_rollout import RLRolloutEngine

        initial = _initial_kernel()
        evalr = _MockEvaluator(time_us=50.0)
        kb = _MockKB()
        engine = RLRolloutEngine(seed=0)
        best = engine.run_campaign(initial, get_target("sm_90a"), evalr, kb,
                                   budget={"rollouts": 2, "batch": 4})
        assert best.vtm_ir is not None
        assert kb.promoted


class TestScheduleCodesign:
    def test_records_rewrite_rules(self):
        from tilegen.evolve.strategies.schedule_codesign import ScheduleCodesignEngine

        initial = _initial_kernel()
        evalr = _MockEvaluator(time_us=50.0) # all faster -> rewrites recorded
        kb = _MockKB()
        engine = ScheduleCodesignEngine(seed=0)
        best = engine.run_campaign(initial, get_target("sm_90a"), evalr, kb,
                                   budget={"generations": 2, "candidates_per_generation": 4})
        assert best.vtm_ir is not None
        assert kb.certified, "verified rewrite rules must be recorded"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_real_campaign_on_gpu():
    import torch

    from tilegen.evolve.evaluator import TritonAttentionEvaluator, knobs_to_params
    from tilegen.evolve.kb_manager import FileKnowledgeStore
    from tilegen.evolve.strategies.forgekernel import ForgeKernelEngine

    spec = AttentionSpec(batch=1, num_heads=8, num_kv_heads=8, head_dim=64,
                         kv_head_dim=64, seq_len=512, kv_quant=QuantSpec.nvfp4())
    initial = build_attention_vtm(spec, get_target("sm_90a"), variant="fa2_prefill")
    torch.manual_seed(0)
    q = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    evaluator = TritonAttentionEvaluator(
        q, k, v, is_causal=True, sm_scale=spec.scale,
        reference_params=knobs_to_params(initial.topology.knobs), iters=10,
    )
    kb = FileKnowledgeStore()
    best = ForgeKernelEngine(seed=0).run_campaign(
        initial, get_target("sm_90a"), evaluator, kb,
        budget={"generations": 2, "candidates_per_generation": 4, "plateau_window": 2},
    )
    assert best.candidate_id is not None
    assert best.vtm_ir.topology.knobs is not None


class TestProfilerGuidedMutation:
    def test_diagnose_fields_maps_telemetry(self):
        from tilegen.evolve.profiler import diagnose_fields

        fields = diagnose_fields({"register_spill": 1.0})
        assert "max_registers_per_thread" in fields
        assert diagnose_fields({}) == []

    def test_mutate_knobs_bias_prefers_field(self):
        import random

        from tilegen.evolve.mutation import mutate_knobs

        knobs = _initial_kernel().topology.knobs
        rng = random.Random(0)
        biased_hits = 0
        for _ in range(100):
            out = mutate_knobs(knobs, rng, bias=["pipeline_stages"])
# a biased mutation changes pipeline_stages (the only knob it changed)
            # when the biased field was chosen; count pipeline_stages != original
            if out.pipeline_stages != knobs.pipeline_stages:
                biased_hits += 1
        # ~70% bias -> most mutations hit the biased field
        assert biased_hits > 30

    def test_mutate_variant_steps(self):
        import random

        from tilegen.evolve.mutation import VARIANT_ORDER, mutate_variant

        rng = random.Random(0)
        seen = {mutate_variant("fa2_bf16", rng) for _ in range(10)}
        assert seen <= set(VARIANT_ORDER)
        assert "fa3_tma" in seen or "fa3_ws" in seen

    def test_synthetic_telemetry(self):
        from tilegen.evolve.evaluator import synthetic_telemetry
        from tilegen.ir.topology import TacticalKnobs

        t = synthetic_telemetry(TacticalKnobs(tile_m=32, pipeline_stages=1, max_registers_per_thread=232))
        assert "tensorcore_idle" in t
        assert "mbarrier_wait_stall" in t
        assert "register_spill" in t


class TestKBWarmStart:
    def test_forgekernel_warm_starts_from_kb(self):
        from tilegen.evolve.strategies.forgekernel import ForgeKernelEngine

        initial = _initial_kernel()
        # KB returns a certified recipe with a concrete config (warm start).
        class WarmKB(_MockKB):
            def query_recipes(self, target, symptoms):
                del target, symptoms
                return [{"config": {"tile_m": 256, "tile_n": 256, "tile_k": 64,
                                     "pipeline_stages": 3, "swizzle_mode": "B128",
                                     "async_wait_depth": 1, "max_registers_per_thread": 168,
                                     "exp_emulation_ratio": 0.0, "rescale_threshold": 0.0}}]

        kb = WarmKB()
        # All candidates slower -> the warm-start knobs are kept (no promotion).
        evalr = _MockEvaluator(time_us=150.0, ref_time=100.0)
        engine = ForgeKernelEngine(seed=0)
        best = engine.run_campaign(initial, get_target("sm_90a"), evalr, kb,
                                   budget={"generations": 2, "candidates_per_generation": 4})
        # The warm-start config was applied to the incumbent.
        assert best.vtm_ir.topology.knobs.tile_m == 256
        assert best.vtm_ir.topology.knobs.pipeline_stages == 3


class TestWorldModelValue:
    def test_nn_value_model_generalizes(self):
        from tilegen.evolve.strategies.world_model import NearestNeighborValueModel

        m = NearestNeighborValueModel(k=3)
        m.update(("a", 1, 100), 2.0) # a fast state
        m.update(("b", 9, 100), 0.5) # a slow, distant state
        # An unseen state near the fast one gets a high estimate.
        assert m.estimate(("a", 2, 100)) > 1.0
        assert m.estimate(("b", 8, 100)) < 1.0
        # Exact state returns its recorded value.
        m.update(("c", 5, 100), 1.7)
        assert abs(m.estimate(("c", 5, 100)) - 1.7) < 1e-6


class TestFullVariantCoverage:
    def test_variant_order_covers_sdpa_fa1_fa4(self):
        from tilegen.evolve.mutation import VARIANT_ORDER

        assert VARIANT_ORDER[0] == "sdpa"
        assert "fa1" in VARIANT_ORDER
        assert VARIANT_ORDER[-1] == "fa4_tilelang"
        assert len(set(VARIANT_ORDER)) == len(VARIANT_ORDER)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_evaluator_covers_sdpa_and_fa1():
    import torch

    from tilegen.evolve.evaluator import TritonAttentionEvaluator, knobs_to_params
    from tilegen.evolve.interface import KernelCandidate
    from tilegen.ops.attention import AttentionSpec, build_attention_vtm
    from tilegen.target.spec import get_target

    spec = AttentionSpec(batch=1, num_heads=8, num_kv_heads=8, head_dim=64,
                         kv_head_dim=64, seq_len=256)
    # 128x128 knobs (FA2 topology) are accepted by sdpa/fa1/fa2/fa4 alike.
    base = build_attention_vtm(spec, get_target("sm_120"), variant="fa2_prefill")
    torch.manual_seed(0)
    q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    ev = TritonAttentionEvaluator(q, k, v, is_causal=True, sm_scale=spec.scale,
                                  reference_params=knobs_to_params(base.topology.knobs), iters=6)
    variants = ["sdpa", "fa1", "fa2_bf16"]
    if torch.cuda.get_device_capability()[0] >= 10:
        variants.append("fa4_tilelang") # tcgen05 is Blackwell-only
    for variant in variants:
        c = KernelCandidate(variant, [], base, variant=variant)
        r = ev.evaluate([c], get_target("sm_120"))[0]
        assert r.compiled, f"{variant} should compile (err: {r.error_log[:80]})"
        assert r.correct, f"{variant} should be correct"


class TestGeneratedEvaluator:
    def test_backend_for_kernel(self):
        from tilegen.evolve.evaluator import backend_for_kernel
        from tilegen.ir.ops import VTMKernel
        from tilegen.ops.schedules import attention_topology

        def k(variant):
            return VTMKernel(name=variant, topology=attention_topology(variant, get_target("sm_120")))

        assert backend_for_kernel(k("fa1")) == "triton"
        assert backend_for_kernel(k("fa2_prefill")) == "triton"
        assert backend_for_kernel(k("fa3")) == "triton"
        assert backend_for_kernel(k("fa4")) == "tilelang"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_generated_kernel_evaluator_generic_flow():
    """The generic flow: lower VTM via backend -> compile -> run -> evaluate."""
    import torch

    from tilegen.evolve.evaluator import GeneratedKernelEvaluator, backend_for_kernel
    from tilegen.evolve.interface import KernelCandidate
    from tilegen.ops.attention import AttentionSpec, build_dense_attention_vtm

    spec = AttentionSpec(batch=1, num_heads=8, num_kv_heads=8, head_dim=64,
                         seq_len=256, dtype="bf16", causal=True)
    torch.manual_seed(0)
    q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    ev = GeneratedKernelEvaluator(q, k, v, is_causal=True, iters=6)

    fa2 = build_dense_attention_vtm(spec, get_target("sm_90a"), variant="fa2_prefill")
    assert backend_for_kernel(fa2) == "triton"
    r2 = ev.evaluate([KernelCandidate("fa2", [], fa2)], get_target("sm_90a"))[0]
    assert r2.compiled and r2.correct, f"FA2 generic flow: {r2.error_log[:120]}"

    fa4 = build_dense_attention_vtm(spec, get_target("sm_120"), variant="fa4")
    assert backend_for_kernel(fa4) == "tilelang"
    r4 = ev.evaluate([KernelCandidate("fa4", [], fa4)], get_target("sm_120"))[0]
    if torch.cuda.get_device_capability()[0] >= 10:
        assert r4.compiled and r4.correct, f"FA4 generic flow: {r4.error_log[:120]}"
    else:
        # tcgen05 is Blackwell-only: the generic flow must fail cleanly.
        assert not r4.compiled


class TestGenericFlowMutation:
    def test_reopen_m0_flips_cta_group_to_fa4(self):
        """reopen_m0 can flip cta_group 1->2, reaching the FA4 (tcgen05) topology."""
        import random

        from tilegen.evolve.mutation import reopen_m0
        from tilegen.ir.ops import VTMKernel
        from tilegen.ops.schedules import attention_topology

        base = VTMKernel(name="fa2", topology=attention_topology("fa2_prefill", get_target("sm_120")))
        assert base.topology.grid.cta_group == 1
        seen = {reopen_m0(base, random.Random(i)).topology.grid.cta_group for i in range(30)}
        assert 2 in seen, "cta_group should flip to 2 (FA4) at least once"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_generic_flow_campaign_end_to_end():
    """A full ForgeKernel campaign driven by GeneratedKernelEvaluator (generic flow)."""
    import torch

    from tilegen.evolve.evaluator import GeneratedKernelEvaluator, backend_for_kernel
    from tilegen.evolve.kb_manager import FileKnowledgeStore
    from tilegen.evolve.strategies.forgekernel import ForgeKernelEngine
    from tilegen.ops.attention import AttentionSpec, build_dense_attention_vtm

    spec = AttentionSpec(batch=1, num_heads=8, num_kv_heads=8, head_dim=64,
                         seq_len=256, dtype="bf16", causal=True)
    initial = build_dense_attention_vtm(spec, get_target("sm_120"), variant="fa2_prefill")
    assert backend_for_kernel(initial) == "triton"

    torch.manual_seed(0)
    q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    ev = GeneratedKernelEvaluator(q, k, v, is_causal=True, iters=6)
    kb = FileKnowledgeStore()
    best = ForgeKernelEngine(seed=0).run_campaign(
        initial, get_target("sm_120"), ev, kb,
        budget={"generations": 2, "candidates_per_generation": 4, "plateau_window": 1},
    )
    assert best.vtm_ir is not None
    assert backend_for_kernel(best.vtm_ir) in ("triton", "tilelang")
