# tests/test_noise.py
#
# Tests for the Phase-1 measurement layer: robust statistics, the noise-floor
# protocol (§5 of tilegen-docs/meta-evolver.md), the LCB acceptance bound, and
# the unified bootstrap gate inside the reference-guided evolvers.

import random

import pytest

from tilegen.evolve.interface import EvalResult
from tilegen.evolve.kb_manager import FileKnowledgeStore
from tilegen.evolve.noise import (
    DEFAULT_MIN_FLOOR,
    estimate_noise_floor_from_runs,
    gate_value,
    lc_bound,
    robust_stats,
)
from tilegen.evolve.strategies.forgekernel import ForgeKernelEngine
from tilegen.evolve.zero import ZeroEngine
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec, build_attention_vtm
from tilegen.target.spec import get_target


def _initial_kernel():
    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    return build_attention_vtm(spec, get_target("sm_90a"), variant="fa2_prefill")


class _JitterEvaluator:
    """Deterministic mock: flat or jittered timing samples, speedup configurable.

    ``sample_times`` returns reference-time samples for the reference params
    and candidate-time samples otherwise, so the noise layer sees both sides
    of the comparison correctly.
    """

    def __init__(self, speedup=1.1, jitter=0.0, ref_time=100.0, ref_params=None):
        self.speedup = speedup
        self.jitter = jitter
        self.ref_time = ref_time
        self.ref_params = ref_params
        self._ref_sampled = False

    def profile_reference(self, sealed_ref_id, target):
        del sealed_ref_id, target
        return {"device_time_us": self.ref_time}

    def evaluate(self, candidates, target, sealed_ref_id=None):
        del target, sealed_ref_id
        out = []
        for c in candidates:
            t = self.ref_time / self.speedup
            out.append(EvalResult(
                c.candidate_id, True, True, device_time_us=t,
                speedup_vs_ref=self.speedup, max_rel_error=0.0,
            ))
        return out

    def sample_times(self, params, variant="fa2_bf16", warmup=2, iters=8):
        del variant, warmup
        rng = random.Random(42)
# Convention: the engine samples the reference first; later calls are
        # candidate confirmation runs.
        is_ref = not self._ref_sampled
        self._ref_sampled = True
        base = self.ref_time if is_ref else self.ref_time / self.speedup
        return [base + self.jitter * (i - iters / 2) for i in range(iters)]


class TestRobustStats:
    def test_median_mad_sigma(self):
        m, mad, sigma = robust_stats([1.0, 1.0, 1.1, 1.2, 5.0])
        assert m == 1.1
        assert mad == pytest.approx(0.1)
        assert sigma == pytest.approx(1.4826 * 0.1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            robust_stats([])


class TestNoiseFloor:
    def test_warmup_exclusion_and_min_floor(self):
# First two runs are warm-up artifacts (huge); they must not pollute
        # the estimate.
        runs = [1000.0, 1000.0] + [100.0] * 15
        rec = estimate_noise_floor_from_runs(runs, W=2, k=2.0)
        assert rec.m_r == 100.0
        assert rec.sigma_r == 0.0
        assert rec.floor == DEFAULT_MIN_FLOOR # max(1.5%, k*sigma/m)

    def test_jittered_floor(self):
        runs = [100.0 + 5.0 * (i - 8) for i in range(17)]
        rec = estimate_noise_floor_from_runs(runs, W=2, k=2.0)
        assert rec.floor == pytest.approx(2.0 * rec.sigma_r / rec.m_r)

    def test_gate_value(self):
        assert gate_value(0.015, 1.05) == 1.05 # bootstrap wins
        assert gate_value(0.10, 1.0) == pytest.approx(1.10)
        assert gate_value(0.015, 1.0) == pytest.approx(1.015)


class TestLcBound:
    def test_direction_correct_latency(self):
        # Reference clearly faster than candidate -> bound below 1.
        assert lc_bound(12.0, 0.1, 10.0, 0.1) < 1.0
        # Candidate clearly faster than reference -> bound above 1.
        assert lc_bound(10.0, 0.1, 12.0, 0.1) > 1.0

    def test_denominator_guard(self):
        assert lc_bound(0.0, 0.1, 10.0, 0.1) is None


def _ref_params():
    from tilegen.evolve.evaluator import knobs_to_params

    return knobs_to_params(_initial_kernel().topology.knobs)


class TestUnifiedGateInStrategies:
    def test_flat_noise_promotes(self, tmp_path):
        kb = FileKnowledgeStore(tmp_path)
        engine = ForgeKernelEngine(seed=0)
        cand = engine.run_campaign(
            _initial_kernel(), get_target("sm_90a"),
            _JitterEvaluator(speedup=1.1, jitter=0.0, ref_params=_ref_params()), kb,
            budget={"generations": 1, "candidates_per_generation": 2},
        )
        # Flat samples -> floor = 1.5%; 1.1 >= max(1.015, 1.05) -> accepted.
        assert not cand.candidate_id.startswith("generation-0")
        refs = list(tmp_path.glob("reference_*.json"))
        assert refs and refs[0].read_text(encoding="utf-8")

    def test_noisy_reference_stays_silent(self, tmp_path):
# Large jitter -> floor >> speedup-1 -> the gate rejects everything,
        # and the campaign returns the sealed reference untouched.
        kb = FileKnowledgeStore(tmp_path)
        engine = ZeroEngine(seed=0)
        cand = engine.run_campaign(
            _initial_kernel(), get_target("sm_90a"),
            _JitterEvaluator(speedup=1.1, jitter=60.0, ref_params=_ref_params()), kb,
            budget={"generations": 1, "candidates_per_generation": 2},
        )
        assert cand.candidate_id in ("generation-0", "generation-0-warm")
        assert not list(tmp_path.glob("reference_*.json"))

    def test_confirmation_fields_recorded(self, tmp_path):
        kb = FileKnowledgeStore(tmp_path)
        engine = ForgeKernelEngine(seed=0)
        engine.run_campaign(
            _initial_kernel(), get_target("sm_90a"),
            _JitterEvaluator(speedup=1.1, jitter=0.0, ref_params=_ref_params()), kb,
            budget={"generations": 1, "candidates_per_generation": 2},
        )
        import json

        ref = json.loads(list(tmp_path.glob("reference_*.json"))[0].read_text(encoding="utf-8"))
        assert "confirmation" in ref
        assert ref["confirmation"]["m_c"] is not None
        assert ref["confirmation"]["lcb_speedup"] is not None
