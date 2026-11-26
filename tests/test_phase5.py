# tests/test_phase5.py
#
# Tests for Phase-5 loop hygiene: epoch-scoped attempt history, the
# budget-modulated K_max revision cap, the diagnostic panel, and the
# attempt-history mutation exclusion.

import json
import random

from tilegen.evolve.attempt_history import AttemptHistory
from tilegen.evolve.interface import EvalResult
from tilegen.evolve.kb_manager import FileKnowledgeStore
from tilegen.evolve.metrics import CampaignMetrics
from tilegen.evolve.mutation import mutate_knobs
from tilegen.evolve.strategies.forgekernel import ForgeKernelEngine
from tilegen.evolve.triggers import RevisionGuard
from tilegen.ir.topology import TacticalKnobs
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec, build_attention_vtm
from tilegen.target.spec import get_target


def _kernel():
    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    return build_attention_vtm(spec, get_target("sm_90a"), variant="fa2_prefill")


class _SlowEvaluator:
    def profile_reference(self, sealed_ref_id, target):
        del sealed_ref_id, target
        return {"device_time_us": 100.0}

    def evaluate(self, candidates, target, sealed_ref_id=None):
        del target, sealed_ref_id
        return [EvalResult(c.candidate_id, True, True, device_time_us=150.0,
                           speedup_vs_ref=0.667) for c in candidates]


class TestAttemptHistory:
    def test_cap_and_failed_fields(self):
        h = AttemptHistory(cap=4)
        h.record(["tile_m"], "stall")
        h.record(["tile_n", "tile_k"], "stall")
        h.record(["pipeline_stages"] * 5, "stall")
        assert len(h) == 4 # capped, oldest dropped first
        assert h.failed_fields() == {"pipeline_stages"}

    def test_epoch_advance_clears(self):
        h = AttemptHistory()
        h.record(["tile_m"], "stall")
        h.advance_epoch()
        assert h.epoch == 1
        assert h.failed_fields() == set()
        # Entries recorded for a stale epoch are ignored.
        h.record(["tile_n"], "stall", epoch=0)
        assert h.failed_fields() == set()


class TestRevisionGuard:
    def test_cap_and_reset(self):
        g = RevisionGuard(k_max_base=2, k_max_slack=0, k_global_cap=8)
        assert g.allow(0.0)
        g.record_revision()
        assert g.allow(0.0)
        g.record_revision()
        assert not g.allow(0.0) # K_max_effective = 2 reached
        g.on_promotion()
        assert g.allow(0.0)

    def test_budget_modulation(self):
        g = RevisionGuard(k_max_base=2, k_max_slack=2)
        g.record_revision()
        g.record_revision()
        g.record_revision() # K = 3 < K_max_effective = 4 with no budget spent
        assert g.allow(0.0)
        # With the budget consumed, the effective cap drops to the base.
        g2 = RevisionGuard(k_max_base=2, k_max_slack=2)
        g2.record_revision()
        g2.record_revision()
        assert not g2.allow(1.0) # 2 + 0*2 = 2, K = 2 -> blocked

    def test_global_cap(self):
        g = RevisionGuard(k_max_base=9, k_max_slack=0, k_global_cap=3)
        for _ in range(3):
            assert g.allow(0.0)
            g.record_revision()
        assert not g.allow(0.0)


class TestMetricsPanel:
    def test_panel_and_record(self, tmp_path):
        m = CampaignMetrics(submissions=10, effective_promotions=2,
                            rollbacks=8, m0_revisions=2, noise_rejections=1)
        assert m.promotion_rate == 0.2
        assert m.rollback_rate == 0.8
        assert m.m0_yield == 1.0
        m.record(tmp_path)
        entries = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert entries[-1]["promotion_rate"] == 0.2


class TestMutationExclude:
    def test_exclude_never_touches_field(self):
        knobs = TacticalKnobs()
        rng = random.Random(0)
        for _ in range(50):
            out = mutate_knobs(knobs, rng, exclude={"tile_m"})
            assert out.tile_m == knobs.tile_m


class TestEngineIntegration:
    def test_guard_blocks_after_kmax(self, tmp_path):
        kb = FileKnowledgeStore(tmp_path)
        engine = ForgeKernelEngine(seed=0)
        engine.run_campaign(
            _kernel(), get_target("sm_90a"), _SlowEvaluator(), kb,
            budget={"generations": 3, "candidates_per_generation": 2,
                    "plateau_window": 1, "budget_consumed_fraction": 1.0},
        )
        entries = json.loads(
            (tmp_path / "metrics.json").read_text(encoding="utf-8"))
        last = entries[-1]
        assert last["m0_revisions"] == 2 # K_max_effective = 2
        assert last["m0_revisions_blocked"] == 1
        assert last["rollbacks"] == 3
        assert last["submissions"] == 6
