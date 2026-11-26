# tests/test_audit_pricing.py
#
# Tests for the isolation audit (code_override gate, profiler symbol
# stripping) and rule-based revision pricing.

import pytest

from tilegen.evolve.adapter import candidate_from_dict, propose_candidates
from tilegen.evolve.pricing import (
    apply_priced_revision,
    price_revision,
)
from tilegen.evolve.profiler import normalize_ncu_metrics
from tilegen.ir.topology import GridMode, PhaseMode
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec, build_attention_vtm
from tilegen.target.spec import get_target


def _kernel():
    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    return build_attention_vtm(spec, get_target("sm_90a"), variant="fa2_prefill")


class TestIsolationAudit:
    def test_adapter_rejects_code_override_by_default(self):
        item = {"vtm_ir": _kernel().to_dict(), "code_override": "malicious"}
        with pytest.raises(ValueError, match="code_override not allowed"):
            propose_candidates([item])
        # Explicitly allowed still round-trips.
        cands = propose_candidates([item], allow_code_override=True)
        assert cands[0].code_override == "malicious"

    def test_evaluator_gate(self):
        torch = pytest.importorskip("torch")
        from tilegen.evolve.evaluator import GeneratedKernelEvaluator

        q = torch.randn(1, 8, 128, 64, device="cpu", dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        ev = GeneratedKernelEvaluator(q, k, v) # gate default off
        from tilegen.evolve.interface import KernelCandidate

        cand = KernelCandidate("c", [], _kernel(),
                               code_override="def not_real(): pass")
        res = ev.evaluate([cand], get_target("sm_90a"))
        assert not res[0].compiled
        assert "code_override" in (res[0].error_log or "")

    def test_symbol_stripping_hashes_unknown_keys(self):
        out = normalize_ncu_metrics({
            "sm__warps_active.avg.pct_of_peak_sustained_active": 50.0,
            "my_kernel_function_file_metric": 1.0,
        })
        assert "tensorcore_idle" in out
        anon = [k for k in out if k.startswith("anon_")]
        assert len(anon) == 1 and len(anon[0]) == len("anon_") + 8
        assert "my_kernel_function_file_metric" not in out


class TestPricing:
    def test_orders_by_severity_and_drops_sub_noise(self):
        records = price_revision({
            "tensorcore_idle": 0.9,
            "cta_boundary_stall": 0.4,
            "register_spill": 0.001, # below the 1.5% floor: dropped
        })
        assert records[0].decision == "grid_mode"
        assert records[0].action == "PERSISTENT_SM"
        assert records[0].price == pytest.approx(1.0 * 0.4)
        assert [r.price for r in records] == pytest.approx([0.4, 0.9 * 0.4])

    def test_empty_telemetry(self):
        assert price_revision({}) == []

    def test_apply_top_priced_grid_mode(self):
        kernel = _kernel()
        from dataclasses import replace

        grid = replace(kernel.topology.grid, mode=GridMode.DATA_PARALLEL)
        kernel = replace(kernel, topology=replace(kernel.topology, grid=grid))
        from tilegen.evolve.pricing import PricingRecord

        out = apply_priced_revision(
            kernel, [PricingRecord("grid_mode", "PERSISTENT_SM", 1.0, "profile")])
        assert out.topology.grid.mode == GridMode.PERSISTENT_SM

    def test_apply_phase_mode(self):
        kernel = _kernel()
        from tilegen.evolve.pricing import PricingRecord

        out = apply_priced_revision(
            kernel, [PricingRecord("phase_mode", "PING_PONG", 0.8, "profile")])
        assert out.topology.roles.phase_mode == PhaseMode.PING_PONG

    def test_fallback_to_random_reopen(self):
        import random

        kernel = _kernel()
        before = kernel.topology.grid.mode
        out = apply_priced_revision(kernel, [], random.Random(0))
        assert out.topology.grid.mode != before
