# tests/test_schedules.py
#
# Unit tests for FA1-FA4 attention schedule topologies and variant
# resolution

import pytest
from tilegen.ir.topology import GridMode, PhaseMode
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec, build_attention_vtm
from tilegen.ops.schedules import VARIANTS, attention_topology, resolve_variant
from tilegen.target.spec import get_target


class TestResolveVariant:
    def test_auto_by_capability(self):
        assert resolve_variant(get_target("sm_100a")) == "fa4"
        assert resolve_variant(get_target("sm_90a")) == "fa3"
        assert resolve_variant(get_target("sm_80")) == "fa2_prefill"
        assert resolve_variant(get_target("ascend_910c")) == "fa1"

    def test_explicit(self):
        assert resolve_variant(get_target("sm_90a"), "fa1") == "fa1"

    def test_unknown(self):
        with pytest.raises(ValueError):
            resolve_variant(get_target("sm_90a"), "fa9")


class TestVariantTopologies:
    def test_fa1_baseline(self):
        t = attention_topology("fa1", get_target("sm_80"))
        assert t.grid.mode == GridMode.DATA_PARALLEL
        assert t.roles.phase_mode == PhaseMode.SERIAL
        assert t.knobs.pipeline_stages == 1

    def test_fa2_decode_split_kv(self):
        t = attention_topology("fa2_decode", get_target("sm_80"))
        assert t.grid.mode == GridMode.SPLIT_KV
        assert t.grid.split == 4

    def test_fa3_ping_pong(self):
        t = attention_topology("fa3", get_target("sm_90a"))
        assert t.grid.mode == GridMode.PERSISTENT_SM
        assert t.roles.phase_mode == PhaseMode.PING_PONG
        assert t.roles.consumer_warpgroups == 2

    def test_fa4_2cta_and_knobs(self):
        t = attention_topology("fa4", get_target("sm_100a"))
        assert t.grid.cta_group == 2
        assert t.knobs.exp_emulation_ratio > 0.0
        assert t.knobs.rescale_threshold > 0.0

    def test_fa4_clamped_without_2cta(self):
        t = attention_topology("fa4", get_target("sm_90a"))
        assert t.grid.cta_group == 1


class TestBuildVariants:
    def test_variants_produce_distinct_kernels(self):
        spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
        kernels = {
            v: build_attention_vtm(spec, get_target("sm_90a"), variant=v) for v in VARIANTS
        }
        topologies = {v: k.topology.to_dict() for v, k in kernels.items()}
        # Every variant must lock in a distinct topology.
        assert len(set(str(t) for t in topologies.values())) == len(VARIANTS)
