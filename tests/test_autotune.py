# tests/test_autotune.py
#
# Tests for the constraint-pruned autotuner (pruning is GPU-free)

from tilegen.autotune import (
    SearchSpace,
    TuneConfig,
    estimate_registers,
    estimate_smem_bytes,
    prune_configs,
)
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec
from tilegen.target.spec import get_target


class TestSearchSpace:
    def test_expand(self):
        space = SearchSpace(
            tile_m=[64], tile_n=[64], tile_k=[32, 64],
            pipeline_stages=[1], swizzle_mode=["NONE"], num_warps=[4],
        )
        cfgs = space.expand()
        assert len(cfgs) == 2
        assert all(isinstance(c, TuneConfig) for c in cfgs)

    def test_from_dict(self):
        space = SearchSpace.from_dict({"tile_m": [128], "tile_n": [128]})
        assert space.tile_m == [128]
        assert space.pipeline_stages == (1, 2, 3, 4) # default preserved


class TestEstimates:
    def test_smem_monotonic_in_stages(self):
        c1 = TuneConfig(128, 128, 64, pipeline_stages=1)
        c4 = TuneConfig(128, 128, 64, pipeline_stages=4)
        assert estimate_smem_bytes(c4, 64) > estimate_smem_bytes(c1, 64)

    def test_registers(self):
        assert estimate_registers(TuneConfig(128, 128, 64, num_warps=4), 64) >= 64


class TestPruning:
    def test_quant_alignment(self):
        # NVFP4: block 16, pack 2 -> tile_k must be %16==0 and %2==0.
        quant = QuantSpec.nvfp4()
        spec = AttentionSpec(head_dim=64, kv_quant=quant)
        cfgs = [
            TuneConfig(128, 128, 64), # aligned -> kept
            TuneConfig(128, 128, 40), # 40 % 16 != 0 -> pruned
            TuneConfig(128, 127, 64), # tile_n 127 % 2 != 0 -> pruned
        ]
        kept = prune_configs(cfgs, spec, get_target("sm_90a"), quant=quant)
        kept_k = {c.tile_k for c in kept}
        kept_n = {c.tile_n for c in kept}
        assert 64 in kept_k
        assert 40 not in kept_k
        assert 127 not in kept_n

    def test_smem_budget(self):
        quant = QuantSpec.int8_kv() # block 64
        spec = AttentionSpec(head_dim=64, kv_quant=quant)
        target = get_target("sm_90a")
        # A huge config should exceed the smem budget and be pruned.
        huge = TuneConfig(256, 256, 256, pipeline_stages=4)
        small = TuneConfig(64, 64, 64, pipeline_stages=1)
        kept = prune_configs([huge, small], spec, target, quant=quant)
        assert small in kept
