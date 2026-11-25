# tests/test_ir.py
#
# Unit tests for the VTM IR builder, verifier, types, topology, and ops

import pytest
from tilegen.ir.builder import VTMBuilder, verify_kernel
from tilegen.ir.topology import (
    ExecutionRoles,
    GridMode,
    GridTopology,
    Materialize,
    PhaseMode,
    RasterOrder,
    TacticalKnobs,
    TopologySpec,
)
from tilegen.ir.types import MemoryTier
from tilegen.layout.quant import QuantSpec


def _simple_builder():
    b = VTMBuilder("test_kernel")
    b.input("A", (128, 64), "bf16")
    b.output("C", (128, 128), "bf16")
    b.quant("q", QuantSpec.int8_kv())
    b.alloc("a_s", (128, 64), "bf16", MemoryTier.SCRATCHPAD)
    b.alloc("acc", (128, 128), "fp32", MemoryTier.ACCUMULATOR)
    b.load("A", "a_s")
    b.matmul("a_s", "a_s", "acc", transpose_b=True, clear=True)
    b.store("acc", "C")
    return b


class TestTopology:
    def test_defaults(self):
        t = TopologySpec()
        assert t.grid.mode == GridMode.DATA_PARALLEL
        assert t.roles.phase_mode == PhaseMode.SERIAL
        assert t.knobs.swizzle_mode == "NONE"

    def test_invalid_cta_group(self):
        with pytest.raises(ValueError):
            GridTopology(cta_group=3)

    def test_invalid_swizzle(self):
        with pytest.raises(ValueError):
            TacticalKnobs(swizzle_mode="B256")

    def test_to_dict_roundtrip_fields(self):
        t = TopologySpec(
            grid=GridTopology(GridMode.PERSISTENT_SM, RasterOrder.SWIZZLED_L2),
            roles=ExecutionRoles(1, 2, PhaseMode.PING_PONG),
        )
        d = t.to_dict()
        assert d["grid"]["mode"] == "persistent_sm"
        assert d["roles"]["phase_mode"] == "ping_pong"


class TestBuilder:
    def test_build_and_serialize(self):
        k = _simple_builder().build()
        assert k.name == "test_kernel"
        text = k.serialize()
        assert text.startswith("vtm.kernel @test_kernel(")
        assert "vtm.matmul" in text
        assert "vtm.store" in text

    def test_undefined_buffer_rejected(self):
        b = _simple_builder()
        b.load("MISSING", "a_s")
        with pytest.raises(ValueError):
            b.build()

    def test_undefined_quant_rejected(self):
        b = VTMBuilder("bad")
        b.input("A", (128, 64), "uint8")
        b.output("C", (128, 64), "bf16")
        b.alloc("a_s", (128, 64), "uint8", MemoryTier.SCRATCHPAD)
        b.alloc("r", (128, 64), "bf16", MemoryTier.COMPUTE_INPUT)
        b.load("A", "a_s")
        b.dequant("a_s", "r", "missing_quant")
        with pytest.raises(ValueError):
            b.build()

    def test_quant_alignment(self):
        # tile_k=64 must be divisible by block_size=48? no; use 48 -> reject.
        b = _simple_builder()
        b.topology = TopologySpec(
            knobs=TacticalKnobs(tile_k=48),
        )
        # quant q is int8_kv block_size 64 -> 48 % 64 != 0
        with pytest.raises(ValueError):
            b.build()


class TestRecomputeBoundary:
    def test_override(self):
        from tilegen.ir.topology import RecomputeBoundary

        r = RecomputeBoundary(default=Materialize.REGISTER_FUSED)
        assert r.for_tensor("S") == Materialize.REGISTER_FUSED
        r = RecomputeBoundary(
            default=Materialize.REGISTER_FUSED,
            overrides={"S": Materialize.SCRATCHPAD_CACHED},
        )
        assert r.for_tensor("S") == Materialize.SCRATCHPAD_CACHED
