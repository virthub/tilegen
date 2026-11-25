# tests/test_attention.py
#
# Unit tests for attention specs, VTM builders, and the canonical dispatch
# facade

from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec, build_paged_attention_vtm
from tilegen.target.spec import get_target


class TestBuildPagedAttention:
    def test_blackwell_2cta(self):
        spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
        k = build_paged_attention_vtm(spec, get_target("sm_100a"))
        assert k.name == "paged_flash_attn_fp4"
        assert k.topology.grid.mode.value == "persistent_sm"
        assert k.topology.grid.cta_group == 2
        assert "kv_paged" in k.layouts
        assert "kv_quant" in k.quants

    def test_hopper_no_2cta(self):
        spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
        k = build_paged_attention_vtm(spec, get_target("sm_90a"))
        assert k.topology.grid.cta_group == 1

    def test_serialize_contains_pipeline(self):
        spec = AttentionSpec(kv_quant=QuantSpec.int4_kv())
        text = build_paged_attention_vtm(spec, get_target("sm_90a")).serialize()
        assert "vtm.pipeline_for" in text
        assert "vtm.dequant" in text

    def test_scale_default(self):
        spec = AttentionSpec(head_dim=128)
        assert abs(spec.scale - (1.0 / (128 ** 0.5))) < 1e-12


class TestBuildDecodeVTM:
    def test_decode_kernel_structure(self):
        from tilegen.ops.attention import build_decode_vtm

        spec = AttentionSpec(
            batch=4, num_heads=8, num_kv_heads=2, head_dim=64,
            seq_len=128, num_kv_blocks=32, page_size=16, dtype="bf16",
        )
        k = build_decode_vtm(spec, get_target("sm_90a"))
        assert k.name == "paged_decode"
        assert k.topology.grid.mode.value == "data_parallel"
        # Two GatherOps (K and V) + load + 2 matmul + softmax + store.
        ops = [op.op for op in k.body]
        assert ops.count("gather") == 2
        assert "kv_paged" in k.layouts
        # GatherOp carries the paged-indirection operands.
        gather = next(op for op in k.body if op.op == "gather")
        assert gather.block_table == "block_tables"
        assert gather.seq_len == "seq_lens"
        assert gather.layout == "kv_paged"

    def test_decode_serializes_gather(self):
        from tilegen.ops.attention import build_decode_vtm

        spec = AttentionSpec(batch=2, num_heads=4, num_kv_heads=1, head_dim=64,
                             seq_len=64, num_kv_blocks=16, page_size=16, dtype="bf16")
        text = build_decode_vtm(spec, get_target("sm_90a")).serialize()
        assert "vtm.gather" in text
        assert "%seq_lens" in text and "%block_tables" in text

    def test_gather_undefined_layout_rejected(self):
        from tilegen.ir.builder import VTMBuilder
        from tilegen.ir.types import MemoryTier

        b = VTMBuilder("bad_gather")
        b.input("K_cache", (8, 16, 2, 64), "bf16")
        b.input("block_tables", (4, 8), "int32")
        b.input("seq_lens", (4,), "int32")
        b.output("O", (4, 8, 64), "bf16")
        b.alloc("k_tile", (128, 64), "bf16", MemoryTier.SCRATCHPAD)
        b.gather("K_cache", "k_tile", "block_tables", "seq_lens", "missing_layout")
        import pytest

        with pytest.raises(ValueError):
            b.build()
