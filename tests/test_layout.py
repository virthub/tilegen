# tests/test_layout.py
#
# Unit tests for layout descriptors (ContiguousLayout, PagedKVLayout,
# TensorSpec)

import pytest
from tilegen.layout.layout import ContiguousLayout, PagedKVLayout, TensorSpec
from tilegen.layout.quant import QuantSpec


class TestContiguous:
    def test_row_major_strides(self):
        c = ContiguousLayout((2, 3, 4))
        assert c.strides == (12, 4, 1)
        assert c.numel() == 24

    def test_offset(self):
        c = ContiguousLayout((2, 3, 4))
        assert c.offset((0, 0, 0)) == 0
        assert c.offset((1, 2, 3)) == 23

    def test_explicit_strides(self):
        c = ContiguousLayout((2, 3), strides=(1, 2))  # column-major-ish
        assert c.offset((1, 2)) == 1 + 2 * 2

    def test_rejects_symbolic_without_strides(self):
        from tilegen.ir.types import SymbolicDim

        with pytest.raises(ValueError):
            ContiguousLayout((SymbolicDim("N"), 4))


class TestPagedKV:
    def test_num_blocks(self):
        p = PagedKVLayout(block_size=16, num_kv_heads=8, head_dim=64)
        assert p.num_blocks_for(1) == 1
        assert p.num_blocks_for(16) == 1
        assert p.num_blocks_for(17) == 2

    def test_in_page_offset(self):
        p = PagedKVLayout(block_size=16, num_kv_heads=8, head_dim=64)
        base = 16 * 8 * 64
        assert p.in_page_offset(2, 0, 0) == 2 * base
        assert p.in_page_offset(0, 3, 1) == 3 * 8 * 64 + 1 * 64

    def test_range_guards(self):
        p = PagedKVLayout(block_size=16, num_kv_heads=8, head_dim=64)
        with pytest.raises(ValueError):
            p.in_page_offset(0, 16, 0)
        with pytest.raises(ValueError):
            p.in_page_offset(0, 0, 8)


class TestTensorSpec:
    def test_to_dict(self):
        spec = TensorSpec(
            dtype="uint8",
            shape=(256, 32, 8, 64),
            layout=PagedKVLayout(32, 8, 64),
            quant=QuantSpec.nvfp4(),
        )
        d = spec.to_dict()
        assert d["layout"]["kind"] == "paged_kv"
        assert d["quant"]["dtype"] == "fp4_e2m1"
