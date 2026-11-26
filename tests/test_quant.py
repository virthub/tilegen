# tests/test_quant.py
#
# Unit tests for QuantSpec packing, granularities, and presets

import pytest
from tilegen.layout.quant import QuantSpec


class TestBitEncoding:
    def test_sub_byte(self):
        assert QuantSpec.int4_kv().is_sub_byte
        assert QuantSpec.nvfp4().is_sub_byte
        assert not QuantSpec.int8_kv().is_sub_byte

    def test_bits(self):
        assert QuantSpec(dtype="int8").bits_per_element == 8
        assert QuantSpec(dtype="fp8_e4m3").bits_per_element == 8
        assert QuantSpec(dtype="fp4_e2m1").bits_per_element == 4
        assert QuantSpec(dtype="bf16").bits_per_element == 16


class TestPacking:
    def test_packed_bytes(self):
        q = QuantSpec.int4_kv()
        assert q.packed_bytes(0) == 0
        assert q.packed_bytes(1) == 1
        assert q.packed_bytes(2) == 1
        assert q.packed_bytes(3) == 2
        assert q.packed_bytes(128) == 64

    def test_scale_blocks(self):
        q = QuantSpec.int8_kv() # block_size 64
        assert q.num_scale_blocks(64) == 1
        assert q.num_scale_blocks(65) == 2
        assert q.num_scale_blocks(192) == 3


class TestPresets:
    def test_nvfp4(self):
        q = QuantSpec.nvfp4()
        assert (q.dtype, q.granularity, q.block_size) == ("fp4_e2m1", "block", 16)
        assert q.scale_dtype == "fp8_e4m3" and q.tensor_scale_dtype == "float32"
        assert q.pack_factor == 2

    def test_mxfp(self):
        assert QuantSpec.mxfp4().block_size == 32
        assert QuantSpec.mxfp8().block_size == 32
        assert QuantSpec.mxfp8().dtype == "fp8_e4m3"

    def test_awq_gptq(self):
        q = QuantSpec.awq_gptq_int4()
        assert q.block_size == 128 and q.has_zero_point
        assert q.interleave == "awq_k8"


class TestValidation:
    def test_rejects_unknown_dtype(self):
        with pytest.raises(ValueError):
            QuantSpec(dtype="int3")

    def test_rejects_block_without_size(self):
        with pytest.raises(ValueError):
            QuantSpec(dtype="int8", granularity="block", block_size=None)

    def test_rejects_bad_pack_factor(self):
        with pytest.raises(ValueError):
            QuantSpec(dtype="int4", pack_factor=1)

    def test_roundtrip_dict(self):
        q = QuantSpec.nvfp4()
        assert QuantSpec.from_dict(q.to_dict()) == q
