# tests/test_guards.py
#
# Unit tests for host-side boundary guards and quantization alignment
# checks

import pytest
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec
from tilegen.ops.guards import (
    BoundaryError,
    check_quant_alignment,
    validate_attention,
)


class TestQuantAlignment:
    def test_block_aligned_ok(self):
        check_quant_alignment(64, 128, QuantSpec.nvfp4()) # block 16, pack 2

    def test_block_misaligned(self):
        with pytest.raises(BoundaryError):
            check_quant_alignment(40, 128, QuantSpec.nvfp4()) # 40 % 16 != 0

    def test_pack_misaligned(self):
        with pytest.raises(BoundaryError):
            check_quant_alignment(64, 127, QuantSpec.nvfp4()) # 127 % 2 != 0


class TestValidateAttention:
    def _spec(self):
        return AttentionSpec(kv_quant=QuantSpec.nvfp4())

    def test_valid(self):
        shape = (256, 32, 8, 64)
        validate_attention(
            spec=self._spec(), k_cache_shape=shape, v_cache_shape=shape,
            o_shape=(1, 8, 2048, 128), tile_k=64, tile_n=128,
        )

    def test_bad_rank(self):
        with pytest.raises(BoundaryError):
            validate_attention(
                spec=self._spec(), k_cache_shape=(256, 32, 8), v_cache_shape=(256, 32, 8, 64),
                o_shape=(1, 8, 2048, 128), tile_k=64, tile_n=128,
            )

    def test_bad_head_dim(self):
        with pytest.raises(BoundaryError):
            validate_attention(
                spec=self._spec(), k_cache_shape=(256, 32, 8, 48),
                v_cache_shape=(256, 32, 8, 48), o_shape=(1, 8, 2048, 128),
                tile_k=64, tile_n=128,
            )

    def test_mismatched_kv_shapes(self):
        with pytest.raises(BoundaryError):
            validate_attention(
                spec=self._spec(), k_cache_shape=(256, 32, 8, 64),
                v_cache_shape=(128, 32, 8, 64), o_shape=(1, 8, 2048, 128),
                tile_k=64, tile_n=128,
            )

    def test_out_alias(self):
        shape = (256, 32, 8, 64)
        with pytest.raises(BoundaryError):
            validate_attention(
                spec=self._spec(), k_cache_shape=shape, v_cache_shape=shape,
                o_shape=shape, tile_k=64, tile_n=128, out_aliases_input=True,
            )
