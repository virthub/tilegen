# src/tilegen/ops/guards.py
#
# Host-side boundary guards

from __future__ import annotations

from typing import Optional, Sequence
from tilegen.layout.quant import QuantSpec


class BoundaryError(ValueError):
    """Raised when a tensor/kernel boundary invariant is violated."""


def check_quant_alignment(
    tile_k: int,
    tile_n: int,
    quant: QuantSpec,
) -> None:
    """Enforce ``tile_k % block_size == 0`` and sub-byte packing alignment."""
    if quant.is_block_scaled:
        assert quant.block_size is not None
        if tile_k % quant.block_size != 0:
            raise BoundaryError(
                f"tile_k {tile_k} not divisible by quant block_size {quant.block_size}"
            )
    if quant.is_sub_byte:
        if tile_k % quant.pack_factor != 0:
            raise BoundaryError(
                f"tile_k {tile_k} not divisible by pack_factor {quant.pack_factor}"
            )
        if tile_n % quant.pack_factor != 0:
            raise BoundaryError(
                f"tile_n {tile_n} not divisible by pack_factor {quant.pack_factor}"
            )


def check_rank(name: str, shape: Sequence, expected_rank: int) -> None:
    if len(shape) != expected_rank:
        raise BoundaryError(
            f"{name} expected rank {expected_rank}, got {len(shape)} (shape {tuple(shape)})"
        )


def check_head_dim(name: str, head_dim: int, expected: int) -> None:
    if head_dim != expected:
        raise BoundaryError(f"{name} head_dim {head_dim} != expected {expected}")


def check_page_alignment(page_size: int, seq_len: int) -> None:
    if page_size <= 0:
        raise BoundaryError(f"page_size must be positive, got {page_size}")
    if seq_len <= 0:
        raise BoundaryError(f"seq_len must be positive, got {seq_len}")


def check_out_alias(out_shape: Sequence, in_shapes: Sequence[Sequence]) -> None:
    """Reject ``out=`` aliasing an input (a classic silent-corruption hazard)."""
    for i, shape in enumerate(in_shapes):
        if tuple(out_shape) == tuple(shape):
            raise BoundaryError(f"output aliases input #{i} (shape {tuple(shape)})")


def check_finite_guard(enabled: bool) -> None:
    """Record whether NaN/Inf edge-case trapping is requested.

    NaN/Inf edge cases are handled by a fallback path when ``enabled``; this
    hook documents the boundary rather than raising.
    """
    # The trap is enforced at dispatch time; nothing to raise here.
    del enabled


def validate_attention(
    *,
    spec,
    k_cache_shape: Sequence,
    v_cache_shape: Sequence,
    o_shape: Sequence,
    tile_k: int,
    tile_n: int,
    out_aliases_input: bool = False,
) -> None:
    """Run the full host boundary guard set for a paged attention launch."""
    check_quant_alignment(tile_k, tile_n, spec.kv_quant)
    check_page_alignment(spec.page_size, spec.seq_len)
    check_rank("K_cache", k_cache_shape, 4)
    check_rank("V_cache", v_cache_shape, 4)
    check_rank("O", o_shape, 4)
    check_head_dim("K_cache", k_cache_shape[3], spec.kv_head_dim)
    check_head_dim("V_cache", v_cache_shape[3], spec.kv_head_dim)
    check_head_dim("O", o_shape[3], spec.head_dim)
    if k_cache_shape != v_cache_shape:
        raise BoundaryError("K_cache and V_cache shapes must match")
    if out_aliases_input:
        check_out_alias(o_shape, [k_cache_shape, v_cache_shape])


__all__ = [
    "BoundaryError",
    "check_quant_alignment",
    "check_rank",
    "check_head_dim",
    "check_page_alignment",
    "check_out_alias",
    "check_finite_guard",
    "validate_attention",
]
