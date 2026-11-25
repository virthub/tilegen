# src/tilegen/layout/layout.py
#
# Orthogonal layout descriptors

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
from tilegen.ir.types import ShapeDim, SymbolicDim

__all__ = ["ContiguousLayout", "PagedKVLayout", "TensorSpec"]


def _concrete(dim: ShapeDim) -> Optional[int]:
    return dim.value if isinstance(dim, SymbolicDim) else dim


@dataclass(frozen=True)
class ContiguousLayout:
    """Dense row-major (or explicit-stride) physical storage mapping."""

    shape: Tuple[ShapeDim, ...]
    strides: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.strides is None:
            # Row-major contiguous strides (elements, not bytes).
            object.__setattr__(self, "strides", self._row_major_strides(self.shape))

    @staticmethod
    def _row_major_strides(shape: Tuple[ShapeDim, ...]) -> Tuple[int, ...]:
        strides: list[int] = []
        acc = 1
        for dim in reversed(shape):
            c = _concrete(dim)
            if c is None:
                raise ValueError(
                    "ContiguousLayout with symbolic dims requires explicit strides"
                )
            strides.append(acc)
            acc *= c
        return tuple(reversed(strides))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> Optional[int]:
        total = 1
        for dim in self.shape:
            c = _concrete(dim)
            if c is None:
                return None
            total *= c
        return total

    def offset(self, indices: Sequence[int]) -> int:
        assert self.strides is not None
        if len(indices) != self.ndim:
            raise ValueError(
                f"expected {self.ndim} indices, got {len(indices)}"
            )
        return sum(i * s for i, s in zip(indices, self.strides))

    def to_dict(self) -> dict:
        return {
            "kind": "contiguous",
            "shape": [str(d) if isinstance(d, SymbolicDim) else d for d in self.shape],
            "strides": list(self.strides) if self.strides else None,
        }


@dataclass(frozen=True)
class PagedKVLayout:
    """Paged KV-cache storage mapping.

    Physical storage is ``[num_blocks, block_size, num_kv_heads, head_dim]``;
    a logical ``(kv_head, token_pos)`` coordinate is resolved through the
    block table (``block_tables``) and an in-page offset.
    """

    block_size: int          # tokens per page
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for name, value in (
            ("block_size", self.block_size),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    def num_blocks_for(self, seq_len: int) -> int:
        return (seq_len + self.block_size - 1) // self.block_size

    def in_page_offset(self, block_idx: int, token_pos: int, head: int) -> int:
        """Row-major element offset into the flattened page tensor."""
        if token_pos < 0 or token_pos >= self.block_size:
            raise ValueError(
                f"token_pos {token_pos} out of range [0, {self.block_size})"
            )
        if head < 0 or head >= self.num_kv_heads:
            raise ValueError(f"head {head} out of range [0, {self.num_kv_heads})")
        return (
            block_idx * (self.block_size * self.num_kv_heads * self.head_dim)
            + token_pos * (self.num_kv_heads * self.head_dim)
            + head * self.head_dim
        )

    def to_dict(self) -> dict:
        return {
            "kind": "paged_kv",
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }


LayoutSpec = ContiguousLayout | PagedKVLayout


@dataclass(frozen=True)
class TensorSpec:
    """The wrapper every input tensor enters the compiler as."""

    dtype: str
    shape: Tuple[ShapeDim, ...]
    layout: LayoutSpec
    quant: object | None = None  # QuantSpec | None (avoid import cycle at type level)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_dict(self) -> dict:
        from tilegen.layout.quant import QuantSpec

        return {
            "dtype": self.dtype,
            "shape": [str(d) if isinstance(d, SymbolicDim) else d for d in self.shape],
            "layout": self.layout.to_dict(),
            "quant": self.quant.to_dict() if isinstance(self.quant, QuantSpec) else None,
        }
