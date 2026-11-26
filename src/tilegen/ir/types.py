# src/tilegen/ir/types.py
#
# VTM abstract IR types: memory tiers, symbolic dims and buffers

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

ShapeDim = Union[int, "SymbolicDim"]


class MemoryTier(Enum):
    """The four functional VTM memory tiers."""

    GLOBAL = "global" # High-capacity off-chip memory (HBM/GM)
    SCRATCHPAD = "scratchpad" # Software-managed on-chip staging (SMEM/LDS/UB)
    COMPUTE_INPUT = "compute_input" # Operand staging feeding matrix units (REG/L0A,L0B)
    ACCUMULATOR = "accumulator" # High-precision accumulation (TMEM/AGPR/L0C)


class DataType(Enum):
    """Element data types used across the VTM IR."""

    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"
    INT8 = "int8"
    UINT8 = "uint8"
    INT32 = "int32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4_E2M1 = "fp4_e2m1"


@dataclass(frozen=True)
class SymbolicDim:
    """A dimension that may be symbolic at compile time.

    Sequence lengths and batch sizes are kept symbolic so a single AOT binary
    can serve many dynamic shapes; only architectural invariants (``head_dim``,
    ``dtype``, ``page_size``, ``quant_spec``) are specialized.
    """

    name: str
    value: Optional[int] = None
    min: Optional[int] = None
    max: Optional[int] = None

    def __post_init__(self) -> None:
        if self.value is not None and self.value <= 0:
            raise ValueError(f"SymbolicDim value must be positive, got {self.value}")
        if self.min is not None and self.min <= 0:
            raise ValueError(f"SymbolicDim min must be positive, got {self.min}")
        if self.max is not None and self.max <= 0:
            raise ValueError(f"SymbolicDim max must be positive, got {self.max}")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("SymbolicDim min cannot exceed max")

    @property
    def is_concrete(self) -> bool:
        return self.value is not None

    def concretize(self, value: int) -> "SymbolicDim":
        if self.min is not None and value < self.min:
            raise ValueError(f"{value} < min {self.min} for dim {self.name}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{value} > max {self.max} for dim {self.name}")
        return SymbolicDim(self.name, value=value, min=self.min, max=self.max)

    def __str__(self) -> str: # pragma: no cover - readability only
        return self.name if self.value is None else str(self.value)


@dataclass(frozen=True)
class VTMBuffer:
    """A typed multi-tier buffer in the VTM IR."""

    name: str
    tier: MemoryTier
    shape: Tuple[ShapeDim, ...]
    dtype: str = "fp32"
    swizzle: Optional[str] = None # NONE / B32 / B64 / B128

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("VTMBuffer requires a non-empty name")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier.value,
            "shape": [str(d) if isinstance(d, SymbolicDim) else d for d in self.shape],
            "dtype": self.dtype,
            "swizzle": self.swizzle,
        }


__all__ = ["MemoryTier", "DataType", "SymbolicDim", "VTMBuffer", "ShapeDim"]
