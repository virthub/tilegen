# src/tilegen/layout/quant.py
#
# Unified quantization descriptor (QuantSpec)

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

# Bits per element for every supported storage encoding. Anything below 8
# bits is "sub-byte" and requires physical packing.
_BITS_PER_ELEMENT: dict[str, int] = {
    "int1": 1,
    "int2": 2,
    "int4": 4,
    "uint4": 4,
    "fp4_e2m1": 4,
    "int8": 8,
    "uint8": 8,
    "ue8m0": 8,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "fp16": 16,
    "float16": 16,
    "bf16": 16,
    "bfloat16": 16,
    "fp32": 32,
    "float32": 32,
    "tf32": 32,
    "fp64": 64,
    "float64": 64,
    "int32": 32,
    "uint32": 32,
    "int64": 64,
}

_VALID_GRANULARITIES = ("per_tensor", "per_channel", "per_token", "block")
_VALID_INTERLEAVES = (None, "awq_k8", "marlin", "linear_nibble")


@dataclass(frozen=True)
class QuantSpec:
    """Orthogonal quantization descriptor (INT4/INT8/FP4/FP8 and friends)."""

    dtype: str = "fp16"
    granularity: str = "per_tensor"
    block_size: Optional[int] = None
    scale_dtype: str = "float32"
    tensor_scale_dtype: Optional[str] = None
    pack_factor: Optional[int] = None
    pack_axis: int = -1
    has_zero_point: bool = False
    interleave: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dtype not in _BITS_PER_ELEMENT:
            raise ValueError(f"Unknown element dtype {self.dtype!r}")
        if self.granularity not in _VALID_GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {_VALID_GRANULARITIES}, got {self.granularity!r}"
            )
        if self.interleave not in _VALID_INTERLEAVES:
            raise ValueError(
                f"interleave must be one of {_VALID_INTERLEAVES}, got {self.interleave!r}"
            )
        if self.granularity == "block" and (self.block_size is None or self.block_size <= 0):
            raise ValueError("block granularity requires a positive block_size")
        if self.scale_dtype not in _BITS_PER_ELEMENT:
            raise ValueError(f"Unknown scale_dtype {self.scale_dtype!r}")
        if self.tensor_scale_dtype is not None and self.tensor_scale_dtype not in _BITS_PER_ELEMENT:
            raise ValueError(f"Unknown tensor_scale_dtype {self.tensor_scale_dtype!r}")

# A sub-byte dtype must pack multiple elements per byte; derive the
        # pack factor from the encoding when it was left unspecified.
        expected_pack = 8 // self.bits_per_element if self.bits_per_element < 8 else 1
        if self.pack_factor is None:
            object.__setattr__(self, "pack_factor", expected_pack)
        if self.pack_factor <= 0:
            raise ValueError("pack_factor must be positive")
        if self.bits_per_element < 8 and self.pack_factor != expected_pack:
            raise ValueError(
                f"sub-byte dtype {self.dtype!r} implies pack_factor={expected_pack}, "
                f"got {self.pack_factor}"
            )
        if self.bits_per_element >= 8 and self.pack_factor != 1:
            raise ValueError(
                f"byte-aligned dtype {self.dtype!r} implies pack_factor=1, got {self.pack_factor}"
            )

    @property
    def bits_per_element(self) -> int:
        return _BITS_PER_ELEMENT[self.dtype]

    @property
    def is_sub_byte(self) -> bool:
        return self.bits_per_element < 8

    @property
    def is_quantized(self) -> bool:
        return self.bits_per_element <= 8

    @property
    def is_block_scaled(self) -> bool:
        return self.granularity == "block"

    @property
    def format(self) -> str:
        """Coarse quantization format (``fp4``/``fp8``/``int4``/``int8``).

        Matches the keys of ``TargetSpec.quant_paths`` so capability lookup can
        use the element dtype directly.
        """
        d = self.dtype
        if d.startswith("fp4"):
            return "fp4"
        if d.startswith("fp8"):
            return "fp8"
        if d in ("int4", "uint4"):
            return "int4"
        if d in ("int8", "uint8"):
            return "int8"
        return d

    def packed_bytes(self, num_elements: int) -> int:
        """Bytes required to store ``num_elements`` logical elements."""
        if num_elements < 0:
            raise ValueError("num_elements must be non-negative")
        if not self.is_sub_byte:
            return num_elements * (self.bits_per_element // 8)
        return (num_elements + self.pack_factor - 1) // self.pack_factor

    def packed_stride(self, logical_dim: int) -> int:
        """Physical stride (in bytes) after packing ``logical_dim`` elements."""
        return self.packed_bytes(logical_dim)

    def num_scale_blocks(self, logical_dim: int) -> int:
        """Number of scale blocks covering ``logical_dim`` elements."""
        if self.granularity == "block":
            assert self.block_size is not None
            return (logical_dim + self.block_size - 1) // self.block_size
        if self.granularity == "per_channel":
            return logical_dim
        if self.granularity == "per_token":
            return logical_dim
        return 1

    def scale_shape(self, logical_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Shape of the per-tensor/per-channel/per-token/block scale tensor."""
        if self.granularity == "per_tensor":
            return ()
        if self.granularity == "per_channel":
            return (logical_shape[-1],)
        if self.granularity == "per_token":
            return (logical_shape[0],)
        assert self.block_size is not None
        return tuple(
            (dim + self.block_size - 1) // self.block_size if axis != 1 else dim
            for axis, dim in enumerate(logical_shape)
        )

    @classmethod
    def nvfp4(cls) -> "QuantSpec":
        """NVIDIA Blackwell NVFP4: fp4_e2m1, block=16, fp8_e4m3 scales."""
        return cls(
            dtype="fp4_e2m1",
            granularity="block",
            block_size=16,
            scale_dtype="fp8_e4m3",
            tensor_scale_dtype="float32",
            pack_factor=2,
        )

    @classmethod
    def mxfp4(cls) -> "QuantSpec":
        return cls(dtype="fp4_e2m1", granularity="block", block_size=32, scale_dtype="ue8m0", pack_factor=2)

    @classmethod
    def mxfp8(cls) -> "QuantSpec":
        return cls(dtype="fp8_e4m3", granularity="block", block_size=32, scale_dtype="ue8m0", pack_factor=1)

    @classmethod
    def awq_gptq_int4(cls) -> "QuantSpec":
        return cls(
            dtype="uint4",
            granularity="block",
            block_size=128,
            scale_dtype="float16",
            has_zero_point=True,
            pack_factor=2,
            interleave="awq_k8",
        )

    @classmethod
    def int4_kv(cls) -> "QuantSpec":
        return cls(
            dtype="int4",
            granularity="block",
            block_size=64,
            scale_dtype="float16",
            has_zero_point=True,
            pack_factor=2,
        )

    @classmethod
    def int8_kv(cls) -> "QuantSpec":
        return cls(
            dtype="int8",
            granularity="block",
            block_size=64,
            scale_dtype="float16",
            has_zero_point=True,
            pack_factor=1,
        )

    @classmethod
    def fp8_e4m3(cls) -> "QuantSpec":
        return cls(dtype="fp8_e4m3", granularity="per_tensor", scale_dtype="float32")

    def to_dict(self) -> dict:
        return {
            "dtype": self.dtype,
            "granularity": self.granularity,
            "block_size": self.block_size,
            "scale_dtype": self.scale_dtype,
            "tensor_scale_dtype": self.tensor_scale_dtype,
            "pack_factor": self.pack_factor,
            "pack_axis": self.pack_axis,
            "has_zero_point": self.has_zero_point,
            "interleave": self.interleave,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuantSpec":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _bits(dtype: str) -> int:
    try:
        return _BITS_PER_ELEMENT[dtype]
    except KeyError as exc: # pragma: no cover - guarded by QuantSpec.__post_init__
        raise ValueError(f"Unknown dtype {dtype!r}") from exc


__all__ = ["QuantSpec", "_BITS_PER_ELEMENT"]
