# src/tilegen/target/spec.py
#
# Hardware capability resolution (TargetSpec) and built-in profiles

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Mapping, Optional


@dataclass(frozen=True)
class TargetSpec:
    """A hardware capability profile used during VTM lowering."""

    name: str
    arch: str
    device: str = "cuda"                  # cuda / rocm / ascend
    sm: Optional[str] = None              # e.g. sm_90a, gfx942, ascend_910c
    num_sms: int = 108
    max_smem_bytes: int = 228 * 1024
    max_registers_per_thread: int = 255
    warp_size: int = 32
    features: FrozenSet[str] = frozenset()
    # format -> one of native / fused_dequant / emulated
    quant_paths: Mapping[str, str] = field(default_factory=dict)
    # VTM abstract tier -> hardware name
    memory_tiers: Mapping[str, str] = field(default_factory=dict)

    def supports(self, feature: str) -> bool:
        return feature in self.features

    def quant_path(self, dtype: str) -> str:
        """Return native / fused_dequant / emulated for a quantized dtype."""
        return self.quant_paths.get(dtype, "emulated")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arch": self.arch,
            "device": self.device,
            "sm": self.sm,
            "num_sms": self.num_sms,
            "max_smem_bytes": self.max_smem_bytes,
            "max_registers_per_thread": self.max_registers_per_thread,
            "warp_size": self.warp_size,
            "features": sorted(self.features),
            "quant_paths": dict(self.quant_paths),
            "memory_tiers": dict(self.memory_tiers),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TargetSpec":
        known = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in data.items() if k in known}
        if "features" in payload and isinstance(payload["features"], list):
            payload["features"] = frozenset(payload["features"])
        return cls(**payload)


def _gpu_tiers() -> dict[str, str]:
    return {
        "global": "HBM / Global Memory",
        "scratchpad": "SMEM",
        "compute_input": "Registers",
        "accumulator": "TMEM",
    }


BUILTIN_TARGETS: dict[str, TargetSpec] = {
    "sm_80": TargetSpec(
        name="sm_80",
        arch="NVIDIA Ampere A100",
        device="cuda",
        sm="sm_80",
        num_sms=108,
        max_smem_bytes=164 * 1024,
        features=frozenset({"mma_sync", "cp_async", "bfloat16", "tf32"}),
        quant_paths={"int8": "native", "int4": "fused_dequant", "fp8": "emulated", "fp4": "emulated"},
        memory_tiers={"global": "HBM", "scratchpad": "SMEM", "compute_input": "Registers", "accumulator": "Registers"},
    ),
    "sm_89": TargetSpec(
        name="sm_89",
        arch="NVIDIA Ada L40S",
        device="cuda",
        sm="sm_89",
        num_sms=142,
        max_smem_bytes=228 * 1024,
        features=frozenset({"mma_sync", "cp_async", "bfloat16", "tf32", "fp8"}),
        quant_paths={"int8": "native", "int4": "fused_dequant", "fp8": "native", "fp4": "emulated"},
        memory_tiers={"global": "HBM", "scratchpad": "SMEM", "compute_input": "Registers", "accumulator": "Registers"},
    ),
    "sm_90a": TargetSpec(
        name="sm_90a",
        arch="NVIDIA Hopper H100",
        device="cuda",
        sm="sm_90a",
        num_sms=132,
        max_smem_bytes=228 * 1024,
        features=frozenset({"tma", "wgmma", "warp_specialization", "mbarrier", "fp8", "bfloat16"}),
        quant_paths={"int8": "native", "int4": "fused_dequant", "fp8": "native", "fp4": "emulated"},
        memory_tiers=_gpu_tiers(),
    ),
    "sm_100a": TargetSpec(
        name="sm_100a",
        arch="NVIDIA Blackwell B200",
        device="cuda",
        sm="sm_100a",
        num_sms=148,
        max_smem_bytes=228 * 1024,
        features=frozenset({"tma", "tcgen05", "tensor_memory", "warp_specialization", "mbarrier", "fp8", "nvfp4", "mxfp4", "mxfp8", "2cta"}),
        quant_paths={"int8": "native", "int4": "fused_dequant", "fp8": "native", "fp4": "native"},
        memory_tiers=_gpu_tiers(),
    ),
    "sm_120": TargetSpec(
        name="sm_120",
        arch="NVIDIA Blackwell Ultra",
        device="cuda",
        sm="sm_120",
        num_sms=148,
        max_smem_bytes=228 * 1024,
        features=frozenset({"tma", "tcgen05", "tensor_memory", "warp_specialization", "mbarrier", "fp8", "nvfp4", "mxfp4", "mxfp8", "2cta"}),
        quant_paths={"int8": "native", "int4": "fused_dequant", "fp8": "native", "fp4": "native"},
        memory_tiers=_gpu_tiers(),
    ),
    "gfx942": TargetSpec(
        name="gfx942",
        arch="AMD CDNA3 MI300X",
        device="rocm",
        sm="gfx942",
        num_sms=304,
        max_smem_bytes=64 * 1024,
        max_registers_per_thread=256,
        features=frozenset({"mfma", "v_mfma", "fp8", "bfloat16"}),
        quant_paths={"int8": "native", "int4": "fused_dequant", "fp8": "native", "fp4": "emulated"},
        memory_tiers={"global": "HBM", "scratchpad": "LDS", "compute_input": "VGPR", "accumulator": "AGPR"},
    ),
    "ascend_910b": TargetSpec(
        name="ascend_910b",
        arch="Huawei Ascend 910B",
        device="ascend",
        sm="ascend_910b",
        num_sms=1,
        max_smem_bytes=192 * 1024,
        features=frozenset({"cube", "vector", "dma", "int8", "bfloat16"}),
        quant_paths={"int8": "native", "int4": "fused_dequant", "fp8": "emulated", "fp4": "emulated"},
        memory_tiers={"global": "GM", "scratchpad": "L1 / UB", "compute_input": "L0A / L0B", "accumulator": "L0C"},
    ),
    "ascend_910c": TargetSpec(
        name="ascend_910c",
        arch="Huawei Ascend 910C",
        device="ascend",
        sm="ascend_910c",
        num_sms=1,
        max_smem_bytes=192 * 1024,
        features=frozenset({"cube", "vector", "dma", "int8", "bfloat16"}),
        quant_paths={"int8": "native", "int4": "fused_dequant", "fp8": "emulated", "fp4": "emulated"},
        memory_tiers={"global": "GM", "scratchpad": "L1 / UB", "compute_input": "L0A / L0B", "accumulator": "L0C"},
    ),
    "ascend_950": TargetSpec(
        name="ascend_950",
        arch="Huawei Ascend 950",
        device="ascend",
        sm="ascend_950",
        num_sms=1,
        max_smem_bytes=256 * 1024,
        features=frozenset({"cube", "vector", "dma", "int8", "mxfp8", "mxfp4", "bfloat16"}),
        quant_paths={"int8": "native", "int4": "fused_dequant", "fp8": "native", "fp4": "native"},
        memory_tiers={"global": "GM", "scratchpad": "L1 / UB", "compute_input": "L0A / L0B", "accumulator": "L0C"},
    ),
}


def get_target(name: str) -> TargetSpec:
    if name not in BUILTIN_TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {sorted(BUILTIN_TARGETS)}")
    return BUILTIN_TARGETS[name]


def register_target(spec: TargetSpec) -> None:
    BUILTIN_TARGETS[spec.name] = spec


__all__ = ["TargetSpec", "BUILTIN_TARGETS", "get_target", "register_target"]
