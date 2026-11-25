# src/tilegen/ir/from_dict.py
#
# VTM IR parser: reconstruct VTMKernel objects from serialized dicts (the
# inverse of VTMKernel.to_dict), with structural verification

from __future__ import annotations

import json
from typing import Any
from tilegen.ir.builder import verify_kernel
from tilegen.ir.ops import (
    AllGatherOp,
    AllReduceOp,
    DequantOp,
    GatherOp,
    LoadOp,
    MaskOp,
    MatmulOp,
    OnlineSoftmaxOp,
    PipelineForOp,
    ReduceScatterOp,
    RingStepOp,
    StoreOp,
    VTMKernel,
    VTMOp,
)
from tilegen.ir.topology import (
    ExecutionRoles,
    GridMode,
    GridTopology,
    Materialize,
    PhaseMode,
    RasterOrder,
    RecomputeBoundary,
    TacticalKnobs,
    TopologySpec,
)
from tilegen.ir.types import MemoryTier, SymbolicDim, VTMBuffer

# NOTE: tilegen.layout.layout imports tilegen.ir.types, so importing it here at
# module level would create a package-init cycle (tilegen.layout -> ir ->
# from_dict -> tilegen.layout).  Both layout imports are therefore lazy.


def _shape_dim(x: Any):
    """Parse a shape element back into an int or SymbolicDim.

    ``to_dict`` writes plain ints verbatim but renders SymbolicDims as their
    ``str()`` (a name when symbolic, the value when concrete).  Numeric strings
    are therefore rebuilt as *concrete* SymbolicDims so that the round-tripped
    kernel serializes byte-identically again.
    """
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    if isinstance(x, str):
        try:
            value = int(x)
        except ValueError:
            return SymbolicDim(name=x)
        return SymbolicDim(name=f"s{x}", value=value)
    raise TypeError(f"invalid shape element: {x!r}")


def _buffer(data: dict) -> VTMBuffer:
    if not isinstance(data, dict):
        raise TypeError("buffer entry must be a dict")
    try:
        return VTMBuffer(
            name=data["name"],
            tier=MemoryTier(data["tier"]),
            shape=tuple(_shape_dim(d) for d in data["shape"]),
            dtype=data.get("dtype", "fp32"),
            swizzle=data.get("swizzle"),
        )
    except KeyError as exc:
        raise ValueError(f"buffer entry missing field {exc}") from exc


def _layout(data: dict):
    from tilegen.layout.layout import ContiguousLayout, PagedKVLayout

    kind = data.get("kind")
    if kind == "contiguous":
        return ContiguousLayout(
            shape=tuple(_shape_dim(d) for d in data["shape"]),
            strides=tuple(data["strides"]) if data.get("strides") else None,
        )
    if kind == "paged_kv":
        return PagedKVLayout(
            block_size=data["block_size"],
            num_kv_heads=data["num_kv_heads"],
            head_dim=data["head_dim"],
        )
    raise ValueError(f"unknown layout kind {kind!r}")


def _topology(data: dict) -> TopologySpec:
    g = data.get("grid") or {}
    r = data.get("roles") or {}
    rc = data.get("recompute") or {}
    k = data.get("knobs") or {}
    knobs = {key: v for key, v in k.items() if key in TacticalKnobs.__dataclass_fields__}
    return TopologySpec(
        grid=GridTopology(
            mode=GridMode(g.get("mode", "data_parallel")),
            raster=RasterOrder(g.get("raster", "row_major")),
            split=g.get("split", 1),
            cta_group=g.get("cta_group", 1),
        ),
        roles=ExecutionRoles(
            producer_warpgroups=r.get("producer_warpgroups", 1),
            consumer_warpgroups=r.get("consumer_warpgroups", 1),
            phase_mode=PhaseMode(r.get("phase_mode", "serial")),
        ),
        recompute=RecomputeBoundary(
            default=Materialize(rc.get("default", "register_fused")),
            overrides={n: Materialize(v) for n, v in (rc.get("overrides") or {}).items()},
        ),
        knobs=TacticalKnobs(**knobs),
    )


def _op(data: dict) -> VTMOp:
    if not isinstance(data, dict):
        raise TypeError("body entry must be a dict")
    kind = data.get("op")
    if kind == "load":
        return LoadOp(data["source"], data["dest"], role=data.get("role", "memory_transfer"))
    if kind == "dequant":
        return DequantOp(
            data["source"],
            data["dest"],
            data["quant"],
            materialize=Materialize(data.get("materialize", "register_fused")),
            role=data.get("role", "memory_transfer"),
        )
    if kind == "matmul":
        return MatmulOp(
            data["a"],
            data["b"],
            data["acc"],
            transpose_b=data.get("transpose_b", False),
            clear=data.get("clear", True),
            role=data.get("role", "matrix_compute"),
        )
    if kind == "mask":
        return MaskOp(data["acc"], type=data.get("type", "CAUSAL"), role=data.get("role", "vector_postop"))
    if kind == "online_softmax":
        return OnlineSoftmaxOp(
            data["acc"],
            data["out"],
            scale=data.get("scale", 1.0),
            exp_emulation_ratio=data.get("exp_emulation_ratio", 0.0),
            rescale_threshold=data.get("rescale_threshold", 0.0),
            is_final=data.get("is_final", False),
            global_reduce=data.get("global_reduce", False),
            role=data.get("role", "vector_postop"),
        )
    if kind == "store":
        return StoreOp(data["source"], data["dest"], role=data.get("role", "memory_transfer"))
    if kind == "gather":
        return GatherOp(
            data["source"],
            data["dest"],
            data["block_table"],
            data["seq_len"],
            layout=data.get("layout", ""),
            role=data.get("role", "memory_transfer"),
        )
    if kind == "pipeline_for":
        return PipelineForOp(
            data["induction"],
            data["start"],
            data["end"],
            stages=data.get("stages", 1),
            body=[_op(o) for o in data.get("body", [])],
        )
    if kind == "all_gather":
        return AllGatherOp(
            data["tensor"], dim=data.get("dim", 0), degree=data.get("degree", 1),
            role=data.get("role", "collective"),
        )
    if kind == "reduce_scatter":
        return ReduceScatterOp(
            data["tensor"], dim=data.get("dim", 0), degree=data.get("degree", 1),
            op=data.get("op", "sum"), role=data.get("role", "collective"),
        )
    if kind == "all_reduce":
        return AllReduceOp(
            data["tensor"], reduce_op=data.get("reduce_op", "max"),
            role=data.get("role", "collective"),
        )
    if kind == "ring_step":
        return RingStepOp(
            data["send"], data["recv"],
            send_rank=data.get("send_rank", 0), recv_rank=data.get("recv_rank", 0),
            role=data.get("role", "collective"),
        )
    raise ValueError(f"unknown VTM op {kind!r}")


def vtm_kernel_from_dict(data: dict, verify: bool = True) -> VTMKernel:
    """Reconstruct a :class:`VTMKernel` from its ``to_dict`` serialization."""
    from tilegen.layout.quant import QuantSpec

    if not isinstance(data, dict):
        raise TypeError("kernel data must be a dict")
    if "name" not in data:
        raise ValueError("kernel data missing 'name'")
    kernel = VTMKernel(
        name=data["name"],
        inputs=tuple(_buffer(b) for b in data.get("inputs", [])),
        outputs=tuple(_buffer(b) for b in data.get("outputs", [])),
        buffers=tuple(_buffer(b) for b in data.get("buffers", [])),
        topology=_topology(data.get("topology") or {}),
        layouts={n: _layout(l) for n, l in (data.get("layouts") or {}).items()},
        quants={n: QuantSpec.from_dict(q) for n, q in (data.get("quants") or {}).items()},
        body=tuple(_op(o) for o in data.get("body", [])),
    )
    if verify:
        verify_kernel(kernel)
    return kernel


def vtm_kernel_from_json(text: str, verify: bool = True) -> VTMKernel:
    """Parse a JSON-serialized VTM kernel (the output of ``json.dumps(kernel.to_dict())``)."""
    return vtm_kernel_from_dict(json.loads(text), verify=verify)


__all__ = ["vtm_kernel_from_dict", "vtm_kernel_from_json"]
