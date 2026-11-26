# src/tilegen/ir/ops.py
#
# Typed tile IR nodes and the VTMKernel container

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from tilegen.ir.topology import Materialize, TopologySpec
from tilegen.ir.types import VTMBuffer


@dataclass(frozen=True)
class VTMOp:
    """Base class for all typed VTM tile operations."""

    op: str

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["op"] = self.op
        return d


@dataclass(frozen=True)
class LoadOp(VTMOp):
    source: str = ""
    dest: str = ""
    role: str = "memory_transfer"

    def __init__(self, source: str, dest: str, role: str = "memory_transfer", **kw: Any) -> None:
        object.__setattr__(self, "op", "load")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "dest", dest)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class DequantOp(VTMOp):
    source: str = ""
    dest: str = ""
    quant: str = ""
    materialize: Materialize = Materialize.REGISTER_FUSED
    role: str = "memory_transfer"

    def __init__(
        self,
        source: str,
        dest: str,
        quant: str,
        materialize: Materialize = Materialize.REGISTER_FUSED,
        role: str = "memory_transfer",
        **kw: Any,
    ) -> None:
        object.__setattr__(self, "op", "dequant")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "dest", dest)
        object.__setattr__(self, "quant", quant)
        object.__setattr__(self, "materialize", materialize)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class MatmulOp(VTMOp):
    a: str = ""
    b: str = ""
    acc: str = ""
    transpose_b: bool = False
    clear: bool = True
    role: str = "matrix_compute"

    def __init__(
        self,
        a: str,
        b: str,
        acc: str,
        transpose_b: bool = False,
        clear: bool = True,
        role: str = "matrix_compute",
        **kw: Any,
    ) -> None:
        object.__setattr__(self, "op", "matmul")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "acc", acc)
        object.__setattr__(self, "transpose_b", transpose_b)
        object.__setattr__(self, "clear", clear)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class MaskOp(VTMOp):
    acc: str = ""
    type: str = "CAUSAL"
    role: str = "vector_postop"

    def __init__(self, acc: str, type: str = "CAUSAL", role: str = "vector_postop", **kw: Any) -> None:
        object.__setattr__(self, "op", "mask")
        object.__setattr__(self, "acc", acc)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class OnlineSoftmaxOp(VTMOp):
    acc: str = ""
    out: str = ""
    scale: float = 1.0
    exp_emulation_ratio: float = 0.0
    rescale_threshold: float = 0.0
    is_final: bool = False
    global_reduce: bool = False # all-reduce max/sum across CP/SP ranks
    role: str = "vector_postop"

    def __init__(
        self,
        acc: str,
        out: str,
        scale: float = 1.0,
        exp_emulation_ratio: float = 0.0,
        rescale_threshold: float = 0.0,
        is_final: bool = False,
        global_reduce: bool = False,
        role: str = "vector_postop",
        **kw: Any,
    ) -> None:
        object.__setattr__(self, "op", "online_softmax")
        object.__setattr__(self, "acc", acc)
        object.__setattr__(self, "out", out)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "exp_emulation_ratio", exp_emulation_ratio)
        object.__setattr__(self, "rescale_threshold", rescale_threshold)
        object.__setattr__(self, "is_final", is_final)
        object.__setattr__(self, "global_reduce", global_reduce)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class StoreOp(VTMOp):
    source: str = ""
    dest: str = ""
    role: str = "memory_transfer"

    def __init__(self, source: str, dest: str, role: str = "memory_transfer", **kw: Any) -> None:
        object.__setattr__(self, "op", "store")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "dest", dest)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class GatherOp(VTMOp):
    """Paged-KV gather: materialize a variable-length sequence from a paged cache.

    For program ``r``, walks ``block_table[r, :]`` and gathers ``seq_len[r]``
    tokens of ``source`` into ``dest`` (block-table indirection).
    """

    source: str = ""
    dest: str = ""
    block_table: str = ""
    seq_len: str = ""
    layout: str = ""
    role: str = "memory_transfer"

    def __init__(
        self,
        source: str,
        dest: str,
        block_table: str,
        seq_len: str,
        layout: str = "",
        role: str = "memory_transfer",
        **kw: Any,
    ) -> None:
        object.__setattr__(self, "op", "gather")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "dest", dest)
        object.__setattr__(self, "block_table", block_table)
        object.__setattr__(self, "seq_len", seq_len)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class PipelineForOp(VTMOp):
    induction: str = ""
    start: str = "0"
    end: str = "1"
    stages: int = 1
    body: tuple = () # tuple[VTMOp, ...]

    def __init__(
        self,
        induction: str,
        start: str,
        end: str,
        stages: int = 1,
        body: Sequence[VTMOp] = (),
        **kw: Any,
    ) -> None:
        object.__setattr__(self, "op", "pipeline_for")
        object.__setattr__(self, "induction", induction)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "body", tuple(body))


@dataclass(frozen=True)
class AllGatherOp(VTMOp):
    """Collective: all-gather a sequence-sharded tensor along ``dim``.

    Each CP/SP rank contributes its local shard; the result is the full
    concatenation (the Approach-A CP K/V visibility step).  Lowered host-side
    to ``torch.distributed.all_gather`` by the :class:`ParallelExecutor`.
    """

    tensor: str = ""
    dim: int = 0
    degree: int = 1
    role: str = "collective"

    def __init__(self, tensor: str, dim: int = 0, degree: int = 1, role: str = "collective", **kw: Any) -> None:
        object.__setattr__(self, "op", "all_gather")
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class ReduceScatterOp(VTMOp):
    """Collective: reduce-scatter an output tensor along ``dim`` back to owners."""

    tensor: str = ""
    dim: int = 0
    degree: int = 1
    op: str = "sum" # sum | max | min
    role: str = "collective"

    def __init__(self, tensor: str, dim: int = 0, degree: int = 1, op: str = "sum", role: str = "collective", **kw: Any) -> None:
        object.__setattr__(self, "op", "reduce_scatter")
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class AllReduceOp(VTMOp):
    """Collective: all-reduce softmax statistics (max/sum) across CP/SP ranks."""

    tensor: str = ""
    reduce_op: str = "max" # max | sum
    role: str = "collective"

    def __init__(self, tensor: str, reduce_op: str = "max", role: str = "collective", **kw: Any) -> None:
        object.__setattr__(self, "op", "all_reduce")
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "reduce_op", reduce_op)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class RingStepOp(VTMOp):
    """Collective: one P2P ring step (send local shard to ``send_rank``, receive
    from ``recv_rank``).  Represents the in-kernel ring-attention pattern
    (Approach B); not yet lowered by the runnable backends."""

    send: str = ""
    recv: str = ""
    send_rank: int = 0
    recv_rank: int = 0
    role: str = "collective"

    def __init__(self, send: str, recv: str, send_rank: int = 0, recv_rank: int = 0, role: str = "collective", **kw: Any) -> None:
        object.__setattr__(self, "op", "ring_step")
        object.__setattr__(self, "send", send)
        object.__setattr__(self, "recv", recv)
        object.__setattr__(self, "send_rank", send_rank)
        object.__setattr__(self, "recv_rank", recv_rank)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class VTMKernel:
    """A serializable VTM kernel: buffers, topology, layouts, quants, body."""

    name: str
    inputs: tuple = () # tuple[VTMBuffer, ...]
    outputs: tuple = () # tuple[VTMBuffer, ...]
    buffers: tuple = () # tuple[VTMBuffer, ...] internal allocations
    topology: TopologySpec = field(default_factory=TopologySpec)
    layouts: dict = field(default_factory=dict) # name -> layout descriptor
    quants: dict = field(default_factory=dict) # name -> QuantSpec
    body: tuple = () # tuple[VTMOp, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "buffers", tuple(self.buffers))
        object.__setattr__(self, "body", tuple(self.body))

    @classmethod
    def from_dict(cls, data: dict, verify: bool = True) -> "VTMKernel":
        """Reconstruct a kernel from its ``to_dict`` serialization."""
        from tilegen.ir.from_dict import vtm_kernel_from_dict

        return vtm_kernel_from_dict(data, verify=verify)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inputs": [b.to_dict() for b in self.inputs],
            "outputs": [b.to_dict() for b in self.outputs],
            "buffers": [b.to_dict() for b in self.buffers],
            "topology": self.topology.to_dict(),
            "layouts": {k: _layout_to_dict(v) for k, v in self.layouts.items()},
            "quants": {k: _quant_to_dict(v) for k, v in self.quants.items()},
            "body": [_op_to_dict(op) for op in self.body],
        }

    def serialize(self) -> str:
        """Render the kernel as VTM text."""
        lines: list[str] = [f"vtm.kernel @{self.name}("]
        for b in self.inputs:
            lines.append(f"    %{b.name}: buffer<{b.tier.value}, [{_fmt_shape(b)}], {b.dtype}>,")
        for b in self.outputs:
            suffix = "," if b is not self.outputs[-1] else ""
            lines.append(f"    %{b.name}: buffer<{b.tier.value}, [{_fmt_shape(b)}], {b.dtype}>{suffix}")
        t = self.topology
        lines.append(") attributes {")
        lines.append(
            f"    topology = #vtm.grid<mode = {t.grid.mode.value.upper()}, "
            f"raster = {t.grid.raster.value.upper()}, split = {t.grid.split}, "
            f"cta_group = {t.grid.cta_group}>,"
        )
        lines.append(
            f"    roles = #vtm.roles<producer_wg = {t.roles.producer_warpgroups}, "
            f"consumer_wg = {t.roles.consumer_warpgroups}, "
            f"phase = {t.roles.phase_mode.value.upper()}>,"
        )
        k = t.knobs
        lines.append(
            f"    knobs = #vtm.knobs<tile = [{k.tile_m}, {k.tile_n}, {k.tile_k}], "
            f"stages = {k.pipeline_stages}, wait_depth = {k.async_wait_depth}, "
            f"regs = {k.max_registers_per_thread}, swizzle = {k.swizzle_mode}>,"
        )
        if self.layouts:
            lines.append(f"    layouts = {{ {', '.join(f'{n} = #{n}' for n in self.layouts)} }}, ")
        if self.quants:
            lines.append(f"    quants = {{ {', '.join(f'{n} = #{n}' for n in self.quants)} }}")
        lines.append("} {")
        for b in self.buffers:
            lines.append(f"    %{b.name} = vtm.alloc<{b.tier.value}, [{_fmt_shape(b)}], {b.dtype}>")
        for op in self.body:
            lines.append("    " + _serialize_op(op))
        lines.append("}")
        return "\n".join(lines)


def _fmt_shape(buf: VTMBuffer) -> str:
    return ", ".join(str(d) for d in buf.shape)


def _serialize_op(op: VTMOp) -> str:
    if isinstance(op, LoadOp):
        return f"vtm.load %{op.source} -> %{op.dest} {{role = {op.role}}}"
    if isinstance(op, DequantOp):
        return (
            f"%{op.dest} = vtm.dequant %{op.source} -> compute_input "
            f"{{quant = @{op.quant}, materialize = {op.materialize.value.upper()}}}"
        )
    if isinstance(op, MatmulOp):
        return (
            f"vtm.matmul %{op.a}, %{op.b}, acc = %{op.acc} "
            f"{{transpose_b = {str(op.transpose_b).lower()}, clear = {str(op.clear).lower()}, "
            f"role = {op.role}}}"
        )
    if isinstance(op, MaskOp):
        return f"vtm.mask %{op.acc} {{type = {op.type}, role = {op.role}}}"
    if isinstance(op, OnlineSoftmaxOp):
        return (
            f"%{op.out} = vtm.online_softmax %{op.acc}, %{op.out} {{"
            f"scale = {op.scale}, exp_emulation_ratio = {op.exp_emulation_ratio}, "
            f"rescale_threshold = {op.rescale_threshold}, is_final = {str(op.is_final).lower()}, "
            f"global_reduce = {str(op.global_reduce).lower()}}}"
        )
    if isinstance(op, StoreOp):
        return f"vtm.store %{op.source} -> %{op.dest} {{role = {op.role}}}"
    if isinstance(op, GatherOp):
        return (
            f"%{op.dest} = vtm.gather %{op.source} from %{op.block_table}, %{op.seq_len} "
            f"{{layout = @{op.layout}, role = {op.role}}}"
        )
    if isinstance(op, AllGatherOp):
        return f"vtm.all_gather %{op.tensor} {{dim = {op.dim}, degree = {op.degree}}}"
    if isinstance(op, ReduceScatterOp):
        return f"vtm.reduce_scatter %{op.tensor} {{dim = {op.dim}, degree = {op.degree}, op = {op.op}}}"
    if isinstance(op, AllReduceOp):
        return f"vtm.all_reduce %{op.tensor} {{reduce_op = {op.reduce_op}}}"
    if isinstance(op, RingStepOp):
        return f"vtm.ring_step %{op.send} -> %{op.recv} {{send_rank = {op.send_rank}, recv_rank = {op.recv_rank}}}"
    if isinstance(op, PipelineForOp):
        inner = "\n".join("      " + _serialize_op(o) for o in op.body)
        return (
            f"vtm.pipeline_for %{op.induction} in ({op.start} .. {op.end}) "
            f"stages({op.stages}) {{\n{inner}\n    }}"
        )
    return repr(op)


def _layout_to_dict(layout: Any) -> dict:
    return layout.to_dict() if hasattr(layout, "to_dict") else dict(layout)


def _quant_to_dict(quant: Any) -> dict:
    if hasattr(quant, "to_dict"):
        return quant.to_dict()
    return dataclass_to_dict(quant)


def _op_to_dict(op: VTMOp) -> dict:
    d = dict(op.__dict__)
    d["op"] = op.op
    if isinstance(op, PipelineForOp):
        d["body"] = [_op_to_dict(o) for o in op.body]
    if isinstance(op, DequantOp):
        d["materialize"] = op.materialize.value
    return d


def dataclass_to_dict(obj: Any) -> dict:
    """Fallback: convert a dataclass-like object to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"cannot convert {obj!r} to dict")


__all__ = [
    "VTMOp",
    "LoadOp",
    "DequantOp",
    "MatmulOp",
    "MaskOp",
    "OnlineSoftmaxOp",
    "StoreOp",
    "GatherOp",
    "PipelineForOp",
    "AllGatherOp",
    "ReduceScatterOp",
    "AllReduceOp",
    "RingStepOp",
    "VTMKernel",
]
