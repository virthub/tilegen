# src/tilegen/ir/builder.py
#
# Python VTM IR builder & structural verifier

from __future__ import annotations

from typing import Iterable, Optional, Sequence
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
from tilegen.ir.topology import Materialize, TopologySpec
from tilegen.ir.types import MemoryTier, VTMBuffer
from tilegen.layout.quant import QuantSpec


class VTMBuilder:
    """Fluent builder for a single VTM kernel."""

    def __init__(self, name: str, topology: Optional[TopologySpec] = None) -> None:
        self.name = name
        self.topology = topology or TopologySpec()
        self._inputs: list[VTMBuffer] = []
        self._outputs: list[VTMBuffer] = []
        self._buffers: list[VTMBuffer] = []
        self._body: list[VTMOp] = []
        self._layouts: dict[str, object] = {}
        self._quants: dict[str, QuantSpec] = {}

    def input(self, name: str, shape, dtype: str = "fp32") -> "VTMBuilder":
        self._inputs.append(VTMBuffer(name, MemoryTier.GLOBAL, tuple(shape), dtype))
        return self

    def output(self, name: str, shape, dtype: str = "fp32") -> "VTMBuilder":
        self._outputs.append(VTMBuffer(name, MemoryTier.GLOBAL, tuple(shape), dtype))
        return self

    def alloc(
        self,
        name: str,
        shape,
        dtype: str = "fp32",
        tier: MemoryTier = MemoryTier.SCRATCHPAD,
        swizzle: Optional[str] = None,
    ) -> "VTMBuilder":
        self._buffers.append(VTMBuffer(name, tier, tuple(shape), dtype, swizzle))
        return self

    def layout(self, name: str, layout: object) -> "VTMBuilder":
        self._layouts[name] = layout
        return self

    def quant(self, name: str, quant: QuantSpec) -> "VTMBuilder":
        self._quants[name] = quant
        return self

    def load(self, source: str, dest: str, role: str = "memory_transfer") -> "VTMBuilder":
        self._body.append(LoadOp(source, dest, role))
        return self

    def dequant(
        self,
        source: str,
        dest: str,
        quant: str,
        materialize: Materialize = Materialize.REGISTER_FUSED,
    ) -> "VTMBuilder":
        self._body.append(DequantOp(source, dest, quant, materialize))
        return self

    def matmul(
        self,
        a: str,
        b: str,
        acc: str,
        transpose_b: bool = False,
        clear: bool = True,
    ) -> "VTMBuilder":
        self._body.append(MatmulOp(a, b, acc, transpose_b, clear))
        return self

    def mask(self, acc: str, type: str = "CAUSAL") -> "VTMBuilder":
        self._body.append(MaskOp(acc, type))
        return self

    def online_softmax(
        self,
        acc: str,
        out: str,
        scale: float = 1.0,
        exp_emulation_ratio: float = 0.0,
        rescale_threshold: float = 0.0,
        is_final: bool = False,
        global_reduce: bool = False,
    ) -> "VTMBuilder":
        self._body.append(
            OnlineSoftmaxOp(acc, out, scale, exp_emulation_ratio, rescale_threshold, is_final, global_reduce)
        )
        return self

    def store(self, source: str, dest: str, role: str = "memory_transfer") -> "VTMBuilder":
        self._body.append(StoreOp(source, dest, role))
        return self

    def gather(
        self,
        source: str,
        dest: str,
        block_table: str,
        seq_len: str,
        layout: str = "",
    ) -> "VTMBuilder":
        self._body.append(GatherOp(source, dest, block_table, seq_len, layout))
        return self

    def all_gather(self, tensor: str, dim: int = 0, degree: int = 1) -> "VTMBuilder":
        self._body.append(AllGatherOp(tensor, dim, degree))
        return self

    def reduce_scatter(self, tensor: str, dim: int = 0, degree: int = 1, op: str = "sum") -> "VTMBuilder":
        self._body.append(ReduceScatterOp(tensor, dim, degree, op))
        return self

    def all_reduce(self, tensor: str, reduce_op: str = "max") -> "VTMBuilder":
        self._body.append(AllReduceOp(tensor, reduce_op))
        return self

    def ring_step(self, send: str, recv: str, send_rank: int = 0, recv_rank: int = 0) -> "VTMBuilder":
        self._body.append(RingStepOp(send, recv, send_rank, recv_rank))
        return self

    def pipeline_for(
        self,
        induction: str,
        start: str,
        end: str,
        stages: int,
        body: Sequence[VTMOp],
    ) -> "VTMBuilder":
        self._body.append(PipelineForOp(induction, start, end, stages, body))
        return self

    def build(self) -> VTMKernel:
        kernel = VTMKernel(
            name=self.name,
            inputs=tuple(self._inputs),
            outputs=tuple(self._outputs),
            buffers=tuple(self._buffers),
            topology=self.topology,
            layouts=dict(self._layouts),
            quants=dict(self._quants),
            body=tuple(self._body),
        )
        verify_kernel(kernel)
        return kernel


def verify_kernel(kernel: VTMKernel) -> None:
    """Check structural invariants before lowering."""
    defined = _defined_buffers(kernel)
    for b in kernel.buffers:
        if b.name in {x.name for x in kernel.inputs} | {x.name for x in kernel.outputs}:
            raise ValueError(f"internal buffer {b.name!r} shadows an input/output")
    for op in _walk(kernel.body):
        _verify_op(op, defined, kernel)

    # Quantization block alignment: tile_k % block_size == 0.
    tile_k = kernel.topology.knobs.tile_k
    for name, quant in kernel.quants.items():
        if quant.is_block_scaled and quant.block_size is not None:
            if tile_k % quant.block_size != 0:
                raise ValueError(
                    f"quant {name!r}: tile_k {tile_k} not divisible by "
                    f"block_size {quant.block_size}"
                )
    # Sub-byte packing requires tile alignment to pack_factor too.
    for name, quant in kernel.quants.items():
        if quant.is_sub_byte and tile_k % quant.pack_factor != 0:
            raise ValueError(
                f"quant {name!r}: tile_k {tile_k} not divisible by pack_factor {quant.pack_factor}"
            )


def _defined_buffers(kernel: VTMKernel) -> dict[str, VTMBuffer]:
    defined: dict[str, VTMBuffer] = {}
    for b in kernel.inputs + kernel.outputs + kernel.buffers:
        defined[b.name] = b
    return defined


def _walk(ops: Iterable[VTMOp]):
    for op in ops:
        yield op
        if isinstance(op, PipelineForOp):
            yield from _walk(op.body)


def _verify_op(op: VTMOp, defined: dict[str, VTMBuffer], kernel: VTMKernel) -> None:
    if isinstance(op, LoadOp):
        _require(op.source, defined)
        _require(op.dest, defined)
    elif isinstance(op, DequantOp):
        _require(op.source, defined)
        _require(op.dest, defined)
        if op.quant not in kernel.quants:
            raise ValueError(f"dequant references undefined quant descriptor @{op.quant}")
    elif isinstance(op, MatmulOp):
        for name in (op.a, op.b, op.acc):
            _require(name, defined)
    elif isinstance(op, MaskOp):
        _require(op.acc, defined)
    elif isinstance(op, OnlineSoftmaxOp):
        _require(op.acc, defined)
        _require(op.out, defined)
    elif isinstance(op, StoreOp):
        _require(op.source, defined)
        _require(op.dest, defined)
    elif isinstance(op, GatherOp):
        for name in (op.source, op.dest, op.block_table, op.seq_len):
            _require(name, defined)
        if op.layout and op.layout not in kernel.layouts:
            raise ValueError(f"gather references undefined layout @{op.layout}")
    elif isinstance(op, AllGatherOp):
        _require(op.tensor, defined)
        if op.degree <= 0:
            raise ValueError("all_gather degree must be positive")
    elif isinstance(op, ReduceScatterOp):
        _require(op.tensor, defined)
        if op.degree <= 0:
            raise ValueError("reduce_scatter degree must be positive")
    elif isinstance(op, AllReduceOp):
        _require(op.tensor, defined)
        if op.reduce_op not in ("max", "sum", "min"):
            raise ValueError(f"all_reduce reduce_op {op.reduce_op!r} invalid")
    elif isinstance(op, RingStepOp):
        _require(op.send, defined)
        _require(op.recv, defined)
    elif isinstance(op, PipelineForOp):
        if op.stages <= 0:
            raise ValueError("pipeline_for stages must be positive")


def _require(name: str, defined: dict[str, VTMBuffer]) -> None:
    if name not in defined:
        raise ValueError(f"reference to undefined buffer %{name}")


__all__ = ["VTMBuilder", "verify_kernel"]
