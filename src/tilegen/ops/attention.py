# src/tilegen/ops/attention.py
#
# Declarative AttentionSpec and a VTM builder for paged flash attention

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from tilegen.ir.builder import VTMBuilder
from tilegen.ir.ops import (
    DequantOp,
    LoadOp,
    MaskOp,
    MatmulOp,
    OnlineSoftmaxOp,
    VTMKernel,
)
from tilegen.ir.topology import Materialize, TopologySpec
from tilegen.ir.types import MemoryTier, SymbolicDim
from tilegen.layout.layout import PagedKVLayout
from tilegen.layout.quant import QuantSpec
from tilegen.ops.schedules import attention_topology
from tilegen.target.spec import TargetSpec


@dataclass(frozen=True)
class AttentionSpec:
    """Declarative attention semantics (SDPA / FA1-FA4 share this interface)."""

    batch: int = 1
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int = 128
    kv_head_dim: int = 64
    seq_len: int = 2048
    num_kv_blocks: int = 256
    page_size: int = 32
    dtype: str = "bf16"
    kv_quant: QuantSpec = field(default_factory=QuantSpec.int4_kv)
    causal: bool = True
    scale: Optional[float] = None
# Decode-specific (query_len=1, variable kv_len per request).
    mode: str = "prefill" # "prefill" | "decode"
    query_len: int = 1

    def __post_init__(self) -> None:
        if self.scale is None:
            object.__setattr__(self, "scale", 1.0 / (self.head_dim ** 0.5))


def build_attention_vtm(
    spec: AttentionSpec,
    target: TargetSpec,
    variant: str = "auto",
    name: str = "paged_flash_attn_fp4",
) -> VTMKernel:
    """Emit the VTM IR for a paged, quantized flash-attention kernel."""
    s_q = SymbolicDim("S_q", value=spec.seq_len)
    num_blk = SymbolicDim("NumBlocks", value=spec.num_kv_blocks)
    topology = attention_topology(variant, target)

    b = VTMBuilder(name, topology)
    b.input("Q", (spec.batch, spec.num_heads, s_q, spec.head_dim), spec.dtype)
    b.input("K_cache", (num_blk, spec.page_size, spec.num_kv_heads, spec.kv_head_dim), "uint8")
    b.input("V_cache", (num_blk, spec.page_size, spec.num_kv_heads, spec.kv_head_dim), "uint8")
    b.input("block_tables", (spec.batch, spec.num_heads, num_blk), "int32")
    b.output("O", (spec.batch, spec.num_heads, s_q, spec.head_dim), spec.dtype)

    b.layout(
        "kv_paged",
        PagedKVLayout(
            block_size=spec.page_size, num_kv_heads=spec.num_kv_heads, head_dim=spec.kv_head_dim
        ),
    )
    b.quant("kv_quant", spec.kv_quant)

    b.alloc("q_smem", (128, spec.head_dim), spec.dtype, MemoryTier.SCRATCHPAD, "B128")
    b.alloc("k_smem", (3, 128, spec.kv_head_dim), "uint8", MemoryTier.SCRATCHPAD, "B128")
    b.alloc("v_smem", (3, 128, spec.kv_head_dim), "uint8", MemoryTier.SCRATCHPAD, "B128")
    b.alloc("s_acc", (128, 128), "fp32", MemoryTier.ACCUMULATOR)
    b.alloc("o_acc", (128, 128), "fp32", MemoryTier.ACCUMULATOR)
    b.alloc("k_reg", (128, spec.kv_head_dim), spec.dtype, MemoryTier.COMPUTE_INPUT)
    b.alloc("v_reg", (128, spec.kv_head_dim), spec.dtype, MemoryTier.COMPUTE_INPUT)
    b.alloc("p_reg", (128, 128), spec.dtype, MemoryTier.COMPUTE_INPUT)

    b.load("Q", "q_smem")

    loop_body = [
        LoadOp("K_cache", "k_smem", "memory_transfer"),
        LoadOp("V_cache", "v_smem", "memory_transfer"),
        DequantOp("k_smem", "k_reg", "kv_quant", Materialize.REGISTER_FUSED),
        MatmulOp("q_smem", "k_reg", "s_acc", transpose_b=True, clear=True),
        MaskOp("s_acc", "CAUSAL" if spec.causal else "NONE"),
        OnlineSoftmaxOp("s_acc", "p_reg", scale=spec.scale or 1.0, is_final=False),
        DequantOp("v_smem", "v_reg", "kv_quant", Materialize.REGISTER_FUSED),
        MatmulOp("p_reg", "v_reg", "o_acc", transpose_b=False, clear=False),
    ]

    b.pipeline_for("kv_blk", "0", "%num_kv_blocks", topology.knobs.pipeline_stages, loop_body)
    b.online_softmax("s_acc", "o_acc", scale=spec.scale or 1.0, is_final=True)
    b.store("o_acc", "O")

    return b.build()


def build_paged_attention_vtm(
    spec: AttentionSpec, target: TargetSpec, name: str = "paged_flash_attn_fp4"
) -> VTMKernel:
    """Backward-compatible alias: build with the target's default FA variant."""
    return build_attention_vtm(spec, target, variant="auto", name=name)


def build_decode_vtm(
    spec: AttentionSpec,
    target: TargetSpec,
    name: str = "paged_decode",
) -> VTMKernel:
    """Emit the VTM IR for a batched paged-KV decode (one program per request).

    Uses :class:`GatherOp` (block-table indirection) to materialize each
    request's variable-length K/V, then a 1-query attention over the gathered
    sequence.  This is the IR the runnable Triton codegen targets.
    """
    num_reqs = SymbolicDim("num_reqs", value=spec.batch)
    max_seq = SymbolicDim("max_seq", value=spec.seq_len)
    num_blk = SymbolicDim("num_blocks", value=spec.num_kv_blocks)
    max_blk = SymbolicDim("max_blocks", value=(spec.seq_len + spec.page_size - 1) // spec.page_size)
    H_q, H_kv, D = spec.num_heads, spec.num_kv_heads, spec.head_dim

    from tilegen.ir.topology import (
        ExecutionRoles,
        GridMode,
        GridTopology,
        PhaseMode,
        RasterOrder,
        TacticalKnobs,
        TopologySpec,
    )

    topology = TopologySpec(
        grid=GridTopology(GridMode.DATA_PARALLEL, RasterOrder.ROW_MAJOR),
        roles=ExecutionRoles(1, 1, PhaseMode.SERIAL),
        knobs=TacticalKnobs(tile_m=H_q, tile_n=64, tile_k=D, pipeline_stages=2),
    )

    b = VTMBuilder(name, topology)
    b.input("Q", (num_reqs, H_q, D), spec.dtype)
    b.input("K_cache", (num_blk, spec.page_size, H_kv, D), spec.dtype)
    b.input("V_cache", (num_blk, spec.page_size, H_kv, D), spec.dtype)
    b.input("block_tables", (num_reqs, max_blk), "int32")
    b.input("seq_lens", (num_reqs,), "int32")
    b.output("O", (num_reqs, H_q, D), spec.dtype)

    b.layout(
        "kv_paged",
        PagedKVLayout(block_size=spec.page_size, num_kv_heads=H_kv, head_dim=D),
    )

    b.alloc("q_tile", (H_q, D), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("k_tile", (max_seq, D), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("v_tile", (max_seq, D), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("qk", (H_q, max_seq), "fp32", MemoryTier.ACCUMULATOR)
    b.alloc("p", (H_q, max_seq), spec.dtype, MemoryTier.COMPUTE_INPUT)
    b.alloc("acc", (H_q, D), "fp32", MemoryTier.ACCUMULATOR)

    b.gather("K_cache", "k_tile", "block_tables", "seq_lens", "kv_paged")
    b.gather("V_cache", "v_tile", "block_tables", "seq_lens", "kv_paged")
    b.load("Q", "q_tile")
    b.matmul("q_tile", "k_tile", "qk", transpose_b=True, clear=True)
    b.online_softmax("qk", "p", scale=spec.scale or 1.0, is_final=True)
    b.matmul("p", "v_tile", "acc", clear=False)
    b.store("acc", "O")

    return b.build()


def build_dense_attention_vtm(
    spec: AttentionSpec,
    target: TargetSpec,
    name: str = "dense_fa2",
    variant: str = "fa2_prefill",
) -> VTMKernel:
    """Emit the VTM IR for a *dense* FlashAttention-2 forward (no paging).

    This is the prefill pattern the runnable Triton codegen
    (``TritonBackend._lower_dense_fa2`` / ``_lower_fa3``) targets: outer-Q /
    inner-KV online softmax over dense Q/K/V.  ``variant="fa3"`` selects the
    TMA + warp-specialization topology (``GridMode.PERSISTENT_SM`` +
    ``PhaseMode.PING_PONG``), which lowers to TMA descriptor loads.
    """
    H_q, D = spec.num_heads, spec.head_dim
    s_q = SymbolicDim("S_q", value=spec.seq_len)
    topology = attention_topology(variant, target)
    bm, bn = topology.knobs.tile_m, topology.knobs.tile_n

    b = VTMBuilder(name, topology)
    b.input("Q", (spec.batch, H_q, s_q, D), spec.dtype)
    b.input("K", (spec.batch, H_q, s_q, D), spec.dtype)
    b.input("V", (spec.batch, H_q, s_q, D), spec.dtype)
    b.output("O", (spec.batch, H_q, s_q, D), spec.dtype)

    b.alloc("q_tile", (bm, D), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("k_tile", (D, bn), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("v_tile", (bn, D), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("qk", (bm, bn), "fp32", MemoryTier.ACCUMULATOR)
    b.alloc("p", (bm, bn), spec.dtype, MemoryTier.COMPUTE_INPUT)
    b.alloc("acc", (bm, D), "fp32", MemoryTier.ACCUMULATOR)

    b.load("Q", "q_tile")
    b.pipeline_for(
        "kv_blk", "0", "hi", stages=topology.knobs.pipeline_stages,
        body=[
            LoadOp("K", "k_tile"),
            MatmulOp("q_tile", "k_tile", "qk", transpose_b=False, clear=True),
            MaskOp("qk", "CAUSAL" if spec.causal else "NONE"),
            OnlineSoftmaxOp("qk", "p", scale=spec.scale or 1.0, is_final=False),
            LoadOp("V", "v_tile"),
            MatmulOp("p", "v_tile", "acc", clear=False),
        ],
    )
    b.store("acc", "O")
    return b.build()


def build_paged_prefill_vtm(
    spec: AttentionSpec,
    target: TargetSpec,
    name: str = "paged_prefill",
) -> VTMKernel:
    """Emit the VTM IR for a *chunked paged prefill* (Q dense, K/V paged).

    Q is ``[num_reqs, query_len, H_q, D]``; K/V live in a paged cache reached
    via ``GatherOp`` (block-table indirection).  This is the pattern
    ``TritonBackend._lower_paged_prefill`` lowers to a runnable kernel.
    """
    q_len = spec.query_len if spec.query_len > 0 else spec.seq_len
    H_q, H_kv, D = spec.num_heads, spec.num_kv_heads, spec.head_dim
    num_reqs = SymbolicDim("num_reqs", value=spec.batch)
    ql = SymbolicDim("q_len", value=q_len)
    num_blk = SymbolicDim("num_blocks", value=spec.num_kv_blocks)
    max_blk = SymbolicDim(
        "max_blocks", value=(spec.seq_len + spec.page_size - 1) // spec.page_size
    )
    topology = attention_topology("fa2_prefill", target)
    bm = topology.knobs.tile_m

    b = VTMBuilder(name, topology)
    b.input("Q", (num_reqs, ql, H_q, D), spec.dtype)
    b.input("K_cache", (num_blk, spec.page_size, H_kv, D), spec.dtype)
    b.input("V_cache", (num_blk, spec.page_size, H_kv, D), spec.dtype)
    b.input("block_tables", (num_reqs, max_blk), "int32")
    b.input("seq_lens", (num_reqs,), "int32")
    b.output("O", (num_reqs, ql, H_q, D), spec.dtype)

    b.layout(
        "kv_paged",
        PagedKVLayout(block_size=spec.page_size, num_kv_heads=H_kv, head_dim=D),
    )

    b.alloc("q_tile", (bm, H_q, D), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("k_tile", (spec.seq_len, D), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("v_tile", (spec.seq_len, D), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("qk", (bm, spec.seq_len), "fp32", MemoryTier.ACCUMULATOR)
    b.alloc("p", (bm, spec.seq_len), spec.dtype, MemoryTier.COMPUTE_INPUT)
    b.alloc("acc", (bm, D), "fp32", MemoryTier.ACCUMULATOR)

    b.gather("K_cache", "k_tile", "block_tables", "seq_lens", "kv_paged")
    b.gather("V_cache", "v_tile", "block_tables", "seq_lens", "kv_paged")
    b.load("Q", "q_tile")
    b.matmul("q_tile", "k_tile", "qk", transpose_b=True, clear=True)
    b.mask("qk", "CAUSAL" if spec.causal else "NONE")
    b.online_softmax("qk", "p", scale=spec.scale or 1.0, is_final=True)
    b.matmul("p", "v_tile", "acc", clear=False)
    b.store("acc", "O")
    return b.build()


def build_cp_attention_vtm(
    spec: AttentionSpec,
    target: TargetSpec,
    parallel,
    name: str = "cp_dense_fa2",
    variant: str = "fa2_prefill",
) -> VTMKernel:
    """Emit the VTM IR for a *context-parallel* dense attention (Approach A).

    Each rank holds a ``1/cp`` sequence shard of Q (and its K/V).  The IR
    annotates the all-gather of K/V (the visibility step) around the otherwise
    *unchanged* local online-softmax attention; for the ``RING`` comm pattern
    the softmax carries ``global_reduce`` (all-reduce of max/sum).  Execution is
    the host-side :class:`tilegen.parallel.ParallelExecutor`, so the local FA2/
    FA3/FA4 kernels are reused verbatim.
    """
    from tilegen.parallel.spec import CommPattern

    H_q, D = spec.num_heads, spec.head_dim
    cp = max(1, parallel.cp_degree)
    s_q = SymbolicDim("S_q", value=spec.seq_len // cp) # local Q shard
    s_kv = SymbolicDim("S_kv", value=spec.seq_len // cp) # local K/V shard
    topology = attention_topology(variant, target)
    bm, bn = topology.knobs.tile_m, topology.knobs.tile_n

    b = VTMBuilder(name, topology)
    b.input("Q", (spec.batch, H_q, s_q, D), spec.dtype)
    b.input("K", (spec.batch, H_q, s_kv, D), spec.dtype)
    b.input("V", (spec.batch, H_q, s_kv, D), spec.dtype)
    b.output("O", (spec.batch, H_q, s_q, D), spec.dtype)

    b.alloc("q_tile", (bm, D), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("k_tile", (D, bn), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("v_tile", (bn, D), spec.dtype, MemoryTier.SCRATCHPAD)
    b.alloc("qk", (bm, bn), "fp32", MemoryTier.ACCUMULATOR)
    b.alloc("p", (bm, bn), spec.dtype, MemoryTier.COMPUTE_INPUT)
    b.alloc("acc", (bm, D), "fp32", MemoryTier.ACCUMULATOR)

    # Approach A: gather the sequence-sharded K/V before the local attention.
    b.all_gather("K", dim=2, degree=cp)
    b.all_gather("V", dim=2, degree=cp)

    global_reduce = parallel.comm_pattern == CommPattern.RING
    b.load("Q", "q_tile")
    b.pipeline_for(
        "kv_blk", "0", "hi", stages=topology.knobs.pipeline_stages,
        body=[
            LoadOp("K", "k_tile"),
            MatmulOp("q_tile", "k_tile", "qk", transpose_b=False, clear=True),
            MaskOp("qk", "CAUSAL" if spec.causal else "NONE"),
            OnlineSoftmaxOp("qk", "p", scale=spec.scale or 1.0, is_final=False, global_reduce=global_reduce),
            LoadOp("V", "v_tile"),
            MatmulOp("p", "v_tile", "acc", clear=False),
        ],
    )
    b.online_softmax("qk", "acc", scale=spec.scale or 1.0, is_final=True, global_reduce=global_reduce)
    b.store("acc", "O")
    return b.build()


__all__ = [
    "AttentionSpec",
    "build_attention_vtm",
    "build_paged_attention_vtm",
    "build_decode_vtm",
    "build_dense_attention_vtm",
    "build_paged_prefill_vtm",
    "build_cp_attention_vtm",
]

