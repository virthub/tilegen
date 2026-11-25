# src/tilegen/integrations/vllm_pspkv/attention_backend.py
#
# TILEGEN_PSPKV: TileGen attention backend for the vllm-pspkv fork (vLLM
# 0.34.0 + PSP-KV precision-scalable paged KV cache).  Registered as an OOT
# backend via the vLLM V1 registry; selected with
# ``VLLM_ATTENTION_BACKEND=TILEGEN_PSPKV``.
#
# The backend consumes the fork's PSP-KV byte cache unchanged: per-block inner
# precision (precision_table), per-block K/V byte offsets (block_offsets) and
# per-group scales (scale_table) are read from CommonAttentionMetadata, and the
# cached prefix is dequantized with a self-contained torch implementation
# covering every PSP-KV format (lossless / e4m3 / e5m2 / fp4 / int8) before
# TileGen/native SDPA attention runs on the dense K/V.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import torch
from vllm import envs
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import register_backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

# vllm-pspkv is fork-only; every fork-specific import is lazy so the module can
# be imported (and registered) in upstream vLLM 0.34 as well.


def _fork_module():
    from vllm.v1.attention.backends import psp_kv_attn

    return psp_kv_attn


def _psp_kv_block_size() -> int:
    return int(envs.PSP_KV_BLOCK_SIZE)


_FP4_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, float("inf"), float("inf"),
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, float("-inf"), float("-inf")]
_FP4_BYTE_TABLE: Optional[torch.Tensor] = None


def _fp4_byte_table(device: torch.device) -> torch.Tensor:
    global _FP4_BYTE_TABLE
    if _FP4_BYTE_TABLE is None or _FP4_BYTE_TABLE.device != device:
        t = torch.tensor(_FP4_E2M1, dtype=torch.float32, device=device)
        low = t.repeat(16)               # low[b]  = t[b & 0xF]
        high = t.repeat_interleave(16)   # high[b] = t[b >> 4]
        _FP4_BYTE_TABLE = torch.stack([low, high], dim=1).contiguous()
    return _FP4_BYTE_TABLE


def _fmt_k(p: int) -> int:
    return 4 if p == 4 else ((p & 3) if p >= 4 else p)


def _fmt_v(p: int) -> int:
    return 4 if p == 4 else ((p >> 2) if p >= 4 else p)


def _dequant_fp8(q: torch.Tensor, fmt: int) -> torch.Tensor:
    """Bit-exact e4m3/e5m2 dequant (uint8 -> float32)."""
    q = q.to(torch.int32)
    sign = (q >> 7) & 1
    if fmt == 2:  # e5m2
        exp = (q >> 2) & 0x1F
        mant = q & 0x3
        normal = (1.0 + mant.float() / 4.0) * torch.pow(2.0, exp.float() - 15.0)
        subnormal = mant.float() * (2.0 ** -16.0)
        special = torch.where(mant == 0, float("inf"), float("nan"))
        max_exp = 31
    else:  # e4m3
        exp = (q >> 3) & 0xF
        mant = q & 0x7
        normal = (1.0 + mant.float() / 8.0) * torch.pow(2.0, exp.float() - 7.0)
        subnormal = mant.float() * (2.0 ** -9.0)
        special = torch.where(mant == 7, float("nan"), normal)
        max_exp = 15
    val = torch.where(exp == 0, subnormal, normal)
    val = torch.where(exp == max_exp, special, val)
    return torch.where(sign == 1, -val, val)


def _scale_row(scale_table, row: int, num_kv_heads: int, head_size: int):
    """Broadcast one scale-table row to per-head scales for [H, B, D] blocks."""
    if scale_table is None:
        return 1.0
    s = scale_table[row].float()
    if s.numel() == num_kv_heads:
        return s[:, None, None]  # per-head row [H]
    s = s.reshape(num_kv_heads, -1)  # [H, G] or [H, head]
    if s.shape[1] == head_size:
        return s[:, None, :]  # per-element row
    groups = s.shape[1]
    return s.repeat_interleave(head_size // groups, dim=1)[:, None, :]


def _dequant_block(cache, off: int, fmt: int, num_kv_heads: int,
                   block_size: int, head_size: int, scalar_bytes: int) -> torch.Tensor:
    """Dequantize one block's K or V into [num_kv_heads, block_size, head_size].

    The byte buffer is token-major ([t][h][d]), so the flat block view reshapes
    as (block_size, num_kv_heads, head_size) and is then permuted.
    """
    if fmt == 0:
        nbytes = block_size * head_size * scalar_bytes * num_kv_heads
        return cache[off:off + nbytes].view(torch.float16).reshape(
            block_size, num_kv_heads, head_size).permute(1, 0, 2).float()
    if fmt == 3:
        nbytes = (block_size * head_size // 2) * num_kv_heads
        buf = cache[off:off + nbytes]
        return _fp4_byte_table(cache.device)[buf.to(torch.int64)].reshape(
            block_size, num_kv_heads, head_size).permute(1, 0, 2)
    if fmt == 4:
        nbytes = block_size * head_size * num_kv_heads
        return cache[off:off + nbytes].to(torch.int8).float().reshape(
            block_size, num_kv_heads, head_size).permute(1, 0, 2)
    nbytes = block_size * head_size * num_kv_heads
    return _dequant_fp8(cache[off:off + nbytes], fmt).reshape(
        block_size, num_kv_heads, head_size).permute(1, 0, 2)


def _dequant_prefix_torch(k_cache, v_cache, block_tables, precision_table,
                          scale_table, block_offsets, seq_idx, context_len,
                          num_kv_heads, head_size, block_size):
    """Dequantize one request's cached prefix into [num_kv_heads, ctx, head_size]."""
    n_blk = (context_len + block_size - 1) // block_size
    bids = block_tables[seq_idx, :n_blk]
    num_blocks = precision_table.numel()
    scalar_bytes = 2  # lossless storage is fp16/bf16
    k_out = torch.empty(n_blk, num_kv_heads, block_size, head_size,
                        dtype=torch.float32, device=k_cache.device)
    v_out = torch.empty_like(k_out)
    for i, bid in enumerate(bids.tolist()):
        p = int(precision_table[bid].item())
        k_off = int(block_offsets[bid].item())
        v_off = int(block_offsets[num_blocks + bid].item())
        k_out[i] = _dequant_block(k_cache, k_off, _fmt_k(p), num_kv_heads,
                                  block_size, head_size, scalar_bytes)
        v_out[i] = _dequant_block(v_cache, v_off, _fmt_v(p), num_kv_heads,
                                  block_size, head_size, scalar_bytes)
        k_out[i] = k_out[i] * _scale_row(scale_table, bid, num_kv_heads, head_size)
        v_out[i] = v_out[i] * _scale_row(scale_table, num_blocks + bid,
                                         num_kv_heads, head_size)
    k = k_out.permute(1, 0, 2, 3).reshape(
        num_kv_heads, n_blk * block_size, head_size)[:, :context_len].contiguous()
    v = v_out.permute(1, 0, 2, 3).reshape(
        num_kv_heads, n_blk * block_size, head_size)[:, :context_len].contiguous()
    return k, v


@dataclass
class TileGenPSPKVAttentionMetadata(AttentionMetadata):
    """Per-layer metadata for the TILEGEN_PSPKV backend.

    Wraps the common attention metadata, which carries the PSP-KV precision
    tables, scale tables, block offsets and block tables.
    """

    common_metadata: CommonAttentionMetadata


class TileGenPSPKVAttentionMetadataBuilder(
    AttentionMetadataBuilder[TileGenPSPKVAttentionMetadata]
):
    """Builder mirroring the fork's PSP_KV builder (single-token decode)."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "TILEGEN_PSPKV does not support speculative decoding: decode "
                "attends one query token per request."
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> TileGenPSPKVAttentionMetadata:
        del common_prefix_len, fast_build, kwargs
        return TileGenPSPKVAttentionMetadata(common_metadata=common_attn_metadata)


class TileGenPSPKVAttentionImpl(AttentionImpl[TileGenPSPKVAttentionMetadata]):
    """TileGen attention over the PSP-KV byte cache.

    Decode: current-token write-back (fork op) + single-launch Triton decode
    (fused per-block dequant + attention, all five PSP-KV formats) with a
    per-request torch fallback for mixed-precision batches.  Prefill: delegated
    to the fork's implementation (scratch + varlen flash-attn + prefix dequant)
    so the cache population is identical to PSP_KV's.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        del alibi_slopes, sliding_window, kv_cache_dtype, logits_soft_cap
        del attn_type, kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self._p0_key: tuple | None = None
        self._p0: int | None = None
        self._p0_homogeneous: bool = False
        self._fork_prefill_impl = None
        self._psp_ops = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TileGenPSPKVAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del output_scale, output_block_scale, kwargs
        if attn_metadata is None:
            # Profiling run: no metadata yet.
            return output.fill_(0)

        # Ensure the fork's PSP-KV extension (torch.ops.psp_kv.*) is loaded
        # before any runner-side quantize calls — mirrors PSP_KV's forward,
        # which locates the op namespace before prefill dispatch.
        if getattr(self, "_psp_ops", None) is None:
            self._psp_ops = _fork_module()._get_psp_kv_ops()

        query = query.contiguous()
        meta = attn_metadata.common_metadata

        _timing = os.environ.get("TILEGEN_PSPKV_TIME")
        if _timing:
            import time as _time_mod

            torch.cuda.synchronize()
            _t0 = _time_mod.perf_counter()

        num_actual = getattr(meta, "num_actual_tokens", None)
        num_reqs = getattr(meta, "num_reqs", None)
        is_prefill = (
            not torch.cuda.is_current_stream_capturing()
            and num_actual is not None
            and num_reqs is not None
            and num_actual > num_reqs
        )
        if is_prefill:
            res = self._forward_prefill(query, key, value, kv_cache, meta, output, layer)
        else:
            res = self._forward_decode(query, key, value, kv_cache, meta, output, layer)
        if _timing:
            torch.cuda.synchronize()
            print(f"[tilegen-decode] {'prefill' if is_prefill else 'decode'} step "
                  f"{(_time_mod.perf_counter() - _t0) * 1e3:.2f} ms", flush=True)
        return res

    def _resolve_tables(self, meta, layer):
        fork = _fork_module()
        precision_table = getattr(meta, "precision_table", None)
        if precision_table is None:
            raise RuntimeError(
                "TILEGEN_PSPKV requires CommonAttentionMetadata.precision_table; "
                "run inside the vllm-pspkv fork (MRV2) with PSPKV_FORMAT=1."
            )
        scale_table = getattr(meta, "scale_table", None)
        if scale_table is not None and scale_table.dim() == 4:
            from vllm.model_executor.models.utils import extract_layer_index

            layer_name = getattr(layer, "layer_name", None)
            layer_idx = extract_layer_index(layer_name) if layer_name is not None else 0
            scale_table = scale_table[layer_idx]
        block_offsets = getattr(meta, "block_offsets", None)
        block_tables = getattr(meta, "block_tables", None)
        if block_tables is None:
            block_tables = meta.block_table_tensor
        context_lens = getattr(meta, "context_lens", None)
        if context_lens is None:
            context_lens = meta.seq_lens
        return fork, precision_table, scale_table, block_offsets, block_tables, context_lens

    def _forward_decode(self, query, key, value, kv_cache, meta, output, layer) -> torch.Tensor:
        import torch.nn.functional as F

        fork, precision_table, scale_table, block_offsets, block_tables, context_lens = (
            self._resolve_tables(meta, layer)
        )

        num_kv_heads = getattr(layer, "num_kv_heads_per_partition", None)
        if num_kv_heads is None:
            num_kv_heads = layer.num_kv_heads
        head_size = self.head_size
        block_size = _psp_kv_block_size()
        ratio = self.num_heads // num_kv_heads
        num_reqs = context_lens.shape[0]

        k_cache, v_cache = fork.psp_kv_flat_byte_caches(kv_cache)
        assert context_lens.dtype == torch.int32 and context_lens.is_contiguous()
        assert block_tables.is_contiguous()

        # Decode write-back (mirrors PSP_KV): quantize the current token into
        # its cache slot before attention, so context_lens tokens are attended.
        slot_mapping = getattr(meta, "slot_mapping", None)
        if slot_mapping is not None and key is not None and value is not None \
                and scale_table is not None and block_offsets is not None:
            fork._get_psp_kv_ops().quantize_single_token(
                key.contiguous(), value.contiguous(), k_cache, v_cache,
                slot_mapping, precision_table, block_offsets, scale_table,
                getattr(meta, "format_generation", 1),
            )

        # Batched Triton fast path: homogeneous precision, one launch for all
        # requests (fused per-block dequant + attention).  Mixed-precision
        # batches fall through to the per-request torch dequant loop.  The
        # homogeneity check is cached on the precision table's version counter
        # to avoid a GPU->CPU sync on every decode step.
        p_key = (precision_table.data_ptr(), precision_table._version)
        if getattr(self, "_p0_key", None) == p_key:
            homogeneous = self._p0_homogeneous
            p0 = self._p0
        else:
            prec = precision_table.flatten()
            p0 = int(prec[0].item())
            homogeneous = bool((prec == p0).all().item())
            self._p0_key = p_key
            self._p0 = p0
            self._p0_homogeneous = homogeneous
        if homogeneous and scale_table is not None and block_offsets is not None:
            try:
                from tilegen.layout.layout import PagedKVLayout
                from tilegen.ops.pspkv_decode import pspkv_decode_forward

                _dbg = os.environ.get("TILEGEN_PSPKV_DEBUG")
                if _dbg:
                    import time as _tm

                    torch.cuda.synchronize()
                    _d0 = _tm.perf_counter()
                layout = PagedKVLayout(
                    block_size=block_size, num_kv_heads=num_kv_heads, head_dim=head_size
                )
                out = pspkv_decode_forward(
                    query, k_cache, block_tables, context_lens,
                    precision_table, block_offsets, scale_table, layout,
                    _fmt_k(p0), _fmt_v(p0), sm_scale=self.scale,
                )
                if _dbg:
                    torch.cuda.synchronize()
                    print(f"[tilegen-decode] kernel {(_tm.perf_counter() - _d0) * 1e3:.2f} ms "
                          f"(B={num_reqs} maxblk={block_tables.shape[1]} Hkv={num_kv_heads} "
                          f"D={head_size} R={ratio})", flush=True)
                output.copy_(out)
                return output
            except Exception:
                if os.environ.get("TILEGEN_PSPKV_DEBUG"):
                    import traceback

                    traceback.print_exc()
                pass  # fall through to the reference loop

        for r in range(num_reqs):
            seq = int(context_lens[r].item())
            if seq <= 0:
                output[r] = 0.0
                continue
            k, v = _dequant_prefix_torch(
                k_cache, v_cache, block_tables, precision_table, scale_table,
                block_offsets, r, seq, num_kv_heads, head_size, block_size,
            )
            # k/v: [num_kv_heads, seq, head_size] -> [1, num_kv_heads, seq, D]
            k4 = k.unsqueeze(0).to(query.dtype)
            v4 = v.unsqueeze(0).to(query.dtype)
            if ratio > 1:
                k4 = k4.repeat_interleave(ratio, dim=1)
                v4 = v4.repeat_interleave(ratio, dim=1)
            q4 = query[r].unsqueeze(0).unsqueeze(2)  # [1, H, 1, D]
            out_r = F.scaled_dot_product_attention(
                q4, k4, v4, is_causal=False, scale=self.scale
            )
            output[r] = out_r.squeeze(0).squeeze(1)
        return output

    def _forward_prefill(self, query, key, value, kv_cache, meta, output, layer) -> torch.Tensor:
        """Fork-equivalent prefill (scratch write + varlen flash-attn + prefix
        dequant), delegated to the fork's own implementation so the byte cache
        is populated exactly as PSP_KV expects; the runner's post-forward
        quantize_and_scatter then writes the scratch into the cache.
        """
        fork = _fork_module()
        if getattr(self, "_fork_prefill_impl", None) is None:
            self._fork_prefill_impl = fork.PSPKVAttentionImpl(
                num_heads=self.num_heads, head_size=self.head_size,
                scale=self.scale, num_kv_heads=self.num_kv_heads,
                alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
            )
        impl = self._fork_prefill_impl
        k_cache, v_cache = fork.psp_kv_flat_byte_caches(kv_cache)
        block_tables = getattr(meta, "block_tables", None)
        if block_tables is None:
            block_tables = meta.block_table_tensor
        block_offsets = getattr(meta, "block_offsets", None)
        num_kv_heads = getattr(layer, "num_kv_heads_per_partition", None)
        if num_kv_heads is None:
            num_kv_heads = layer.num_kv_heads
        return impl._prefill_forward(
            query, key, value, meta, output, layer, k_cache, v_cache,
            block_tables, block_offsets, num_kv_heads, self.head_size,
        )


@register_backend("TILEGEN_PSPKV")
class TileGenPSPKVAttentionBackend(AttentionBackend):
    """TileGen backend for the vllm-pspkv fork (PSP-KV byte cache).

    Mirrors the fork's PSP_KV backend for cache geometry (1-D byte buffer,
    block size 128, ``block_offsets`` [K | V]) so the runner allocates the
    identical cache; only the attention implementation differs (TileGen/native
    SDPA over a dequantized prefix).
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "fp8"]

    # The backend performs its own PSP-KV cache write-back (mirrors PSP_KV).
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "TILEGEN_PSPKV"

    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        from vllm.v1.kv_cache_interface import psp_kv_spec_from_attention_spec

        return psp_kv_spec_from_attention_spec(spec) or spec

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128]

    @classmethod
    def get_supported_kernel_block_sizes(
        cls, kv_cache_spec: "AttentionSpec | None" = None
    ) -> list[int]:
        del kv_cache_spec
        return [_psp_kv_block_size()]

    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        del block_size, num_kv_heads, head_size, cache_dtype_str
        return 0

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type[TileGenPSPKVAttentionImpl]:
        return TileGenPSPKVAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TileGenPSPKVAttentionMetadataBuilder]:
        return TileGenPSPKVAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
        dtype: torch.dtype = torch.float16,
    ) -> tuple[int, ...]:
        """1-D byte buffer matching the fork's PSP_KV cache geometry."""
        from vllm.v1.kv_cache_interface import _psp_kv_avg_bytes

        per_elem = _psp_kv_avg_bytes(dtype)
        per_kv_bytes = int(num_blocks * block_size * num_kv_heads * head_size * per_elem)
        return (2 * per_kv_bytes,)


__all__ = [
    "TileGenPSPKVAttentionBackend",
    "TileGenPSPKVAttentionMetadata",
    "TileGenPSPKVAttentionMetadataBuilder",
    "TileGenPSPKVAttentionImpl",
]
