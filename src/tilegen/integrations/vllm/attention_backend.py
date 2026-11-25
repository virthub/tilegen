# src/tilegen/integrations/vllm/attention_backend.py
#
# vLLM OOT TileGenAttentionBackend

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import register_backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec


@register_backend("TILEGEN")
class TileGenAttentionBackend(AttentionBackend):
    """Attention backend dispatching to TileGen's FA2 Triton kernel.

    Plugs into vLLM 0.34's **Model Runner V2 (MRV2)** execution engine: MRV2
    selects this backend via ``VLLM_ATTENTION_BACKEND=TILEGEN``, builds
    ``CommonAttentionMetadata`` from the scheduler, calls
    :meth:`TileGenAttentionMetadataBuilder.build`, and invokes
    :meth:`TileGenAttentionImpl.forward` on the hot path.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16", "bfloat16"]

    # MRV2 writes the KV cache separately (``torch.ops.vllm.unified_kv_cache_update``)
    # before calling ``forward``; this backend only *reads* the cache during decode.
    # (Matches the flash_attn / triton_attn / flashinfer convention.)
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "TILEGEN"

    @staticmethod
    def get_impl_cls() -> type["TileGenAttentionImpl"]:
        return TileGenAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TileGenAttentionMetadataBuilder"]:
        return TileGenAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        del kv_cache_spec
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128]

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True


@dataclass
class TileGenAttentionMetadata(AttentionMetadata):
    """Per-layer attention metadata for the TileGen backend."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True
    num_decode_tokens: int = 0


class TileGenAttentionMetadataBuilder(
    AttentionMetadataBuilder[TileGenAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TileGenAttentionMetadata:
        del common_prefix_len, fast_build
        num_reqs = common_attn_metadata.num_reqs
        causal = common_attn_metadata.causal
        is_dynamic_causal = isinstance(causal, torch.Tensor)
        causal_bool = True if is_dynamic_causal else bool(causal)

        return TileGenAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=causal_bool,
            num_decode_tokens=int(
                common_attn_metadata.query_start_loc_cpu[-1].item()
            )
            - num_reqs,
        )


class TileGenAttentionImpl(AttentionImpl):
    """Dense (unpaged) attention routed to TileGen's FA2 Triton kernel."""

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

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TileGenAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, output_scale, output_block_scale
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        if attn_metadata.num_decode_tokens > 0:
            return self._forward_decode(query, kv_cache, attn_metadata, output)

        return self._forward_prefill(query, key, value, attn_metadata)

    def _forward_decode(self, query, kv_cache, attn_metadata, output) -> torch.Tensor:
        """Paged-KV decode: batched single-launch gather + attention."""
        num_reqs = attn_metadata.seq_lens.shape[0]
        block_size = kv_cache.shape[2]
        hs = self.head_size

        # vLLM cache layout: [num_blocks, num_kv_heads, block_size, 2*hs].
        # Split into K/V and permute to [num_blocks, block_size, num_kv_heads, hs].
        kc = kv_cache[:, :, :, :hs].permute(0, 2, 1, 3).contiguous()
        vc = kv_cache[:, :, :, hs:].permute(0, 2, 1, 3).contiguous()

        # Fast path: pure-decode batch -> one Triton launch for all requests.
        if query.shape[0] == num_reqs:
            from tilegen.attention import paged_decode
            from tilegen.layout.layout import PagedKVLayout

            layout = PagedKVLayout(
                block_size=block_size, num_kv_heads=self.num_kv_heads, head_dim=hs
            )
            out = paged_decode(
                query, kc, vc,
                attn_metadata.block_table.to(torch.int32),
                attn_metadata.seq_lens.to(torch.int32),
                layout, sm_scale=self.scale,
            )
            output.copy_(out)
            return output

        # Fallback: per-request loop (mixed prefill+decode batches).
        import torch.nn.functional as F

        ratio = self.num_heads // self.num_kv_heads
        for r in range(num_reqs):
            seq = int(attn_metadata.seq_lens[r].item())
            nblocks = (seq + block_size - 1) // block_size
            blocks = attn_metadata.block_table[r, :nblocks].long()
            k = kc[blocks].reshape(nblocks * block_size, self.num_kv_heads, hs)[:seq]
            v = vc[blocks].reshape(nblocks * block_size, self.num_kv_heads, hs)[:seq]
            if ratio > 1:
                k = k.repeat_interleave(ratio, dim=1)
                v = v.repeat_interleave(ratio, dim=1)
            k = k.permute(1, 0, 2).unsqueeze(0)          # [1, H, seq, D]
            v = v.permute(1, 0, 2).unsqueeze(0)
            qr = query[r].unsqueeze(0).unsqueeze(2)      # [1, H, 1, D]
            out_r = F.scaled_dot_product_attention(qr, k, v, is_causal=False, scale=self.scale)
            output[r] = out_r.squeeze(0).squeeze(1)
        return output

    def _forward_prefill(self, query, key, value, attn_metadata) -> torch.Tensor:
        """Dense prefill attention (key/value carry the full prompt)."""
        import torch.nn.functional as F

        q4 = query.transpose(0, 1).unsqueeze(0)
        k4 = key.transpose(0, 1).unsqueeze(0)
        v4 = value.transpose(0, 1).unsqueeze(0)

        if self.num_kv_heads != self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            k4 = k4.repeat_interleave(ratio, dim=1)
            v4 = v4.repeat_interleave(ratio, dim=1)

        if query.dtype != torch.bfloat16:
            out = F.scaled_dot_product_attention(
                q4, k4, v4, is_causal=attn_metadata.causal, scale=self.scale
            )
            return out.squeeze(0).transpose(0, 1)

        from tilegen.attention import dense_attention

        out = dense_attention(q4, k4, v4, is_causal=attn_metadata.causal, scale=self.scale)
        return out.squeeze(0).transpose(0, 1)


__all__ = [
    "TileGenAttentionBackend",
    "TileGenAttentionMetadata",
    "TileGenAttentionMetadataBuilder",
    "TileGenAttentionImpl",
]
