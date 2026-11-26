# src/tilegen/integrations/sglang/attention_backend.py
#
# SGLang attention backend

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend as _Base
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
else:
    _Base = object # type: ignore[assignment]


def _kv_cache(kv_cache) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (k, v) from an SGLang ``layer.kv_cache`` pool.

    SGLang stores K/V interleaved on dim 1 of a flat token pool
    ``[num_tokens, 2, num_kv_heads, head_dim]`` (``k = [:, 0]``, ``v = [:, 1]``).
    Also accepts a ``.k_cache``/``.v_cache`` object for older layouts.
    """
    if hasattr(kv_cache, "k_cache") and hasattr(kv_cache, "v_cache"):
        return kv_cache.k_cache, kv_cache.v_cache
    return kv_cache[:, 0].contiguous(), kv_cache[:, 1].contiguous()


def _req_to_token(forward_batch) -> torch.Tensor:
    """Return the ``[num_reqs, max_context]`` token-index table for a batch."""
    if hasattr(forward_batch, "req_to_token_pool"):
        return forward_batch.req_to_token_pool.req_to_token
    return forward_batch.req_to_token


class TileGenAttentionBackend(_Base):
    """SGLang attention backend dispatching to TileGen's FA kernels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
# TileGen kernels read seq_lens / req_pool_indices / req_to_token
        # directly from the batch; no wrapper state needed.
        return None

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        del k, v, save_kv_cache, kwargs
        from tilegen.ops.paged_decode import paged_decode_tokens_forward

        k_pool, v_pool = _kv_cache(layer.kv_cache)
        req_to_token = _req_to_token(forward_batch)
        req_idx = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens

# One batched launch: gather each request's token-indexed context and
# attend (SGLang radix-tree KV, [num_tokens, nkv, D] pool).
        rtt = req_to_token[req_idx.long()] # [num_reqs, max_ctx]
        return paged_decode_tokens_forward(
            q, k_pool, v_pool, rtt, seq_lens, sm_scale=layer.scaling
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        del save_kv_cache, kwargs
        from tilegen.attention import dense_attention

        k_pool, v_pool = _kv_cache(layer.kv_cache)
        req_to_token = _req_to_token(forward_batch)
        req_idx = forward_batch.req_pool_indices
        prefix_lens = forward_batch.extend_prefix_lens
        seq_lens = forward_batch.extend_seq_lens
        num_heads = q.shape[1]
        num_kv_heads = k_pool.shape[1]
        ratio = num_heads // num_kv_heads

# Extend with cached prefix: each request's new tokens attend to the
        # cached prefix (token-indexed, radix tree) + the new tokens (causal).
        o: list[torch.Tensor] = []
        start = 0
        for i in range(len(seq_lens)):
            plen = int(prefix_lens[i].item())
            slen = int(seq_lens[i].item())
            if plen > 0:
                toks = req_to_token[int(req_idx[i].item()), :plen].long()
                k_pre = k_pool[toks] # [plen, nkv, dim]
                v_pre = v_pool[toks]
            else:
                k_pre = k_pool[:0]
                v_pre = v_pool[:0]
            k_new = k[start:start + slen] # [slen, nkv, dim]
            v_new = v[start:start + slen]
            k_cat = torch.cat([k_pre, k_new], dim=0) # [plen+slen, nkv, dim]
            v_cat = torch.cat([v_pre, v_new], dim=0)
            if ratio > 1:
                k_cat = k_cat.repeat_interleave(ratio, dim=1)
                v_cat = v_cat.repeat_interleave(ratio, dim=1)
            # [1, H, slen, D] @ [1, H, plen+slen, D] with prefix-causal masking.
            q4 = q[start:start + slen].transpose(0, 1).unsqueeze(0)
            k4 = k_cat.transpose(0, 1).unsqueeze(0)
            v4 = v_cat.transpose(0, 1).unsqueeze(0)
            out = dense_attention(q4, k4, v4, is_causal=True, scale=layer.scaling)
            o.append(out.squeeze(0).transpose(0, 1)) # [slen, H, dim]
            start += slen
        return torch.cat(o, dim=0) if o else q.new_empty((0, num_heads, q.shape[-1]))

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        return self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)


__all__ = ["TileGenAttentionBackend"]
