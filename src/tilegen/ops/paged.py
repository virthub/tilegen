# src/tilegen/ops/paged.py
#
# Paged-KV attention

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional
from tilegen.layout.layout import PagedKVLayout
from tilegen.layout.quant import QuantSpec
from tilegen.ops.quant_int4 import dequant_int4_kv as _dequant_int4_kv


def paged_attention_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_layout: PagedKVLayout,
    quant: Optional[QuantSpec] = None,
    k_scales: Optional[torch.Tensor] = None,
    v_scales: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Paged-KV attention for decode (``q`` has 1 token per sequence).

    Args:
        q: ``[B, H_q, 1, D]`` (decode) query tensor.
        k_cache/v_cache: ``[num_blocks, block_size, H_kv, D]`` (fp16/bf16) or
            ``[num_blocks, block_size, H_kv, D//2] uint8`` when ``quant`` is INT4.
        block_tables: ``[B, max_blocks]`` int32 block indices per sequence.
        seq_lens: ``[B]`` context length per sequence.
        kv_layout: :class:`PagedKVLayout` descriptor (block_size, heads, head_dim).
        quant: optional :class:`QuantSpec` (INT4) describing packed KV.
        k_scales/v_scales: optional per-block scales for INT4 KV.

    Returns:
        ``[B, H_q, 1, D]`` attention output.
    """
    B, H_q, _, D = q.shape
    block_size = kv_layout.block_size
    H_kv = kv_layout.num_kv_heads
    ratio = H_q // H_kv # GQA repetition factor
    use_int4 = quant is not None and quant.is_sub_byte

    # Fast path: single-launch batched decode kernel (fp16 K/V, no dequant).
    if not use_int4 and k_cache.dtype == torch.float16:
        from tilegen.ops.paged_decode import paged_decode_forward

        out = paged_decode_forward(
            q.squeeze(2), k_cache, v_cache, block_tables, seq_lens,
            kv_layout, sm_scale=sm_scale,
        )
        return out.unsqueeze(2)

    outs = []
    for b in range(B):
        seq = int(seq_lens[b].item())
        nblocks = (seq + block_size - 1) // block_size
        blocks = block_tables[b, :nblocks].long()

        k = k_cache[blocks].reshape(nblocks * block_size, H_kv, -1)[:seq]
        v = v_cache[blocks].reshape(nblocks * block_size, H_kv, -1)[:seq]
        if use_int4:
            k = _dequant_int4_kv(k, k_scales[:seq])
            v = _dequant_int4_kv(v, v_scales[:seq])

        if ratio > 1:
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)

        # [seq, H_q, D] -> [1, H_q, seq, D]
        k = k.permute(1, 0, 2).unsqueeze(0)
        v = v.permute(1, 0, 2).unsqueeze(0)
        qb = q[b].unsqueeze(0) # [1, H_q, 1, D]

        # Decode attention: 1 query token against `seq` keys (no causal mask).
        out = F.scaled_dot_product_attention(qb, k, v, is_causal=False, scale=sm_scale)
        outs.append(out) # [1, H_q, 1, D]

    return torch.cat(outs, dim=0)


__all__ = ["paged_attention_forward"]
