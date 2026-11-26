# src/tilegen/ops/pspkv_decode.py
#
# Batched Triton decode over the PSP-KV byte cache: a single launch, one
# program per request, with per-block dequantization fused into attention
# (lossless fp16 / e4m3 / e5m2 / fp4 / int8). Homogeneous-precision fast
# path; mixed per-block precision falls back to the per-request torch dequant
# path in the TILEGEN_PSPKV backend.

from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _load_dequant(cache, base, t, d, H: tl.constexpr, D: tl.constexpr,
                  FMT: tl.constexpr, h: tl.constexpr):
    """Dequantize one head's [BLOCK, D] tile at byte base ``base``.

    Pure bit arithmetic (no lookup tables): staging LUT gathers for the fp8/
    fp4 paths exceeded the shared-memory limit on sm_120 once the padded
    tensor-core dot was added.
    """
    if FMT == 0: # lossless fp16, 2 bytes/elem little-endian
        head_bytes: tl.constexpr = D * 2
        off = base + t[:, None] * (H * head_bytes) + h * head_bytes + d[None, :] * 2
        lo = tl.load(cache + off).to(tl.uint16)
        hi = tl.load(cache + off + 1).to(tl.uint16)
        x = ((hi << 8) | lo).to(tl.float16, bitcast=True).to(tl.float32)
    elif FMT == 3: # fp4 e2m1 packed nibbles: 1 sign, 2 exp, 1 mant
        head_bytes: tl.constexpr = D // 2
        off = base + t[:, None] * (H * head_bytes) + h * head_bytes + (d[None, :] // 2)
        b = tl.load(cache + off).to(tl.int32)
        nb = (b >> (4 * (d[None, :] % 2))) & 0xF
        exp = (nb & 0x7) >> 1
        mant = nb & 1
        val = tl.exp2(exp.to(tl.float32) - 1.0) * (1.0 + 0.5 * mant.to(tl.float32))
        val = tl.where(exp == 0, 0.5 * mant.to(tl.float32), val)
        x = tl.where(exp == 3, float("inf"), val)
        x = tl.where((nb >> 3) & 1 == 1, -x, x)
    elif FMT == 4: # int8 (two's complement)
        head_bytes: tl.constexpr = D
        off = base + t[:, None] * (H * head_bytes) + h * head_bytes + d[None, :]
        x = tl.load(cache + off).to(tl.int8).to(tl.float32)
    else: # fp8 e4m3 (1) / e5m2 (2)
        head_bytes: tl.constexpr = D
        off = base + t[:, None] * (H * head_bytes) + h * head_bytes + d[None, :]
        b = tl.load(cache + off).to(tl.int32)
        sign = ((b >> 7) & 1).to(tl.float32)
        if FMT == 1: # e4m3: 1 sign, 4 exp, 3 mant
            exp = (b >> 3) & 0xF
            mant = b & 0x7
            normal = (1.0 + mant.to(tl.float32) / 8.0) * tl.exp2(exp.to(tl.float32) - 7.0)
            subn = mant.to(tl.float32) * tl.exp2(-9.0)
            val = tl.where(exp == 0, subn, normal)
            val = tl.where(exp == 15, tl.where(mant == 7, float("nan"), float("inf")), val)
        else: # e5m2: 1 sign, 5 exp, 2 mant
            exp = (b >> 2) & 0x1F
            mant = b & 0x3
            normal = (1.0 + mant.to(tl.float32) / 4.0) * tl.exp2(exp.to(tl.float32) - 15.0)
            subn = mant.to(tl.float32) * tl.exp2(-16.0)
            val = tl.where(exp == 0, subn, normal)
            val = tl.where(exp == 31, tl.where(mant == 0, float("inf"), float("nan")), val)
        x = tl.where(sign == 1.0, -val, val)
    return x


@triton.jit
def _pspkv_decode_kernel(
    Q, CACHE, BLOCK_TABLES, SEQ_LENS, BLOCK_OFFSETS, SCALES, O, sm_scale,
    NUM_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr, MAX_BLOCKS: tl.constexpr,
    K_FMT: tl.constexpr, V_FMT: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    GROUPS: tl.constexpr,
):
# One program per (request, kv-head): the head loop is parallelized across
    # programs so small decode batches still occupy the GPU.
    pid = tl.program_id(0)
    r = pid // NUM_KV_HEADS
    hkv = pid % NUM_KV_HEADS
    seq = tl.load(SEQ_LENS + r)
    nblk = (seq + BLOCK - 1) // BLOCK

# Rows per program, padded to >= 16 so the tensor-core dot path is always
    # used (masked rows contribute nothing to the softmax/PV).
    RP: tl.constexpr = 16 if RATIO < 16 else RATIO
    offs_r = tl.arange(0, RP)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, BLOCK)

    q_base = Q + r * (NUM_HEADS * HEAD_DIM)
    o_base = O + r * (NUM_HEADS * HEAD_DIM)

    q = tl.load(
        q_base + (hkv * RATIO + tl.minimum(offs_r, RATIO - 1))[:, None] * HEAD_DIM
        + offs_d[None, :],
        mask=(offs_r < RATIO)[:, None], other=0.0,
    )
    m = tl.full([RP], float("-inf"), dtype=tl.float32)
    l = tl.zeros([RP], dtype=tl.float32)
    acc = tl.zeros([RP, HEAD_DIM], dtype=tl.float32)

# Dynamic loop bound: the runner pads block_tables to MAX_BLOCKS (the
    # model's max), so iterate exactly nblk times and only touch live blocks.
    for blk in range(0, nblk):
        bid = tl.load(BLOCK_TABLES + r * MAX_BLOCKS + blk)
        k_off = tl.load(BLOCK_OFFSETS + bid).to(tl.int32)
        v_off = tl.load(BLOCK_OFFSETS + NUM_BLOCKS + bid).to(tl.int32)

        k = _load_dequant(CACHE, k_off, offs_s, offs_d, NUM_KV_HEADS, HEAD_DIM,
                          K_FMT, hkv)
        v = _load_dequant(CACHE, v_off, offs_s, offs_d, NUM_KV_HEADS, HEAD_DIM,
                          V_FMT, hkv)

# per-group scales: [2*NUM_BLOCKS, H_kv, GROUPS]; K row = bid,
        # V row = NUM_BLOCKS + bid.
        s_k = tl.load(SCALES + bid * (NUM_KV_HEADS * GROUPS)
                      + hkv * GROUPS + (offs_d // (HEAD_DIM // GROUPS)))
        s_v = tl.load(SCALES + (NUM_BLOCKS + bid) * (NUM_KV_HEADS * GROUPS)
                      + hkv * GROUPS + (offs_d // (HEAD_DIM // GROUPS)))
        k = k * s_k[None, :]
        v = v * s_v[None, :]

        mask = (blk * BLOCK + offs_s) < seq
        qk = tl.dot(q, tl.trans(k.to(q.dtype))) * sm_scale
        qk = tl.where((offs_r < RATIO)[:, None] & mask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m - m_ij)
        l = l * alpha + l_ij
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(q.dtype), v.to(q.dtype))
        m = m_ij

    l = 1.0 / l
    acc = acc * l[:, None]
    tl.store(
        o_base + (hkv * RATIO + offs_r)[:, None] * HEAD_DIM + offs_d[None, :],
        acc.to(tl.bfloat16) if Q.dtype.element_ty == tl.bfloat16 else acc.to(tl.float16),
        mask=(offs_r < RATIO)[:, None],
    )


def pspkv_decode_forward(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    precision_table: torch.Tensor,
    block_offsets: torch.Tensor,
    scale_table: torch.Tensor,
    kv_layout,
    fmt_k: int,
    fmt_v: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Batched PSP-KV decode: one launch for all requests (homogeneous precision).

    ``fmt_k``/``fmt_v`` are the *inner* format codes (0=lossless fp16, 1=e4m3,
    2=e5m2, 3=fp4, 4=int8) derived from the packed precision code.  The caller
    must guarantee the batch is homogeneous; mixed-precision batches should use
    the per-request fallback instead.
    """
    num_reqs, num_heads, head_dim = q.shape
    num_kv_heads = kv_layout.num_kv_heads
    block = kv_layout.block_size
    max_blocks = block_tables.shape[1]
    ratio = num_heads // num_kv_heads
    num_blocks = precision_table.numel()
    groups = head_dim // 16
    if sm_scale is None:
        sm_scale = 1.0 / head_dim ** 0.5

    device = q.device
    if scale_table is None:
        scale_table = torch.ones(2 * num_blocks, num_kv_heads, groups,
                                 dtype=torch.float32, device=device)
    o = torch.empty_like(q)

    _pspkv_decode_kernel[(num_reqs * num_kv_heads,)](
        q, cache, block_tables, seq_lens, block_offsets, scale_table, o,
        sm_scale,
        NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, RATIO=ratio,
        HEAD_DIM=head_dim, BLOCK=block, MAX_BLOCKS=max_blocks,
        K_FMT=fmt_k, V_FMT=fmt_v, NUM_BLOCKS=num_blocks, GROUPS=groups,
        num_warps=4, num_stages=2,
    )
    return o


__all__ = ["pspkv_decode_forward"]
