# tests/test_vllm_pspkv.py
#
# GPU-gated tests for the TILEGEN_PSPKV backend (vllm-pspkv fork integration):
# registration, cache geometry, and decode parity vs the fork's PSP_KV backend

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("TILEGEN_PSPKV tests are CUDA-only", allow_module_level=True)

pytest.importorskip("vllm")

try:
    from vllm.v1.attention.backends import psp_kv_attn as _fork
    from vllm.v1.attention.backends.psp_kv_attn import (
        PSPKVAttentionImpl,
        PSPKVAttentionMetadata,
    )
except ImportError:
    pytest.skip("vllm-pspkv fork backend module not importable", allow_module_level=True)

from tilegen.integrations.vllm_pspkv.attention_backend import (
    TileGenPSPKVAttentionBackend,
    TileGenPSPKVAttentionImpl,
    TileGenPSPKVAttentionMetadata,
)

BLOCK = 128
HEAD = 64
NUM_GROUPS = HEAD // 16


def _fp8_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float8_e4m3fn).view(torch.uint8)


def _fp8_dequant(b: torch.Tensor) -> torch.Tensor:
    return _fork._dequant_fp8(b, 1).float()


def _build_case(device, fmt_code: int = 1, B: int = 2, H_kv: int = 8, H_q: int = 8,
                ctx: int = 200):
    """Build a synthetic PSP-KV cache (Enhanced format, token-major pages).

    Mirrors tests/kernels/attention/test_psp_kv_kernels.py: one page per block
    holds [K bytes | V bytes] per head; block_offsets stacks [K | V]; the
    current decode token is already written into the cache (runner write-back
    convention), so context_lens tokens are attended per request.
    """
    num_blocks = B * 2
    eb = 2.0 if fmt_code == 0 else (0.5 if fmt_code == 3 else 1.0)
    k_bytes = int(BLOCK * HEAD * eb)
    v_bytes = int(BLOCK * HEAD * eb)
    page = H_kv * (k_bytes + v_bytes)
    cache = torch.zeros(num_blocks * page, dtype=torch.uint8, device=device)
    k_offsets = torch.arange(num_blocks, dtype=torch.int32, device=device) * page
    v_offsets = k_offsets + H_kv * k_bytes
    head_stride = int(HEAD * eb)  # bytes per (h) head row
    block_offsets = torch.cat([k_offsets, v_offsets]).contiguous()

    prec_code = 4 if fmt_code == 4 else (fmt_code | (fmt_code << 2))
    prec = torch.full((num_blocks,), prec_code, dtype=torch.uint8, device=device)
    scales = torch.ones(2 * num_blocks, H_kv, NUM_GROUPS, device=device)

    torch.manual_seed(0)
    seq_lens = torch.tensor([ctx, BLOCK + 10], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    query = torch.randn(B, H_q, HEAD, device=device, dtype=torch.float16)
    key = torch.randn(B, H_kv, HEAD, device=device)
    value = torch.randn(B, H_kv, HEAD, device=device)

    from tilegen.integrations.vllm_pspkv.attention_backend import _FP4_E2M1

    e2m1 = torch.tensor(_FP4_E2M1, dtype=torch.float32, device=device)
    e2m1[7] = 3.0  # inf slot unused; keep finite for reference math

    # Per-request reference K/V: prefix + current token, kept exactly
    # representable in the chosen format.
    k_ref: list[torch.Tensor] = []
    v_ref: list[torch.Tensor] = []
    for r in range(B):
        seq = int(seq_lens[r].item())
        if fmt_code == 1:  # fp8 e4m3
            kt = (torch.randint(0, 8, (seq, H_kv, HEAD), device=device) - 4).float() * 0.5
            vt = (torch.randint(0, 8, (seq, H_kv, HEAD), device=device) - 4).float() * 0.5
            key = key.mul(0).add_(1.0)
            value = value.mul(0).add_(-0.5)
        elif fmt_code == 4:  # int8
            kt = (torch.randint(0, 16, (seq, H_kv, HEAD), device=device) - 8).float()
            vt = (torch.randint(0, 16, (seq, H_kv, HEAD), device=device) - 8).float()
            key = key.mul(0).add_(2.0)
            value = value.mul(0).add_(-2.0)
        elif fmt_code == 0:  # lossless fp16
            kt = (torch.randint(0, 8, (seq, H_kv, HEAD), device=device) - 4).float()
            vt = (torch.randint(0, 8, (seq, H_kv, HEAD), device=device) - 4).float()
            kt = kt.half().float()  # exactly what fp16 storage round-trips
            vt = vt.half().float()
            key = key.half().float().mul(0).add_(1.0)
            value = value.half().float().mul(0).add_(-0.5)
        else:  # fp4 e2m1 (exclude the inf nibbles 6, 7, 14 and 15)
            kidx = torch.randint(0, 12, (seq, H_kv, HEAD), device=device)
            vidx = torch.randint(0, 12, (seq, H_kv, HEAD), device=device)
            kn = kidx + 2 * (kidx >= 6)   # -> {0..5, 8..13}, all finite
            vn = vidx + 2 * (vidx >= 6)
            kt = e2m1[kn]
            vt = e2m1[vn]
            key = key.mul(0).add_(1.0)
            value = value.mul(0).add_(-0.5)
            kn[-1] = 2   # e2m1[2] = 1.0, matches the current-token key value
            vn[-1] = 9   # e2m1[9] = -0.5, matches the current-token value value

        kt[-1] = key[r]
        vt[-1] = value[r]
        k_ref.append(kt)
        v_ref.append(vt)
        # write into the byte cache (token-major rows: t * H_kv + h)
        nblk = (seq + BLOCK - 1) // BLOCK
        for blk in range(nblk):
            bid = int(block_table[r, blk].item())
            lo = blk * BLOCK
            hi = min(seq, lo + BLOCK)
            n = hi - lo
            rows = (torch.arange(n, device=device)[:, None] * H_kv
                    + torch.arange(H_kv, device=device)[None, :])
            k_start = int(k_offsets[bid]) + rows * head_stride
            v_start = int(v_offsets[bid]) + rows * head_stride
            cache_flat = cache.view(-1)
            for h in range(H_kv):
                krow = k_start[:, h].reshape(-1, 1)
                vrow = v_start[:, h].reshape(-1, 1)
                if fmt_code == 0:  # fp16, 2 bytes/elem little-endian
                    kb = kt[lo:hi, h].half().view(torch.uint8)  # [n, HEAD*2]
                    vb = vt[lo:hi, h].half().view(torch.uint8)
                    d2 = torch.arange(HEAD, device=device) * 2
                    cache_flat[krow + d2[None, :]] = kb[:, 0::2]
                    cache_flat[krow + d2[None, :] + 1] = kb[:, 1::2]
                    cache_flat[vrow + d2[None, :]] = vb[:, 0::2]
                    cache_flat[vrow + d2[None, :] + 1] = vb[:, 1::2]
                elif fmt_code == 3:  # fp4 nibbles
                    knb = (kn[lo:hi, h] if fmt_code == 3 else None)
                    vnb = (vn[lo:hi, h] if fmt_code == 3 else None)
                    kpack = (knb[:, 0::2] | (knb[:, 1::2] << 4)).to(torch.uint8)
                    vpack = (vnb[:, 0::2] | (vnb[:, 1::2] << 4)).to(torch.uint8)
                    dh = torch.arange(HEAD // 2, device=device)
                    cache_flat[krow + dh[None, :]] = kpack
                    cache_flat[vrow + dh[None, :]] = vpack
                elif fmt_code == 1:
                    kt_blk = kt[lo:hi]
                    vt_blk = vt[lo:hi]
                    d = torch.arange(HEAD, device=device)
                    cache_flat[krow + d[None, :]] = kt_blk[:, h].to(torch.float8_e4m3fn).view(torch.uint8)
                    cache_flat[vrow + d[None, :]] = vt_blk[:, h].to(torch.float8_e4m3fn).view(torch.uint8)
                else:  # int8
                    d = torch.arange(HEAD, device=device)
                    cache_flat[krow + d[None, :]] = kt[lo:hi, h].to(torch.int8).view(torch.uint8)
                    cache_flat[vrow + d[None, :]] = vt[lo:hi, h].to(torch.int8).view(torch.uint8)

    slot_mapping = torch.stack(
        [block_table[r, (seq_lens[r].item() - 1) // BLOCK] * BLOCK
         + (seq_lens[r].item() - 1) % BLOCK for r in range(B)]
    ).to(torch.int64)

    return SimpleNamespace(
        cache=cache, block_offsets=block_offsets, prec=prec, scales=scales,
        block_table=block_table, seq_lens=seq_lens, query=query,
        key=key.contiguous(), value=value.contiguous(), slot_mapping=slot_mapping,
        k_ref=k_ref, v_ref=v_ref,
    )


def _meta(case, device):
    return SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=2,
        seq_lens=case.seq_lens,
        context_lens=case.seq_lens,
        block_tables=case.block_table,
        block_table_tensor=case.block_table,
        precision_table=case.prec,
        scale_table=case.scales,
        block_offsets=case.block_offsets,
        slot_mapping=case.slot_mapping,
        format_generation=1,
        max_query_len=1,
        query_start_loc=torch.arange(3, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.arange(3, dtype=torch.int32),
        num_decode_tokens=2,
    )


def _layer():
    return SimpleNamespace(
        num_kv_heads=8, num_kv_heads_per_partition=8, head_size=HEAD,
        layer_name="model.layers.0",
    )


def _reference(case, scale):
    import torch.nn.functional as F

    outs = []
    for r in range(2):
        q4 = case.query[r].unsqueeze(0).unsqueeze(2)          # [1, H, 1, D]
        k4 = case.k_ref[r].transpose(0, 1).unsqueeze(0).to(torch.float16)
        v4 = case.v_ref[r].transpose(0, 1).unsqueeze(0).to(torch.float16)
        outs.append(F.scaled_dot_product_attention(
            q4, k4, v4, is_causal=False, scale=scale).squeeze(0).squeeze(1))
    return torch.stack(outs)


@pytest.mark.parametrize("fmt_code", [1, 4])
def test_decode_parity_with_psp_kv(fmt_code: int):
    device = "cuda"
    scale = 1.0 / HEAD ** 0.5
    case = _build_case(device, fmt_code=fmt_code)
    meta_ns = _meta(case, device)
    layer = _layer()
    ref = _reference(case, scale)

    # TileGen path first (read-only), then the fork path (write-back quantizes
    # the current token into the cache — must not disturb the first run).
    tg_meta = TileGenPSPKVAttentionMetadata(common_metadata=meta_ns)
    tg_impl = TileGenPSPKVAttentionImpl(num_heads=8, head_size=HEAD, scale=scale,
                                        num_kv_heads=8)
    out_tg = torch.zeros_like(case.query)
    tg_impl.forward(layer, case.query, case.key, case.value, case.cache, tg_meta, out_tg)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_tg, ref, atol=2e-2, rtol=2e-2)

    if fmt_code == 1:
        # fp8 e4m3: also compare against the fork's own PSP_KV decode kernels.
        fork_meta = PSPKVAttentionMetadata(common_metadata=meta_ns)
        fork_impl = PSPKVAttentionImpl(num_heads=8, head_size=HEAD, scale=scale,
                                       num_kv_heads=8, alibi_slopes=None,
                                       sliding_window=None, kv_cache_dtype="auto")
        out_fork = torch.zeros_like(case.query)
        fork_impl.forward(layer, case.query, case.key, case.value, case.cache,
                          fork_meta, out_fork)
        torch.cuda.synchronize()
        torch.testing.assert_close(out_fork, ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(out_tg, out_fork, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("fmt_code", [0, 1, 3, 4])
def test_triton_decode_parity(fmt_code: int):
    """Batched Triton PSP-KV decode kernel vs the FP32 reference."""
    device = "cuda"
    scale = 1.0 / HEAD ** 0.5
    case = _build_case(device, fmt_code=fmt_code)

    from tilegen.integrations.vllm_pspkv.attention_backend import _fmt_k, _fmt_v
    from tilegen.layout.layout import PagedKVLayout
    from tilegen.ops.pspkv_decode import pspkv_decode_forward

    p0 = int(case.prec[0].item())
    layout = PagedKVLayout(block_size=BLOCK, num_kv_heads=8, head_dim=HEAD)
    out = pspkv_decode_forward(
        case.query, case.cache, case.block_table, case.seq_lens,
        case.prec, case.block_offsets, case.scales, layout,
        _fmt_k(p0), _fmt_v(p0), sm_scale=scale,
    )
    ref = _reference(case, scale)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


def test_backend_registration_and_cache_shape():
    assert TileGenPSPKVAttentionBackend.get_name() == "TILEGEN_PSPKV"
    assert TileGenPSPKVAttentionBackend.get_supported_kernel_block_sizes() == [BLOCK]
    shape = TileGenPSPKVAttentionBackend.get_kv_cache_shape(
        num_blocks=16, block_size=BLOCK, num_kv_heads=8, head_size=HEAD,
        cache_dtype_str="auto", dtype=torch.float16,
    )
    from vllm.v1.kv_cache_interface import _psp_kv_avg_bytes

    per_elem = _psp_kv_avg_bytes(torch.float16)
    assert shape == (2 * int(16 * BLOCK * 8 * HEAD * per_elem),)
    assert TileGenPSPKVAttentionBackend.get_kv_cache_block_dim(
        BLOCK, 8, HEAD, "auto") == 0
