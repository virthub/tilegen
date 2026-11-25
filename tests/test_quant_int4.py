# tests/test_quant_int4.py
#
# GPU-gated tests for the INT4 group-quantized KV path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_int4_roundtrip():
    from tilegen.ops.quant_int4 import dequant_int4_kv, quantize_int4_kv

    torch.manual_seed(0)
    for D in (64, 128):
        x = torch.randn(2, 8, 50, D, device="cuda", dtype=torch.float16)
        packed, scale = quantize_int4_kv(x, group_size=64)
        assert packed.shape == (2, 8, 50, D // 2)
        assert packed.dtype == torch.uint8
        assert scale.shape == (2, 8, 50, D // 64)
        xhat = dequant_int4_kv(packed, scale, group_size=64)
        # 4-bit symmetric quantization: ~12.5% max magnitude error is inherent.
        assert (xhat.float() - x.float()).abs().max().item() < 0.6


def test_int4_attention_matches_within_tolerance():
    """INT4 K/V attention matches full-precision within the quant error."""
    import torch.nn.functional as F

    from tilegen.ops.quant_int4 import dequant_int4_kv, quantize_int4_kv

    torch.manual_seed(0)
    B, H, N, D = 2, 4, 128, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    kp, ks = quantize_int4_kv(k, group_size=64)
    vp, vs = quantize_int4_kv(v, group_size=64)
    kq = dequant_int4_kv(kp, ks, group_size=64)
    vq = dequant_int4_kv(vp, vs, group_size=64)

    out = F.scaled_dot_product_attention(q, kq, vq, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    cos = F.cosine_similarity(
        out.reshape(-1, D).float(), ref.reshape(-1, D).float(), dim=-1
    ).mean().item()
    assert cos > 0.98, f"mean cosine {cos}"


def test_int4_in_kernel_decode_matches_reference():
    """The in-kernel INT4 decode (split-dot) matches the torch-dequant reference."""
    import torch.nn.functional as F

    from tilegen.layout.layout import PagedKVLayout
    from tilegen.ops.paged_decode import paged_decode_int4_forward
    from tilegen.ops.quant_int4 import dequant_int4_kv, quantize_int4_kv

    torch.manual_seed(0)
    B, Hq, Hkv, D, block, gs = 2, 8, 2, 64, 16, 64
    seq_lens = torch.tensor([48, 33], device="cuda", dtype=torch.int32)
    nb = 16
    k_dense = torch.randn(B, 48, Hkv, D, device="cuda", dtype=torch.float16)
    v_dense = torch.randn(B, 48, Hkv, D, device="cuda", dtype=torch.float16)
    kp, ks = quantize_int4_kv(k_dense, group_size=gs)
    vp, vs = quantize_int4_kv(v_dense, group_size=gs)
    kc = torch.zeros(nb, block, Hkv, D // 2, device="cuda", dtype=torch.uint8)
    vc = torch.zeros(nb, block, Hkv, D // 2, device="cuda", dtype=torch.uint8)
    ksc = torch.zeros(nb, block, Hkv, D // gs, device="cuda", dtype=torch.float16)
    vsc = torch.zeros(nb, block, Hkv, D // gs, device="cuda", dtype=torch.float16)
    bt = torch.zeros(B, 8, device="cuda", dtype=torch.int32)
    for b in range(B):
        for i in range((int(seq_lens[b]) + block - 1) // block):
            bid = b * 8 + i
            bt[b, i] = bid
            s = i * block
            e = min(s + block, int(seq_lens[b]))
            kc[bid, : e - s] = kp[b, s:e]
            vc[bid, : e - s] = vp[b, s:e]
            ksc[bid, : e - s] = ks[b, s:e]
            vsc[bid, : e - s] = vs[b, s:e]
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.float16)
    layout = PagedKVLayout(block_size=block, num_kv_heads=Hkv, head_dim=D)
    out = paged_decode_int4_forward(q, kc, vc, ksc, vsc, bt, seq_lens, layout, group_size=gs)

    ratio = Hq // Hkv
    ref = torch.zeros(B, Hq, D, device="cuda", dtype=torch.float16)
    for b in range(B):
        seq = int(seq_lens[b])
        k = dequant_int4_kv(kp[b, :seq], ks[b, :seq]).permute(1, 0, 2).unsqueeze(0).repeat_interleave(ratio, dim=1)
        v = dequant_int4_kv(vp[b, :seq], vs[b, :seq]).permute(1, 0, 2).unsqueeze(0).repeat_interleave(ratio, dim=1)
        ref[b] = F.scaled_dot_product_attention(q[b].unsqueeze(0).unsqueeze(2), k, v, is_causal=False).squeeze(0).squeeze(1)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2
