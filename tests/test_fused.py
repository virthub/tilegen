# tests/test_fused.py
#
# GPU-gated tests for the fused elementwise operators
# (RMSNorm/RoPE/SwiGLU)

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_rms_norm_matches_reference():
    from tilegen.ops.fused import rms_norm_forward

    torch.manual_seed(0)
    x = torch.randn(128, 512, device="cuda", dtype=torch.float16)
    w = torch.randn(512, device="cuda", dtype=torch.float16)
    y = rms_norm_forward(x, w, eps=1e-6)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float())
    assert (y.float() - ref).abs().max().item() < 1e-2


def test_rope_matches_reference():
    from tilegen.ops.fused import rope_forward

    torch.manual_seed(0)
    T, H, D = 64, 8, 64
    x = torch.randn(T, H, D, device="cuda", dtype=torch.float16)
    freqs = torch.arange(0, D // 2, device="cuda").float()
    theta = 1.0 / (10000.0 ** (freqs * 2 / D))
    angles = torch.arange(T, device="cuda").float()[:, None] * theta[None, :]
    cos, sin = angles.cos(), angles.sin()
    y = rope_forward(x, cos, sin)
    x1, x2 = x[..., : D // 2].float(), x[..., D // 2 :].float()
    ref = torch.cat([x1 * cos[:, None, :] - x2 * sin[:, None, :],
                     x2 * cos[:, None, :] + x1 * sin[:, None, :]], dim=-1)
    assert (y.float() - ref).abs().max().item() < 1e-2


def test_swiglu_matches_reference():
    from tilegen.ops.fused import swiglu_forward

    torch.manual_seed(0)
    g = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    u = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    y = swiglu_forward(g, u)
    ref = g.float() * torch.sigmoid(g.float()) * u.float()
    assert (y.float() - ref).abs().max().item() < 1e-2


def test_w8a8_gemm_matches_reference():
    from tilegen.ops.quant_gemm import w8a8_gemm_forward

    torch.manual_seed(0)
    M, N, K = 128, 256, 128
    a = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (K, N), device="cuda", dtype=torch.int8)
    c = w8a8_gemm_forward(a, b)
    # int8 @ int8 sums stay < 2^24, so fp32 accumulation is exact.
    ref = (a.float() @ b.float()).round().to(torch.int32)
    assert (c == ref).all().item()


def test_layer_norm_matches_reference():
    import torch.nn.functional as F

    from tilegen.ops.fused import layer_norm_forward

    torch.manual_seed(0)
    x = torch.randn(128, 512, device="cuda", dtype=torch.float16)
    w = torch.randn(512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, device="cuda", dtype=torch.float16)
    y = layer_norm_forward(x, w, b, eps=1e-5)
    ref = F.layer_norm(x.float(), (512,), w.float(), b.float(), eps=1e-5)
    assert (y.float() - ref).abs().max().item() < 1e-2


def test_gelu_and_geglu_match_reference():
    import torch.nn.functional as F

    from tilegen.ops.fused import geglu_forward, gelu_forward

    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    assert (gelu_forward(x).float() - F.gelu(x.float(), approximate="tanh")).abs().max().item() < 1e-2
    g = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    u = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    ref = F.gelu(g.float(), approximate="tanh") * u.float()
    assert (geglu_forward(g, u).float() - ref).abs().max().item() < 1e-2


def test_moe_topk_gate():
    from tilegen.ops.fused import moe_topk_gate

    torch.manual_seed(0)
    logits = torch.randn(32, 8, device="cuda")
    weights, idx = moe_topk_gate(logits, k=2)
    assert weights.shape == (32, 2) and idx.shape == (32, 2)
    assert torch.allclose(weights.sum(-1), torch.ones(32, device="cuda"), atol=1e-5)
    # Selected experts must be the actual top-2 logits.
    top2 = torch.topk(logits, 2, dim=-1).values
    assert torch.allclose(torch.sort(weights, dim=-1).values,
                          torch.sort(torch.softmax(top2, dim=-1), dim=-1).values, atol=1e-5)
