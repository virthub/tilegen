# tests/test_flash.py
#
# GPU-gated tests for the Generation-0 Triton FA2 reference kernel

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_fa2_matches_pytorch_sdpa():
    import torch.nn.functional as F

    from tilegen.ops.flash import flash_attention_forward

    torch.manual_seed(0)
    B, H, N, D = 2, 8, 512, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)

    out, L, M = flash_attention_forward(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    max_err = (out - ref).abs().max().item()
    cos = F.cosine_similarity(out.reshape(-1, D).float(), ref.reshape(-1, D).float(), dim=-1).mean().item()
    assert max_err < 5e-2, f"max_abs_err {max_err}"
    assert cos > 0.99, f"mean_cos {cos}"
    assert L.shape == (B, H, N) and M.shape == (B, H, N)


def test_fa2_non_causal():
    import torch.nn.functional as F

    from tilegen.ops.flash import flash_attention_forward

    torch.manual_seed(1)
    B, H, N, D = 2, 4, 256, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)

    out, _, _ = flash_attention_forward(q, k, v, is_causal=False)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    assert (out - ref).abs().max().item() < 5e-2


def test_fa3_warp_specialize_matches_reference():
    import torch.nn.functional as F

    from tilegen.ops.flash import flash_attention_fa3_forward

    torch.manual_seed(0)
    B, H, N, D = 2, 8, 512, 128
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out, L, M = flash_attention_fa3_forward(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out - ref).abs().max().item() < 5e-2
    assert L.shape == (B, H, N) and M.shape == (B, H, N)
