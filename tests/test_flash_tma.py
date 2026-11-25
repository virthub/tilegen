# tests/test_flash_tma.py
#
# GPU-gated tests for the TMA (FA3 memory-path) FlashAttention variant

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_tma_attention_matches_reference():
    import torch.nn.functional as F

    from tilegen.ops.flash_tma import flash_attention_tma_forward

    torch.manual_seed(0)
    for D in (64, 128):
        B, H, N = 2, 8, 512
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out, L, M = flash_attention_tma_forward(q, k, v, is_causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert (out - ref).abs().max().item() < 5e-2, f"D={D}"
        assert L.shape == (B, H, N) and M.shape == (B, H, N)
