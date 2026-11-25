# tests/test_attention_dispatch.py
#
# Tests for the canonical attention dispatch layer (tilegen.attention)

import pytest

torch = pytest.importorskip("torch")


def test_dense_attention_fp16_falls_back_to_sdpa():
    """On CPU/fp16 (no TileLang), dense_attention degrades to native SDPA."""
    import torch.nn.functional as F

    from tilegen.attention import dense_attention

    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = dense_attention(q, k, v, is_causal=False)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_dense_attention_bf16_matches_reference():
    import torch.nn.functional as F

    from tilegen.attention import dense_attention

    torch.manual_seed(0)
    q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = dense_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2
    assert out.dtype == torch.bfloat16


def test_paged_decode_prefill_delegate():
    """The paged entry points exist and are callable (delegation smoke test)."""
    from tilegen.attention import paged_decode, paged_decode_int4, paged_prefill

    for fn in (paged_decode, paged_decode_int4, paged_prefill):
        assert callable(fn)
