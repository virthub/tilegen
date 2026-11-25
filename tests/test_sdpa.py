# tests/test_sdpa.py
#
# GPU-gated tests for the Phase-3 SDPA dispatch (tilegen::sdpa)

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


@pytest.fixture(scope="module")
def tensors():
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 256, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def test_custom_op_matches_reference(tensors):
    import torch.nn.functional as F

    from tilegen.runtime.sdpa import _register_ops

    _register_ops()
    q, k, v = tensors
    out = torch.ops.tilegen.sdpa(q, k, v, True, None)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out - ref).abs().max().item() < 5e-2


def test_sdpa_kernel_routes_and_is_correct(tensors):
    import torch.nn.functional as F

    import tilegen
    from tilegen.runtime.sdpa import dispatch_count

    q, k, v = tensors
    before = dispatch_count()
    with tilegen.sdpa_kernel():
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    after = dispatch_count()
    assert after == before + 1, "dense bf16 CUDA attention must route to TileGen"

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out - ref).abs().max().item() < 5e-2


def test_sdpa_kernel_falls_back_for_fp32(tensors):
    import torch.nn.functional as F

    import tilegen
    from tilegen.runtime.sdpa import dispatch_count

    q, k, v = tensors
    before = dispatch_count()
    with tilegen.sdpa_kernel():
        out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
    assert dispatch_count() == before, "fp32 must fall back to native SDPA"

    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
    assert (out - ref).abs().max().item() < 1e-3


def test_meta_kernel_registered(tensors):
    from tilegen.runtime.sdpa import _register_ops

    _register_ops()
    assert hasattr(torch.ops.tilegen, "sdpa"), "tilegen::sdpa must be registered"
    assert torch.ops.tilegen.sdpa._schemas, "op must expose at least one schema"
