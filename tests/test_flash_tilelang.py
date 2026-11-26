# tests/test_flash_tilelang.py
#
# GPU-gated tests for the TileLang FA4 (tcgen05/TMEM) flash attention

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 10,
    reason="FA4 tcgen05 requires a Blackwell GPU (sm_100+)",
)


def _fa4():
    try:
        from tilegen.ops.flash_tilelang import flash_attention_tilelang_forward
    except Exception:
        pytest.skip("TileLang not installed / not compilable")
    return flash_attention_tilelang_forward


def test_tilelang_fa4_matches_reference():
    import torch.nn.functional as F

    fa4 = _fa4()
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 512, 128
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out, L, M = fa4(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2
    assert L.shape == (B, H, N) and L.dtype == torch.float32
    assert M.shape == (B, H, N) and M.dtype == torch.float32


def test_tilelang_fa4_bf16():
    import torch.nn.functional as F

    fa4 = _fa4()
    torch.manual_seed(1)
    B, H, N, D = 2, 4, 256, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out, L, M = fa4(q, k, v, is_causal=True)
    assert out.dtype == torch.bfloat16
    assert L.shape == (B, H, N) and M.shape == (B, H, N)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2


def test_tilelang_fa4_codegen_compiles_and_runs():
    import importlib.util
    import os
    import tempfile

    import torch.nn.functional as F

    from tilegen.backend import ensure_backends_registered, get_backend
    from tilegen.ops.attention import AttentionSpec, build_dense_attention_vtm
    from tilegen.target.spec import get_target

    ensure_backends_registered()
    spec = AttentionSpec(
        batch=2, num_heads=4, num_kv_heads=4, head_dim=64,
        seq_len=512, dtype="bf16", causal=True,
    )
    kernel = build_dense_attention_vtm(spec, get_target("sm_120"))
    src = get_backend("tilelang").lower(kernel, get_target("sm_120"))

    fd, path = tempfile.mkstemp(suffix=".py", prefix="gen_tl_")
    os.close(fd)
    with open(path, "w") as f:
        f.write(src)
    m = importlib.util.spec_from_file_location("gen_tl", path)
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)

    torch.manual_seed(0)
    q = torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = mod.dense_fa2_forward(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2
