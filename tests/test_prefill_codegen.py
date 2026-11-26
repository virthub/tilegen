# tests/test_prefill_codegen.py
#
# GPU-gated tests for the generated dense FA2 prefill codegen

import importlib.util
import os
import tempfile
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _load_module(source: str, name: str):
    fd, path = tempfile.mkstemp(suffix=".py", prefix=f"tilegen_{name}_")
    os.close(fd)
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dense_fa2_codegen_compiles_and_runs():
    import torch.nn.functional as F

    from tilegen.backend import ensure_backends_registered, get_backend
    from tilegen.ops.attention import AttentionSpec, build_dense_attention_vtm
    from tilegen.target.spec import get_target

    ensure_backends_registered()
    spec = AttentionSpec(
        batch=2, num_heads=4, num_kv_heads=4, head_dim=64,
        seq_len=512, dtype="bf16", causal=True,
    )
    kernel = build_dense_attention_vtm(spec, get_target("sm_90a"))
    src = get_backend("triton").lower(kernel, get_target("sm_90a"))
    mod = _load_module(src, "gen_dense_fa2")

    torch.manual_seed(0)
    q = torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out, L, M = mod.dense_fa2_forward(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2
    assert L.shape == (2, 4, 512) and M.shape == (2, 4, 512)


def test_dense_fa2_codegen_non_causal():
    import torch.nn.functional as F

    from tilegen.backend import ensure_backends_registered, get_backend
    from tilegen.ops.attention import AttentionSpec, build_dense_attention_vtm
    from tilegen.target.spec import get_target

    ensure_backends_registered()
    spec = AttentionSpec(
        batch=1, num_heads=2, num_kv_heads=2, head_dim=64,
        seq_len=256, dtype="bf16", causal=False,
    )
    kernel = build_dense_attention_vtm(spec, get_target("sm_90a"))
    src = get_backend("triton").lower(kernel, get_target("sm_90a"))
    mod = _load_module(src, "gen_dense_fa2_nc")

    torch.manual_seed(1)
    q = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out, _, _ = mod.dense_fa2_forward(q, k, v, is_causal=False)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2


def test_fa3_ws_codegen_compiles_and_runs():
    import torch.nn.functional as F

    from tilegen.backend import ensure_backends_registered, get_backend
    from tilegen.ops.attention import AttentionSpec, build_dense_attention_vtm
    from tilegen.target.spec import get_target

    ensure_backends_registered()
    spec = AttentionSpec(
        batch=2, num_heads=4, num_kv_heads=4, head_dim=64,
        seq_len=512, dtype="bf16", causal=True,
    )
    # Blackwell (tcgen05): FA3 emits automatic warp specialization.
    kernel = build_dense_attention_vtm(spec, get_target("sm_120"), variant="fa3")
    assert kernel.topology.grid.mode.value == "persistent_sm"
    assert kernel.topology.roles.phase_mode.value == "ping_pong"
    src = get_backend("triton").lower(kernel, get_target("sm_120"))
    assert "warp_specialize=True" in src # automatic warp specialization emitted
    mod = _load_module(src, "gen_fa3")

    torch.manual_seed(0)
    q = torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out, _, _ = mod.dense_fa2_forward(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2
