# tests/test_paged_prefill.py
#
# GPU-gated tests for paged prefill (kernel + generated codegen)

import importlib.util
import os
import tempfile
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _paged_prefill_inputs(q_len=64):
    import torch.nn.functional as F

    torch.manual_seed(0)
    num_reqs, Hq, Hkv, D, block = 2, 8, 2, 64, 16
    seq_lens = torch.tensor([q_len, q_len], device="cuda", dtype=torch.int32)
    num_blocks = 16
    k_dense = torch.randn(num_reqs, q_len, Hkv, D, device="cuda", dtype=torch.float16)
    v_dense = torch.randn(num_reqs, q_len, Hkv, D, device="cuda", dtype=torch.float16)
    kc = torch.randn(num_blocks, block, Hkv, D, device="cuda", dtype=torch.float16)
    vc = torch.randn(num_blocks, block, Hkv, D, device="cuda", dtype=torch.float16)
    bt = torch.zeros(num_reqs, 8, device="cuda", dtype=torch.int32)
    for r in range(num_reqs):
        nb = (q_len + block - 1) // block
        for i in range(nb):
            bid = r * 8 + i
            bt[r, i] = bid
            s = i * block
            e = min(s + block, q_len)
            kc[bid, : e - s] = k_dense[r, s:e]
            vc[bid, : e - s] = v_dense[r, s:e]
    q = torch.randn(num_reqs, q_len, Hq, D, device="cuda", dtype=torch.float16)

    ratio = Hq // Hkv
    ref = torch.zeros(num_reqs, q_len, Hq, D, device="cuda", dtype=torch.float16)
    for r in range(num_reqs):
        k = k_dense[r].permute(1, 0, 2).unsqueeze(0).repeat_interleave(ratio, dim=1)
        v = v_dense[r].permute(1, 0, 2).unsqueeze(0).repeat_interleave(ratio, dim=1)
        q4 = q[r].permute(1, 0, 2).unsqueeze(0)
        ref[r] = F.scaled_dot_product_attention(q4, k, v, is_causal=True).squeeze(0).permute(1, 0, 2)
    return (q, kc, vc, bt, seq_lens, Hkv, block, ref)


def _load_module(source: str, name: str):
    fd, path = tempfile.mkstemp(suffix=".py", prefix=f"tilegen_{name}_")
    os.close(fd)
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_paged_prefill_kernel_matches_reference():
    from tilegen.layout.layout import PagedKVLayout
    from tilegen.ops.paged_prefill import paged_prefill_forward

    q, kc, vc, bt, seq_lens, Hkv, block, ref = _paged_prefill_inputs()
    layout = PagedKVLayout(block_size=block, num_kv_heads=Hkv, head_dim=q.shape[-1])
    out = paged_prefill_forward(q, kc, vc, bt, seq_lens, layout, is_causal=True)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2


def test_paged_prefill_codegen_compiles_and_runs():
    from tilegen.backend import ensure_backends_registered, get_backend
    from tilegen.ops.attention import AttentionSpec, build_paged_prefill_vtm
    from tilegen.target.spec import get_target

    ensure_backends_registered()
    spec = AttentionSpec(
        batch=2, num_heads=8, num_kv_heads=2, head_dim=64,
        seq_len=64, query_len=64, num_kv_blocks=16, page_size=16, dtype="bf16",
        causal=True,
    )
    kernel = build_paged_prefill_vtm(spec, get_target("sm_90a"))
    src = get_backend("triton").lower(kernel, get_target("sm_90a"))
    mod = _load_module(src, "gen_paged_prefill")

    q, kc, vc, bt, seq_lens, Hkv, block, ref = _paged_prefill_inputs()
    out = mod.paged_prefill_forward(q, kc, vc, bt, seq_lens, Hkv, block, is_causal=True)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2
