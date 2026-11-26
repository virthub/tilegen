# tests/test_decode.py
#
# GPU-gated tests for batched paged-KV decode (kernel + generated codegen)

import importlib.util
import os
import tempfile
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _paged_inputs():
    """Scatter dense K/V into a paged cache and return decode inputs + reference."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    B, Hq, Hkv, D, block = 4, 8, 2, 64, 16
    seq_lens = torch.tensor([48, 33, 16, 50], device="cuda", dtype=torch.int32)
    num_blocks = 32
    k_dense = torch.randn(B, 50, Hkv, D, device="cuda", dtype=torch.float16)
    v_dense = torch.randn(B, 50, Hkv, D, device="cuda", dtype=torch.float16)
    kc = torch.randn(num_blocks, block, Hkv, D, device="cuda", dtype=torch.float16)
    vc = torch.randn(num_blocks, block, Hkv, D, device="cuda", dtype=torch.float16)
    bt = torch.zeros(B, 8, device="cuda", dtype=torch.int32)
    for b in range(B):
        nb = (int(seq_lens[b]) + block - 1) // block
        for i in range(nb):
            bid = b * 8 + i
            bt[b, i] = bid
            s = i * block
            e = min(s + block, int(seq_lens[b]))
            kc[bid, : e - s] = k_dense[b, s:e]
            vc[bid, : e - s] = v_dense[b, s:e]
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.float16)

    ratio = Hq // Hkv
    ref = torch.zeros(B, Hq, D, device="cuda", dtype=torch.float16)
    for b in range(B):
        seq = int(seq_lens[b])
        kk = k_dense[b, :seq].permute(1, 0, 2).unsqueeze(0).repeat_interleave(ratio, dim=1)
        vv = v_dense[b, :seq].permute(1, 0, 2).unsqueeze(0).repeat_interleave(ratio, dim=1)
        ref[b] = F.scaled_dot_product_attention(
            q[b].unsqueeze(0).unsqueeze(2), kk, vv, is_causal=False
        ).squeeze(0).squeeze(1)
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


def test_batched_decode_kernel_matches_reference():
    from tilegen.layout.layout import PagedKVLayout
    from tilegen.ops.paged_decode import paged_decode_forward

    q, kc, vc, bt, seq_lens, Hkv, block, ref = _paged_inputs()
    layout = PagedKVLayout(block_size=block, num_kv_heads=Hkv, head_dim=q.shape[-1])
    out = paged_decode_forward(q, kc, vc, bt, seq_lens, layout)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2


def test_decode_codegen_compiles_and_runs():
    from tilegen.backend import ensure_backends_registered, get_backend
    from tilegen.ops.attention import AttentionSpec, build_decode_vtm
    from tilegen.target.spec import get_target

    ensure_backends_registered()
    spec = AttentionSpec(
        batch=4, num_heads=8, num_kv_heads=2, head_dim=64,
        seq_len=64, num_kv_blocks=32, page_size=16, dtype="bf16",
    )
    kernel = build_decode_vtm(spec, get_target("sm_90a"))
    src = get_backend("triton").lower(kernel, get_target("sm_90a"))
    mod = _load_module(src, "gen_decode")

    q, kc, vc, bt, seq_lens, Hkv, block, ref = _paged_inputs()
    out = mod.paged_decode_forward(q, kc, vc, bt, seq_lens, Hkv, block)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2


def test_decode_codegen_is_single_launch():
    """The generated launcher must handle B>1 requests in one call (grid=num_reqs)."""
    from tilegen.backend import ensure_backends_registered, get_backend
    from tilegen.ops.attention import AttentionSpec, build_decode_vtm
    from tilegen.target.spec import get_target

    ensure_backends_registered()
    spec = AttentionSpec(
        batch=4, num_heads=8, num_kv_heads=2, head_dim=64,
        seq_len=64, num_kv_blocks=32, page_size=16, dtype="bf16",
    )
    kernel = build_decode_vtm(spec, get_target("sm_90a"))
    src = get_backend("triton").lower(kernel, get_target("sm_90a"))
# The kernel dispatches with a 1-D grid over requests: assert the source
    # uses ``tl.program_id(0)`` as the request index and a single launch.
    assert "r = tl.program_id(0)" in src
    assert "for blk in range(0, num_blocks)" in src
    assert "bid = tl.load(block_tables + r * MAX_BLOCKS + blk)" in src
