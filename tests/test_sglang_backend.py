# tests/test_sglang_backend.py
#
# Tests for the SGLang integration backend (TileGenAttentionBackend)

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_sglang_backend_imports_without_sglang():
    """The module imports without SGLang installed (TYPE_CHECKING-only imports)."""
    from tilegen.integrations.sglang import TileGenAttentionBackend
    from tilegen.integrations.sglang.attention_backend import _kv_cache

    assert TileGenAttentionBackend is not None
    assert callable(_kv_cache)


def test_kv_cache_interleaved_split():
    """SGLang's K/V are interleaved on dim 1 of the flat token pool."""
    from tilegen.integrations.sglang.attention_backend import _kv_cache

    num_tokens, nkv, dim = 16, 2, 64
    pool = torch.randn(num_tokens, 2, nkv, dim)
    k, v = _kv_cache(pool)
    assert k.shape == (num_tokens, nkv, dim)
    assert v.shape == (num_tokens, nkv, dim)
    assert torch.equal(k, pool[:, 0])
    assert torch.equal(v, pool[:, 1])


def test_kv_cache_separate_layout():
    """Also accepts the older ``k_cache``/``v_cache`` object layout."""
    from tilegen.integrations.sglang.attention_backend import _kv_cache

    kc = torch.randn(16, 2, 64)
    vc = torch.randn(16, 2, 64)
    pool = SimpleNamespace(k_cache=kc, v_cache=vc)
    k, v = _kv_cache(pool)
    assert torch.equal(k, kc) and torch.equal(v, vc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_forward_decode_token_gather_matches_reference():
    """Decode gathers per-request tokens via req_to_token and attends correctly."""
    import torch.nn.functional as F

    from tilegen.integrations.sglang.attention_backend import TileGenAttentionBackend

    torch.manual_seed(0)
    num_reqs, Hq, Hkv, D = 2, 8, 2, 64
    seq_lens = torch.tensor([48, 33], device="cuda", dtype=torch.int32)
    max_ctx = 48
    # Flat token pool [num_tokens, 2, nkv, dim]; token index = req*max_ctx + offset.
    num_tokens = num_reqs * max_ctx
    k_pool = torch.randn(num_tokens, Hkv, D, device="cuda", dtype=torch.float16)
    v_pool = torch.randn(num_tokens, Hkv, D, device="cuda", dtype=torch.float16)
    pool = torch.stack([k_pool, v_pool], dim=1)          # [num_tokens, 2, Hkv, D]

    req_to_token = torch.zeros(num_reqs, max_ctx, dtype=torch.int32, device="cuda")
    for r in range(num_reqs):
        req_to_token[r] = torch.arange(r * max_ctx, (r + 1) * max_ctx, dtype=torch.int32)

    q = torch.randn(num_reqs, Hq, D, device="cuda", dtype=torch.float16)
    layer = SimpleNamespace(kv_cache=pool, scaling=1.0 / 8)
    fb = SimpleNamespace(
        req_pool_indices=torch.arange(num_reqs, device="cuda", dtype=torch.int32),
        seq_lens=seq_lens,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )

    out = TileGenAttentionBackend().forward_decode(q, None, None, layer, fb)

    ratio = Hq // Hkv
    ref = torch.zeros(num_reqs, Hq, D, device="cuda", dtype=torch.float16)
    for r in range(num_reqs):
        seq = int(seq_lens[r])
        toks = req_to_token[r, :seq].long()
        kk = k_pool[toks].permute(1, 0, 2).unsqueeze(0).repeat_interleave(ratio, dim=1)
        vv = v_pool[toks].permute(1, 0, 2).unsqueeze(0).repeat_interleave(ratio, dim=1)
        ref[r] = F.scaled_dot_product_attention(
            q[r].unsqueeze(0).unsqueeze(2), kk, vv, is_causal=False
        ).squeeze(0).squeeze(1)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_forward_extend_prefix_cached_matches_reference():
    """Extend with a cached prefix: new tokens attend to prefix + themselves."""
    import torch.nn.functional as F

    from tilegen.integrations.sglang.attention_backend import TileGenAttentionBackend

    torch.manual_seed(0)
    Hq, Hkv, D = 8, 2, 64
    plen, slen = 32, 16
    num_tokens = 64
    k_pool = torch.randn(num_tokens, Hkv, D, device="cuda", dtype=torch.float16)
    v_pool = torch.randn(num_tokens, Hkv, D, device="cuda", dtype=torch.float16)
    pool = torch.stack([k_pool, v_pool], dim=1)

    req_to_token = torch.zeros(1, plen, dtype=torch.int32, device="cuda")
    req_to_token[0] = torch.arange(plen, dtype=torch.int32, device="cuda")

    q = torch.randn(slen, Hq, D, device="cuda", dtype=torch.float16)
    k = torch.randn(slen, Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(slen, Hkv, D, device="cuda", dtype=torch.float16)
    layer = SimpleNamespace(kv_cache=pool, scaling=1.0 / 8)
    fb = SimpleNamespace(
        req_pool_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        extend_prefix_lens=torch.tensor([plen], device="cuda", dtype=torch.int32),
        extend_seq_lens=torch.tensor([slen], device="cuda", dtype=torch.int32),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )

    out = TileGenAttentionBackend().forward_extend(q, k, v, layer, fb)

    ratio = Hq // Hkv
    k_pre = k_pool[req_to_token[0, :plen].long()]
    v_pre = v_pool[req_to_token[0, :plen].long()]
    k_cat = torch.cat([k_pre, k], dim=0).repeat_interleave(ratio, dim=1)
    v_cat = torch.cat([v_pre, v], dim=0).repeat_interleave(ratio, dim=1)
    q4 = q.transpose(0, 1).unsqueeze(0)
    k4 = k_cat.transpose(0, 1).unsqueeze(0)
    v4 = v_cat.transpose(0, 1).unsqueeze(0)
    ref = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True).squeeze(0).transpose(0, 1)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2
