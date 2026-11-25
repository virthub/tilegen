# tests/test_vllm_plugin.py
#
# Tests for the vLLM out-of-tree plugin

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

if not vllm.__version__.startswith("0.34"):
    pytest.skip(
        f"tilegen vllm plugin targets vLLM 0.34.x (got {vllm.__version__})",
        allow_module_level=True,
    )

try:
    from tilegen.integrations.vllm.platform_plugin import register  # noqa: F401
except Exception as exc:  # noqa: BLE001 - incompatible vLLM version
    pytest.skip(f"tilegen vllm plugin incompatible with vllm {vllm.__version__}: {exc}",
                allow_module_level=True)

torch = pytest.importorskip("torch")


def test_plugin_register_and_backend_interface():
    from tilegen.integrations.vllm.attention_backend import TileGenAttentionBackend
    from tilegen.integrations.vllm.platform_plugin import register

    register()
    assert TileGenAttentionBackend.get_name() == "TILEGEN"
    assert TileGenAttentionBackend.get_impl_cls().__name__ == "TileGenAttentionImpl"
    assert TileGenAttentionBackend.get_builder_cls().__name__ == "TileGenAttentionMetadataBuilder"
    # The @register_backend("TILEGEN") decorator creates a dynamic enum member.
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert AttentionBackendEnum["TILEGEN"].name == "TILEGEN"


def test_rmsnorm_oot_override_registered():
    import vllm.model_executor.custom_op as co

    from tilegen.integrations.vllm.custom_ops import TileGenRMSNorm
    from tilegen.integrations.vllm.platform_plugin import register

    register()
    # OOT override is keyed by class name in op_registry_oot.
    assert co.op_registry_oot.get("RMSNorm") is TileGenRMSNorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_attention_impl_dense_forward():
    import torch.nn.functional as F

    from tilegen.integrations.vllm.attention_backend import (
        TileGenAttentionImpl,
        TileGenAttentionMetadata,
    )

    impl = TileGenAttentionImpl(num_heads=4, head_size=64, scale=1.0 / 8, num_kv_heads=4)
    T = 128
    q = torch.randn(T, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(T, 4, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(T, 4, 64, device="cuda", dtype=torch.bfloat16)
    meta = TileGenAttentionMetadata(
        num_actual_tokens=T, max_query_len=T,
        query_start_loc=torch.tensor([0, T]), max_seq_len=T,
        seq_lens=torch.tensor([T]), block_table=torch.empty(0),
        slot_mapping=torch.empty(0), causal=True, num_decode_tokens=0,
    )
    out = impl.forward(None, q, k, v, None, meta, None)
    ref = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), is_causal=True,
    )
    assert (out - ref.squeeze(0).transpose(0, 1)).abs().max().item() < 5e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_attention_impl_paged_decode():
    import torch.nn.functional as F

    from tilegen.integrations.vllm.attention_backend import (
        TileGenAttentionImpl,
        TileGenAttentionMetadata,
    )

    torch.manual_seed(0)
    num_heads, num_kv_heads, hs, block_size = 8, 2, 64, 16
    num_reqs, num_blocks = 4, 32
    seq_lens = torch.tensor([48, 33, 16, 50], device="cuda", dtype=torch.int32)

    k_dense = torch.randn(num_reqs, 50, num_kv_heads, hs, device="cuda", dtype=torch.float16)
    v_dense = torch.randn(num_reqs, 50, num_kv_heads, hs, device="cuda", dtype=torch.float16)
    kv_cache = torch.randn(num_blocks, num_kv_heads, block_size, 2 * hs,
                           device="cuda", dtype=torch.float16)
    block_table = torch.zeros(num_reqs, num_blocks, device="cuda", dtype=torch.int32)
    for r in range(num_reqs):
        nb = (int(seq_lens[r]) + block_size - 1) // block_size
        for i in range(nb):
            bid = r * 8 + i
            block_table[r, i] = bid
            s = i * block_size
            e = min(s + block_size, int(seq_lens[r]))
            kv_cache[bid, :, : e - s, :hs] = k_dense[r, s:e].permute(1, 0, 2)
            kv_cache[bid, :, : e - s, hs:] = v_dense[r, s:e].permute(1, 0, 2)

    query = torch.randn(num_reqs, num_heads, hs, device="cuda", dtype=torch.float16)
    output = torch.empty(num_reqs, num_heads, hs, device="cuda", dtype=torch.float16)
    meta = TileGenAttentionMetadata(
        num_actual_tokens=num_reqs, max_query_len=1,
        query_start_loc=torch.arange(num_reqs + 1, device="cuda", dtype=torch.int32),
        max_seq_len=50, seq_lens=seq_lens, block_table=block_table,
        slot_mapping=torch.empty(0, device="cuda", dtype=torch.int64),
        causal=True, num_decode_tokens=num_reqs,
    )
    impl = TileGenAttentionImpl(num_heads=num_heads, head_size=hs, scale=1.0 / 8, num_kv_heads=num_kv_heads)
    out = impl.forward(None, query, None, None, kv_cache, meta, output)

    ratio = num_heads // num_kv_heads
    ref = []
    for r in range(num_reqs):
        seq = int(seq_lens[r])
        k = k_dense[r, :seq].permute(1, 0, 2).unsqueeze(0).repeat_interleave(ratio, dim=1)
        v = v_dense[r, :seq].permute(1, 0, 2).unsqueeze(0).repeat_interleave(ratio, dim=1)
        qr = query[r].unsqueeze(0).unsqueeze(2)
        ref.append(F.scaled_dot_product_attention(qr, k, v, is_causal=False))
    ref = torch.cat(ref, dim=0).squeeze(2)
    assert (out.float() - ref.float()).abs().max().item() < 1e-2
