# tests/test_paged.py
#
# GPU-gated tests for the paged-KV attention (block-table gather)

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_paged_attention_matches_dense_reference():
    from tilegen.layout.layout import PagedKVLayout
    from tilegen.ops.paged import paged_attention_forward

    torch.manual_seed(0)
    B, H_q, H_kv, D, block_size = 4, 8, 2, 64, 16
    num_blocks = 32
    seq_lens = torch.tensor([48, 33, 16, 50], device="cuda", dtype=torch.int32)

    k_dense = torch.randn(B, max(seq_lens), H_kv, D, device="cuda", dtype=torch.float16)
    v_dense = torch.randn(B, max(seq_lens), H_kv, D, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(num_blocks, block_size, H_kv, D, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(num_blocks, block_size, H_kv, D, device="cuda", dtype=torch.float16)
    block_tables = torch.zeros(B, num_blocks, device="cuda", dtype=torch.int32)
    for b in range(B):
        nb = (int(seq_lens[b]) + block_size - 1) // block_size
        for i in range(nb):
            bid = b * 8 + i
            block_tables[b, i] = bid
            s = i * block_size
            e = min(s + block_size, int(seq_lens[b]))
            k_cache[bid, : e - s] = k_dense[b, s:e]
            v_cache[bid, : e - s] = v_dense[b, s:e]

    q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.float16)
    layout = PagedKVLayout(block_size=block_size, num_kv_heads=H_kv, head_dim=D)
    out = paged_attention_forward(q, k_cache, v_cache, block_tables, seq_lens, layout)

    import torch.nn.functional as F

    ref = []
    for b in range(B):
        seq = int(seq_lens[b])
        k = k_dense[b, :seq].permute(1, 0, 2).unsqueeze(0).repeat_interleave(H_q // H_kv, dim=1)
        v = v_dense[b, :seq].permute(1, 0, 2).unsqueeze(0).repeat_interleave(H_q // H_kv, dim=1)
        ref.append(F.scaled_dot_product_attention(q[b].unsqueeze(0), k, v, is_causal=False))
    ref = torch.cat(ref, dim=0)
    assert (out.float() - ref.float()).abs().max().item() < 1e-2
