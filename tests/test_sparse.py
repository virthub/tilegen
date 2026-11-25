# tests/test_sparse.py
#
# GPU-gated tests for sparse attention (Sparse MLA selection core)

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_sparse_attention_dense_reduction():
    import torch.nn.functional as F

    from tilegen.ops.sparse import sparse_attention_forward

    torch.manual_seed(0)
    B, H, N, M, D = 2, 4, 8, 16, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)

    out = sparse_attention_forward(q, k, v, top_k=M)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2


def test_sparse_attention_topk():
    from tilegen.ops.sparse import sparse_attention_forward

    torch.manual_seed(0)
    B, H, N, M, D, top_k = 2, 4, 8, 16, 64, 4
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)

    out = sparse_attention_forward(q, k, v, top_k=top_k)
    scores = torch.einsum("bhnd,bhmd->bhnm", q.float(), k.float()) / (D ** 0.5)
    ts, ti = torch.topk(scores, top_k, dim=-1)
    tw = torch.softmax(ts, dim=-1)
    ref = torch.zeros(B, H, N, D, device="cuda", dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            for n in range(N):
                for j in range(top_k):
                    ref[b, h, n] += tw[b, h, n, j] * v[b, h, ti[b, h, n, j]].float()
    assert (out.float() - ref).abs().max().item() < 5e-2
