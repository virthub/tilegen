# tests/test_moe.py
#
# GPU-gated tests for the Mixture-of-Experts forward operator

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_moe_forward_matches_reference():
    import torch.nn.functional as F

    from tilegen.ops.moe import moe_forward

    torch.manual_seed(0)
    T, H, E, I, top_k = 32, 128, 8, 64, 2
    x = torch.randn(T, H, device="cuda", dtype=torch.float16)
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.float16)
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.float16)
    gate_logits = torch.randn(T, E, device="cuda", dtype=torch.float32)

    out = moe_forward(x, w1, w2, gate_logits, top_k=top_k)

    tv, ti = torch.topk(gate_logits, top_k)
    wts = torch.softmax(tv, dim=-1)
    ref = torch.zeros(T, H, device="cuda", dtype=torch.float32)
    for t in range(T):
        for j in range(top_k):
            e = int(ti[t, j])
            w = float(wts[t, j])
            gu = x[t].float() @ w1[e].t().float()
            h = F.silu(gu[:I]) * gu[I:]
            ref[t] += w * (h @ w2[e].t().float())

    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-2, f"relative error {rel}"


def test_moe_topk_indices_match():
    from tilegen.ops.fused import moe_topk_gate

    torch.manual_seed(0)
    logits = torch.randn(16, 8, device="cuda")
    weights, idx = moe_topk_gate(logits, k=2)
    tv, ti = torch.topk(logits, 2, dim=-1)
    assert torch.equal(idx, ti)
    assert torch.allclose(weights, torch.softmax(tv, dim=-1), atol=1e-6)
