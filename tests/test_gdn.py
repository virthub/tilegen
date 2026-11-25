# tests/test_gdn.py
#
# GPU-gated tests for the Gated Delta Network (GDN) linear attention

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_gdn_matches_reference():
    from tilegen.ops.gdn import gdn_forward

    torch.manual_seed(0)
    T, D, d = 8, 64, 32
    s = 1.0 / (D ** 0.5)
    x = torch.randn(T, D, device="cuda", dtype=torch.float16)
    wk = (torch.randn(D, d, device="cuda") * s).to(torch.float16)
    wv = (torch.randn(D, d, device="cuda") * s).to(torch.float16)
    wq = (torch.randn(D, d, device="cuda") * s).to(torch.float16)
    wb = (torch.randn(D, 1, device="cuda") * s).to(torch.float16)
    wo = (torch.randn(d, D, device="cuda") * s).to(torch.float16)

    out = gdn_forward(x, wk, wv, wq, wb, wo)

    k = x.float() @ wk.float()
    v = x.float() @ wv.float()
    q = x.float() @ wq.float()
    beta = torch.sigmoid(x.float() @ wb.float())
    o = torch.zeros(T, d, device="cuda")
    S = torch.zeros(d, d, device="cuda")
    for t in range(T):
        S = (1 - beta[t]) * S + beta[t] * torch.einsum("i,j->ij", k[t], v[t])
        o[t] = q[t] @ S
    ref = (o @ wo.float()).to(torch.float16)

    assert (out.float() - ref.float()).abs().max().item() < 1e-2
