# tests/test_quant_gemm.py
#
# GPU-gated tests for the FP8 quantized GEMM operator

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_fp8_gemm_matches_reference():
    from tilegen.ops.quant_gemm import quant_gemm_forward, quantize_fp8

    torch.manual_seed(0)
    M, N, K = 256, 256, 128
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    a_q, a_s = quantize_fp8(a)
    b_q, b_s = quantize_fp8(b)

    c = quant_gemm_forward(a_q, b_q, a_s, b_s, out_dtype=torch.bfloat16)
    ref = (a_q.float() * a_s) @ (b_q.float() * b_s)
    rel = (c.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-2, f"relative error {rel}"


def test_quantize_fp8_roundtrip_scale():
    from tilegen.ops.quant_gemm import quantize_fp8

    x = torch.tensor([[1.0, -2.0], [3.0, 4.0]], device="cuda", dtype=torch.float32)
    x_q, scale = quantize_fp8(x)
    assert x_q.dtype == torch.float8_e4m3fn
    assert scale.item() > 0
    # Dequantized values should approximate the original.
    approx = (x_q.float() * scale)
    assert (approx - x).abs().max().item() < 1.0


def test_w4a16_gemm_matches_reference():
    from tilegen.ops.quant_gemm import dequant_w4, quantize_w4, w4a16_gemm_forward

    torch.manual_seed(1)
    M, N, K = 128, 256, 256
    group_size = 64
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(K, N, device="cuda", dtype=torch.float16)
    w_packed, w_scale = quantize_w4(w, group_size=group_size)
    w_ref = dequant_w4(w_packed, w_scale, group_size=group_size)

    c = w4a16_gemm_forward(a, w_packed, w_scale, group_size=group_size)
    ref = a.float() @ w_ref
    rel = (c.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 2e-2, f"relative error {rel}"


def test_fp4_gemm_matches_reference():
    from tilegen.ops.quant_gemm import dequant_fp4, fp4_gemm_forward, quantize_fp4

    torch.manual_seed(2)
    M, N, K, block = 128, 256, 256, 16
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(K, N, device="cuda", dtype=torch.float16)
    w_packed, w_scale = quantize_fp4(w, block_size=block)
    w_ref = dequant_fp4(w_packed, w_scale, block_size=block)

    c = fp4_gemm_forward(a, w_packed, w_scale, block_size=block)
    ref = a.float() @ w_ref.float()
    rel = (c.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-2, f"relative error {rel}"
    # Roundtrip quantization error should be bounded by 4-bit precision (~30%).
    rerr = (w_ref.float() - w.float()).abs().max().item() / w.float().abs().max().item()
    assert rerr < 0.3, f"roundtrip rel error {rerr}"
