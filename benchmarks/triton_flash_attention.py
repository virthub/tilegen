# benchmarks/triton_flash_attention.py
#
# Benchmark & correctness harness for the Generation-0 Triton FA2 kernel

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
import triton

from tilegen.ops.flash import RECIPE, flash_attention_forward


def _tflops(elapsed_s, z, h, n_ctx, d, causal: bool) -> float:
    factor = 0.5 if causal else 1.0
    flops = factor * 4.0 * z * h * n_ctx * n_ctx * d
    return flops / elapsed_s / 1e12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--head-dim", type=int, default=64, choices=[64, 128])
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    Z, H, N, D = args.batch, args.heads, args.seq_len, args.head_dim
    q = torch.randn((Z, H, N, D), device=dev, dtype=torch.bfloat16)
    k = torch.randn((Z, H, N, D), device=dev, dtype=torch.bfloat16)
    v = torch.randn((Z, H, N, D), device=dev, dtype=torch.bfloat16)

    print(f"device  : {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})")
    print(f"shape   : B={Z} H={H} N={N} D={D}  causal={args.causal}")
    print(f"torch   : {torch.__version__} | triton {triton.__version__}")
    print(f"recipe  : {RECIPE}")

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=args.causal)
    out, _, _ = flash_attention_forward(q, k, v, is_causal=args.causal)
    torch.cuda.synchronize()
    max_err = (out - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out.reshape(-1, D).float(), ref.reshape(-1, D).float(), dim=-1
    ).mean().item()
    print(f"correct : max_abs_err={max_err:.2e}  mean_cos={cos:.6f}")

    for _ in range(5):
        flash_attention_forward(q, k, v, is_causal=args.causal)
    torch.cuda.synchronize()
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        flash_attention_forward(q, k, v, is_causal=args.causal)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tf = _tflops(dt, Z, H, N, D, args.causal)
    print(f"perf    : {dt * 1e3:.3f} ms/iter  {tf:.1f} TFLOPS")

    size = 8192
    a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(50):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    peak = (2.0 * size ** 3) / ((time.perf_counter() - t1) / 50) / 1e12
    print(f"roofline: cuBLAS bf16 peak = {peak:.1f} TFLOPS;  achieved = {100.0 * tf / peak:.1f}%")

    ok = max_err < 5e-2 and cos > 0.99
    print("RESULT  :", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
