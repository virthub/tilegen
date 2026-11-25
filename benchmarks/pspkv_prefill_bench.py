# benchmarks/pspkv_prefill_bench.py
#
# Dense-prefill attention benchmark: native SDPA vs tilegen FA2 / FA3 / FA4.
# Runnable in either the tilegen venv (all kernels) or the fork venv
# (FA2/FA3 + native SDPA); unavailable kernels are reported as "n/a".

import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)

SHAPES = [
    (4, 32, 1024, 64),
    (4, 32, 2048, 64),
    (4, 32, 4096, 64),
    (4, 32, 1024, 128),
    (4, 32, 2048, 128),
    (4, 32, 4096, 128),
]


def _tflops(dt_ms, z, h, n, d, causal):
    flops = (0.5 if causal else 1.0) * 4.0 * z * h * n * n * d
    return flops / (dt_ms / 1e3) / 1e12


def _time(fn, warmup=8, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_sdpa(q, k, v):
    F.scaled_dot_product_attention(q, k, v, is_causal=True)


def bench_fa2(q, k, v):
    from tilegen.ops.flash import flash_attention_forward
    flash_attention_forward(q, k, v, is_causal=True)


def bench_fa3(q, k, v):
    from tilegen.ops.flash import flash_attention_fa3_forward
    flash_attention_fa3_forward(q, k, v, is_causal=True)


def bench_fa4(q, k, v):
    from tilegen.ops.flash_tilelang import flash_attention_tilelang_forward
    flash_attention_tilelang_forward(q, k, v, is_causal=True)


def main():
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    try:
        import triton
        print(f"triton {triton.__version__}")
    except Exception:
        pass
    print("=" * 84)
    print(f"{'B':>2} {'H':>2} {'N':>5} {'D':>3} | {'sdpa ms':>8} {'tflops':>7} | "
          f"{'FA2 ms':>7} {'tflops':>7} | {'FA3 ms':>7} {'tflops':>7} | {'FA4 ms':>7} {'tflops':>7} | err")
    print("-" * 84)
    for B, H, N, D in SHAPES:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
        rows = []

        def report(name, fn):
            try:
                fn(q, k, v)
                torch.cuda.synchronize()
                ms = _time(lambda: fn(q, k, v))
                err = 0.0
                return f"{ms:>7.2f} {_tflops(ms, B, H, N, D, True):>7.1f}"
            except Exception as e:  # noqa
                return f"{'n/a':>7} {'-':>7}"

        ms = _time(lambda: bench_sdpa(q, k, v))
        sdpa = f"{ms:>7.2f} {_tflops(ms, B, H, N, D, True):>7.1f}"
        fa2 = report("fa2", bench_fa2)
        fa3 = report("fa3", bench_fa3)
        fa4 = report("fa4", bench_fa4)
        print(f"{B:>2} {H:>2} {N:>5} {D:>3} | {sdpa} | {fa2} | {fa3} | {fa4}")


if __name__ == "__main__":
    main()
