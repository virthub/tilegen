# benchmarks/pspkv_prefill_quant.py
#
# Quantify the fresh-prefill attention swap: vllm's flash_attn_varlen_func
# (what the fork's PSP_KV prefill uses) vs tilegen's dense_attention (FA4
# cascade), on varlen batches of fresh prefill (q_len == kv_len, causal).
# Run in the fork venv:
#   TILEGEN_BENCH_PYTHON=<fork-venv-python> TILEGEN_BENCH_PYTHONPATH=<fork-root>:$PWD/src \
#     ./run.sh --benchmark pspkv_prefill_quant [--batch N] [--seq-len N] [--head-dim N]

import argparse
import time

import torch

from tilegen.attention import dense_attention, tilelang_available
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


def _time(fn, warmup=8, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=32)
    args = ap.parse_args()

    B, H, N, D = args.batch, args.heads, args.seq_len, args.head_dim
    torch.manual_seed(0)
    q = torch.randn(B * N, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_q = (torch.arange(B + 1, device="cuda", dtype=torch.int32) * N).to(torch.int32)
    scale = 1.0 / D ** 0.5

    # vllm FlashAttention varlen (the fork's fresh-prefill attention)
    def run_fa():
        flash_attn_varlen_func(q, k, v, N, cu_q, N, cu_q,
                               softmax_scale=scale, causal=True, num_splits=0)

    t_fa = _time(run_fa)

    # tilegen dense_attention per request (FA4 cascade); q_len == kv_len,
    # causal, so the FA2/FA3/FA4 kernels apply directly.
    def run_tg():
        for r in range(B):
            q4 = q[r * N:(r + 1) * N].transpose(0, 1).unsqueeze(0)   # [1,H,N,D]
            k4 = k[r * N:(r + 1) * N].transpose(0, 1).unsqueeze(0)
            v4 = v[r * N:(r + 1) * N].transpose(0, 1).unsqueeze(0)
            dense_attention(q4, k4, v4, is_causal=True, scale=scale)

    t_tg = _time(run_tg)

    ref = flash_attn_varlen_func(q, k, v, N, cu_q, N, cu_q,
                                 softmax_scale=scale, causal=True, num_splits=0)
    out_tg = torch.zeros_like(q)
    for r in range(B):
        q4 = q[r * N:(r + 1) * N].transpose(0, 1).unsqueeze(0)
        k4 = k[r * N:(r + 1) * N].transpose(0, 1).unsqueeze(0)
        v4 = v[r * N:(r + 1) * N].transpose(0, 1).unsqueeze(0)
        out_tg[r * N:(r + 1) * N] = dense_attention(
            q4, k4, v4, is_causal=True, scale=scale)[0].transpose(0, 1)
    torch.cuda.synchronize()
    err = (out_tg - ref).abs().max().item()

    print(f"B={B} H={H} N={N} D={D} | flash_attn_varlen: {t_fa:.3f} ms | "
          f"tilegen dense_attention: {t_tg:.3f} ms | speedup {t_fa / t_tg:.2f}x | "
          f"err {err:.2e} | tilelang_available={tilelang_available()}")


if __name__ == "__main__":
    main()
