# benchmarks/pspkv_tilegen_bench.py
#
# Decode micro-benchmark: vllm-pspkv PSP_KV fused kernels vs the TILEGEN_PSPKV
# backend (dequant + SDPA) on identical synthetic PSP-KV byte caches.
# Requires the fork venv + fork source, e.g.:
#   TILEGEN_BENCH_PYTHON=/root/workspace/vllm-pspkv/.venv34/bin/python \
#   TILEGEN_BENCH_PYTHONPATH=/root/workspace/vllm-pspkv:$PWD/src \
#     ./run.sh --benchmark pspkv_tilegen_bench [--fmt e4m3|int8] [--batch N] [--ctx N]

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

try:
    from vllm.v1.attention.backends.psp_kv_attn import (  # noqa: E402
        PSPKVAttentionImpl,
        PSPKVAttentionMetadata,
    )
    from tilegen.integrations.vllm_pspkv.attention_backend import (  # noqa: E402
        TileGenPSPKVAttentionImpl,
        TileGenPSPKVAttentionMetadata,
        _dequant_prefix_torch,
    )
except ImportError as exc:
    print(f"pspkv_tilegen_bench requires the vllm-pspkv fork environment: {exc}")
    print("Run via run.sh with the fork venv, e.g.:")
    print("  TILEGEN_BENCH_PYTHON=/root/workspace/vllm-pspkv/.venv34/bin/python \\")
    print("  TILEGEN_BENCH_PYTHONPATH=/root/workspace/vllm-pspkv:$PWD/src \\")
    print("    ./run.sh --benchmark pspkv_tilegen_bench --fmt e4m3 --batch 8 --ctx 1024")
    sys.exit(1)

BLOCK = 128
HEAD = 64
NUM_GROUPS = HEAD // 16


def _build_bench_case(device, fmt_code, B, ctx, H_kv=8, H_q=8):
    """Generalized synthetic cache builder (batch B, context ctx)."""
    nblk = (ctx + BLOCK - 1) // BLOCK
    num_blocks = B * nblk
    k_bytes = BLOCK * HEAD
    v_bytes = BLOCK * HEAD
    page = H_kv * (k_bytes + v_bytes)
    cache = torch.zeros(num_blocks * page, dtype=torch.uint8, device=device)
    k_offsets = torch.arange(num_blocks, dtype=torch.int32, device=device) * page
    v_offsets = k_offsets + H_kv * k_bytes
    block_offsets = torch.cat([k_offsets, v_offsets]).contiguous()

    prec_code = 4 if fmt_code == 4 else (fmt_code | (fmt_code << 2))
    prec = torch.full((num_blocks,), prec_code, dtype=torch.uint8, device=device)
    scales = torch.ones(2 * num_blocks, H_kv, NUM_GROUPS, device=device)

    torch.manual_seed(0)
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(B, nblk)
    query = torch.randn(B, H_q, HEAD, device=device, dtype=torch.float16)
    key = torch.randn(B, H_kv, HEAD, device=device).half()
    value = torch.randn(B, H_kv, HEAD, device=device).half()

    cache_flat = cache.view(-1)
    for b in range(B):
        if fmt_code == 1:
            kt = (torch.randint(0, 8, (ctx, H_kv, HEAD), device=device) - 4).float() * 0.5
            vt = (torch.randint(0, 8, (ctx, H_kv, HEAD), device=device) - 4).float() * 0.5
            k_bytes_t = kt.to(torch.float8_e4m3fn).view(torch.uint8)
            v_bytes_t = vt.to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            kt = (torch.randint(0, 16, (ctx, H_kv, HEAD), device=device) - 8).float()
            vt = (torch.randint(0, 16, (ctx, H_kv, HEAD), device=device) - 8).float()
            k_bytes_t = kt.to(torch.int8).view(torch.uint8)
            v_bytes_t = vt.to(torch.int8).view(torch.uint8)
        for blk in range(nblk):
            bid = int(block_table[b, blk].item())
            lo = blk * BLOCK
            hi = min(ctx, lo + BLOCK)
            n = hi - lo
            rows = (torch.arange(n, device=device)[:, None] * H_kv
                    + torch.arange(H_kv, device=device)[None, :])
            k_start = int(k_offsets[bid]) + rows * HEAD
            v_start = int(v_offsets[bid]) + rows * HEAD
            offs_d = torch.arange(HEAD, device=device)
            for h in range(H_kv):
                cache_flat[k_start[:, h].reshape(-1, 1) + offs_d] = k_bytes_t[lo:hi, h]
                cache_flat[v_start[:, h].reshape(-1, 1) + offs_d] = v_bytes_t[lo:hi, h]

    slot_mapping = torch.stack(
        [block_table[b, (seq_lens[b].item() - 1) // BLOCK] * BLOCK
         + (seq_lens[b].item() - 1) % BLOCK for b in range(B)]
    ).to(torch.int64)
    return dict(cache=cache, block_offsets=block_offsets, prec=prec, scales=scales,
                block_table=block_table, seq_lens=seq_lens, query=query, key=key,
                value=value, slot_mapping=slot_mapping, num_blocks=num_blocks,
                nblk=nblk)


def _meta_ns(case):
    B = case["seq_lens"].shape[0]
    return type("M", (), dict(
        num_reqs=B, num_actual_tokens=B, seq_lens=case["seq_lens"],
        context_lens=case["seq_lens"], block_tables=case["block_table"],
        block_table_tensor=case["block_table"], precision_table=case["prec"],
        scale_table=case["scales"], block_offsets=case["block_offsets"],
        slot_mapping=case["slot_mapping"], format_generation=1, max_query_len=1,
        query_start_loc=torch.arange(B + 1, dtype=torch.int32, device="cuda"),
        query_start_loc_cpu=torch.arange(B + 1, dtype=torch.int32),
        num_decode_tokens=B,
    ))()


def _layer():
    return type("L", (), dict(num_kv_heads=8, num_kv_heads_per_partition=8,
                              head_size=HEAD, layer_name="model.layers.0"))()


def _time(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main(argv=None):
    ap = argparse.ArgumentParser(description="PSP_KV kernels vs TILEGEN_PSPKV decode sweep")
    ap.add_argument("--fmt", choices=["e4m3", "int8"], help="precision (default: both)")
    ap.add_argument("--batch", type=int, help="batch size (default: 1,8,32,64)")
    ap.add_argument("--ctx", type=int, help="context length (default: 256,1024,2048)")
    args = ap.parse_args(argv)

    fmts = [(1, "e4m3"), (4, "int8")]
    if args.fmt:
        fmts = [f for f in fmts if f[1] == args.fmt]
    batches = [1, 8, 32, 64] if args.batch is None else [args.batch]
    ctxs = [256, 1024, 2048] if args.ctx is None else [args.ctx]

    device = "cuda"
    scale = 1.0 / HEAD ** 0.5
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"triton {__import__('triton').__version__}")
    print("=" * 104)
    print(f"{'fmt':>5} {'batch':>5} {'ctx':>5} | {'PSP_KV ms':>10} {'TG_PSPKV ms':>11} "
          f"{'ratio':>7} | {'TG_TRITON ms':>12} {'ratio':>7} | {'dequant ms':>10}")
    print("-" * 104)
    for fmt_code, fmt_name in fmts:
        for B in batches:
            for ctx in ctxs:
                case = _build_bench_case(device, fmt_code, B, ctx)
                meta = _meta_ns(case)
                layer = _layer()

                fork_impl = PSPKVAttentionImpl(
                    num_heads=8, head_size=HEAD, scale=scale, num_kv_heads=8,
                    alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto")
                out_fork = torch.zeros_like(case["query"])

                def run_fork():
                    fork_impl.forward(layer, case["query"], case["key"], case["value"],
                                      case["cache"], PSPKVAttentionMetadata(common_metadata=meta),
                                      out_fork)

                tg_impl = TileGenPSPKVAttentionImpl(
                    num_heads=8, head_size=HEAD, scale=scale, num_kv_heads=8)
                out_tg = torch.zeros_like(case["query"])

                def run_tg():
                    tg_impl.forward(layer, case["query"], case["key"], case["value"],
                                    case["cache"],
                                    TileGenPSPKVAttentionMetadata(common_metadata=meta),
                                    out_tg)

                t_fork = _time(run_fork)
                t_tg = _time(run_tg)
                torch.cuda.synchronize()
                err = (out_fork - out_tg).abs().max().item()

                # Batched Triton decode kernel (homogeneous precision).
                from tilegen.integrations.vllm_pspkv.attention_backend import _fmt_k, _fmt_v
                from tilegen.layout.layout import PagedKVLayout
                from tilegen.ops.pspkv_decode import pspkv_decode_forward

                p0 = int(case["prec"][0].item())
                layout = PagedKVLayout(block_size=BLOCK, num_kv_heads=8, head_dim=HEAD)

                def run_tg_triton():
                    pspkv_decode_forward(
                        case["query"], case["cache"], case["block_table"],
                        case["seq_lens"], case["prec"], case["block_offsets"],
                        case["scales"], layout, _fmt_k(p0), _fmt_v(p0), sm_scale=scale)

                t_tr = _time(run_tg_triton)
                torch.cuda.synchronize()
                out_tr = pspkv_decode_forward(
                    case["query"], case["cache"], case["block_table"],
                    case["seq_lens"], case["prec"], case["block_offsets"],
                    case["scales"], layout, _fmt_k(p0), _fmt_v(p0), sm_scale=scale)
                err_tr = (out_fork - out_tr).abs().max().item()

                # TILEGEN_PSPKV cost breakdown: dequant only vs full step.
                def run_dequant():
                    for r in range(B):
                        _dequant_prefix_torch(
                            case["cache"], case["cache"], case["block_table"],
                            case["prec"], case["scales"], case["block_offsets"],
                            r, int(case["seq_lens"][r].item()), 8, HEAD, BLOCK)

                t_dq = _time(run_dequant, warmup=5, iters=20)
                ratio = t_fork / t_tg if t_tg > 0 else float("inf")
                ratio_tr = t_fork / t_tr if t_tr > 0 else float("inf")
                print(f"{fmt_name:>5} {B:>5} {ctx:>5} | {t_fork:>10.3f} {t_tg:>11.3f} "
                      f"{ratio:>7.2f} | {t_tr:>12.3f} {ratio_tr:>7.2f} | {t_dq:>10.3f}  "
                      f"(err {err:.1e}/{err_tr:.1e})")


if __name__ == "__main__":
    main(sys.argv[1:])
