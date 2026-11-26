# src/tilegen/ops/flash_tilelang.py
#
# TileLang FA4 flash attention

from __future__ import annotations

import os

if "CUDA_HOME" not in os.environ and os.path.isdir("/usr/local/cuda-12.8"):
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[3, 4, 5],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _flashattn(
    batch, heads, seq_q, seq_kv, dim, is_causal,
    block_M=128, block_N=128, num_stages=1, threads=256, use_bf16=False,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504 # log2(e)
    sm_scale = (1.0 / dim) ** 0.5 # natural-log softmax scale (FA2 M)
    dtype = T.bfloat16 if use_bf16 else T.float16
    accum_dtype = T.float32
    past_len = seq_kv - seq_q

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, seq_q, dim), dtype),
        K: T.Tensor((batch, heads, seq_kv, dim), dtype),
        V: T.Tensor((batch, heads, seq_kv, dim), dtype),
        Output: T.Tensor((batch, heads, seq_q, dim), dtype),
        L: T.Tensor((batch, heads, seq_q), accum_dtype),
        M: T.Tensor((batch, heads, seq_q), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            L_shared = T.alloc_shared([block_M], accum_dtype)
            M_shared = T.alloc_shared([block_M], accum_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            l_out = T.alloc_fragment([block_M], accum_dtype)
            m_out = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_kv, block_N),
                      T.ceildiv((bx + 1) * block_M + past_len, block_N))
                if is_causal else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i + past_len
                        k_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(accum_dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(accum_dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            # L/M mirror FA2/FA3: L = 1/denominator, M = scaled max.
            for i in T.Parallel(block_M):
                l_out[i] = 1.0 / logsum[i]
                m_out[i] = scores_max[i] * sm_scale
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])
            T.copy(l_out, L_shared)
            T.copy(L_shared, L[bz, by, bx * block_M:(bx + 1) * block_M])
            T.copy(m_out, M_shared)
            T.copy(M_shared, M[bz, by, bx * block_M:(bx + 1) * block_M])

    return main


def flash_attention_tilelang_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale=None,
    block_m: int = 128,
    block_n: int = 128,
    num_stages=None,
    num_warps: int = 8,
):
    """TileLang FA4 (tcgen05/TMEM) flash attention.

    Returns ``(out, L, M)`` — the same signature as FA2/FA3 — where ``L`` holds
    the per-row reciprocal softmax denominator and ``M`` the per-row scaled
    maximum (for the backward pass / Split-KV merge).

    Requires ``q_len == kv_len`` (no prefix-cached / cross attention); the
    kernel's ``past_len`` path is not validated for ``q_len != kv_len``.

    ``sm_scale`` is ignored (the kernel uses ``1/sqrt(dim)``).  ``num_stages``
    auto-selects 1 for D>=128 (smem limit) and 2 otherwise.
    """
    del sm_scale, num_warps
    assert q.dim() == 4 and q.shape == k.shape and q.shape[-1] == v.shape[-1]
    B, H, N, D = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if num_stages is None:
        num_stages = 1 if D >= 128 else 2
    use_bf16 = q.dtype == torch.bfloat16

    kernel = _flashattn(B, H, N, N, D, is_causal,
                        block_M=block_m, block_N=block_n,
                        num_stages=num_stages, threads=256, use_bf16=use_bf16)
    out, L, M = kernel(q, k, v)
    return out, L, M


__all__ = ["flash_attention_tilelang_forward"]
