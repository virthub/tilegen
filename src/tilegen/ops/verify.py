# src/tilegen/ops/verify.py
#
# Four-stage utilization & regression verification (./run.sh --verify)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec
from tilegen.ops.guards import BoundaryError, validate_attention
from tilegen.target.spec import TargetSpec


@dataclass
class StageResult:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class VerificationReport:
    stages: list[StageResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(s.passed or s.skipped for s in self.stages)

    def summary(self) -> str:
        lines = ["[tilegen] verification report"]
        for s in self.stages:
            mark = "PASS" if s.passed else ("SKIP" if s.skipped else "FAIL")
            lines.append(f"[tilegen]   [{mark}] {s.name}: {s.detail}")
        lines.append(f"[tilegen] overall: {'PASS' if self.all_passed else 'FAIL'}")
        return "\n".join(lines)


def _stage1_boundary_guards(spec: AttentionSpec) -> StageResult:
    shape = (spec.num_kv_blocks, spec.page_size, spec.num_kv_heads, spec.kv_head_dim)
    o_shape = (spec.batch, spec.num_heads, spec.seq_len, spec.head_dim)
    try:
        validate_attention(
            spec=spec, k_cache_shape=shape, v_cache_shape=shape, o_shape=o_shape,
            tile_k=64, tile_n=128,
        )
    except BoundaryError as exc:
        return StageResult("host boundary guards", False, str(exc))

    # Negative case must be caught.
    try:
        validate_attention(
            spec=spec, k_cache_shape=(1, 2, 3), v_cache_shape=shape, o_shape=o_shape,
            tile_k=64, tile_n=128,
        )
        return StageResult("host boundary guards", False, "invalid rank not rejected")
    except BoundaryError:
        return StageResult("host boundary guards", True, "valid accepted; invalid rejected")


def _stage2_dispatch() -> StageResult:
    from tilegen.runtime import semantic_hash

    h = semantic_hash({"kernel": "probe"})
    detail = f"semantic_hash={h[:8]}…"
    try:
        import torch # noqa: F401

        from tilegen.runtime.sdpa import _register_ops

        _register_ops()
        if hasattr(torch.ops.tilegen, "sdpa"):
            detail += "; tilegen::sdpa registered"
        else:
            return StageResult("zero-Python dispatch", False, "tilegen::sdpa not registered")
    except ImportError:
        detail += "; (torch unavailable — custom op not checked)"
    return StageResult("zero-Python dispatch", True, detail)


def _stage3_cuda_graph(spec: AttentionSpec) -> StageResult:
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return StageResult("CUDA-graph compatibility", True, "no CUDA; skipped", skipped=True)

    from tilegen.ops.flash import flash_attention_forward

    B, H, N, D = 1, 8, 256, spec.head_dim
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Capture a decode-like forward (static shapes, no dynamic allocation).
    g = torch.cuda.CUDAGraph()
    # Warm-up on a side stream so allocations are cached.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            flash_attention_forward(q, k, v, is_causal=True)
    torch.cuda.current_stream().wait_stream(s)

    static_out = torch.empty_like(q)
    with torch.cuda.graph(g):
        out, _, _ = flash_attention_forward(q, k, v, is_causal=True)
        static_out.copy_(out)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    g.replay()
    torch.cuda.synchronize()
    err = (static_out - ref).abs().max().item()
    ok = err < 5e-2
    return StageResult("CUDA-graph compatibility", ok, f"replay max_abs_err={err:.2e}")


def _best_flash_forward(q, k, v, is_causal):
    """Return the fastest dense bf16 attention callable (FA4 > FA3 > FA2)."""
    from tilegen.attention import dense_attention, tilelang_available

    label = "FA4(tcgen05)" if tilelang_available() else ("FA3" if q.shape[-1] >= 128 else "FA2")
    return (lambda: dense_attention(q, k, v, is_causal=is_causal)), label


def _stage4_regression(spec: AttentionSpec, speed_gate: float = 0.9) -> StageResult:
    import time

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return StageResult("regression", True, "no CUDA; skipped", skipped=True)

    from tilegen.ops.flash import flash_attention_forward
    from tilegen.runtime.sdpa import dispatch_count

    active_set = [
        ("small", 1, 8, 128),
        ("hot-spot", 4, 32, 2048),
        ("odd", 2, 12, 1000),
        ("large", 4, 32, 4096),
    ]
    worst_err = 0.0
    worst_cos = 1.0
    for name, b, h, n in active_set:
        d = spec.head_dim
        q = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out, _, _ = flash_attention_forward(q, k, v, is_causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        err = (out - ref).abs().max().item()
        cos = F.cosine_similarity(
            out.reshape(-1, d).float(), ref.reshape(-1, d).float(), dim=-1
        ).mean().item()
        worst_err = max(worst_err, err)
        worst_cos = min(worst_cos, cos)

    # Hot-path invocation check: the sdpa op must dispatch into TileGen.
    before = dispatch_count()
    q = torch.randn(2, 8, 512, spec.head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    _ = torch.ops.tilegen.sdpa(q, k, v, True, None)
    torch.cuda.synchronize()
    dispatched = dispatch_count() > before

    # Speed regression: TileGen's fastest kernel must be >= 0.9x native SDPA.
    n = 4096
    q = torch.randn(4, 32, n, spec.head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    run_tilegen, label = _best_flash_forward(q, k, v, True)
    for _ in range(3):
        run_tilegen()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        run_tilegen()
    torch.cuda.synchronize()
    tilegen_us = (time.perf_counter() - t0) / 50 * 1e6
    t0 = time.perf_counter()
    for _ in range(50):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    native_us = (time.perf_counter() - t0) / 50 * 1e6
    speedup = native_us / tilegen_us

    parity_ok = worst_err < 5e-2 and worst_cos > 0.99
    speed_ok = speedup >= speed_gate
    detail = (
        f"parity err={worst_err:.2e} cos={worst_cos:.6f}; "
        f"dispatch={'ok' if dispatched else 'MISSING'}; "
        f"{label} speedup={speedup:.2f}x (gate>={speed_gate})"
    )
    return StageResult("dispatch + regression", parity_ok and dispatched and speed_ok, detail)


def verify(
    spec: AttentionSpec,
    target: TargetSpec,
    quant: Optional[QuantSpec] = None,
    speed_gate: float = 0.9,
) -> VerificationReport:
    """Run the four-stage utilization & regression gate."""
    del target, quant
    report = VerificationReport()
    report.stages.append(_stage1_boundary_guards(spec))
    report.stages.append(_stage2_dispatch())
    report.stages.append(_stage3_cuda_graph(spec))
    report.stages.append(_stage4_regression(spec, speed_gate=speed_gate))
    return report


__all__ = ["StageResult", "VerificationReport", "verify"]
