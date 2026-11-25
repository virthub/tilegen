# src/tilegen/cli.py
#
# Unified TileGen CLI (dispatched via uv run python -m tilegen.cli)

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional
from tilegen import __version__

_PHASE_MSG = {}


def _load_yaml(path: Path) -> dict:
    import yaml  # type: ignore

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    return data


def _build_attention_config(cfg: dict):
    from tilegen.layout.quant import QuantSpec
    from tilegen.ops.attention import AttentionSpec

    attn = cfg.get("attention", {})
    kv_quant = attn.get("kv_quant")
    if isinstance(kv_quant, dict):
        attn = dict(attn)
        attn["kv_quant"] = QuantSpec.from_dict(kv_quant)
    return AttentionSpec(**{k: v for k, v in attn.items() if k in AttentionSpec.__dataclass_fields__})


def _resolve_target(cfg: dict, args: argparse.Namespace):
    from tilegen.target.loader import load_target, load_target_file

    if "target" in cfg:
        return load_target(cfg["target"])
    if getattr(args, "target", None):
        return load_target(args.target)
    if "name" in cfg and "arch" in cfg:
        # The conf file is itself a TargetSpec profile (conf/target/*.yaml).
        return load_target_file(args.conf)
    return load_target("sm_90a")


def cmd_compile(args: argparse.Namespace) -> int:
    from tilegen.backend import ensure_backends_registered, get_backend
    from tilegen.ops.attention import build_attention_vtm
    from tilegen.runtime.cache import AOTArtifactCache
    from tilegen.target.loader import load_target, load_target_file

    cfg = _load_yaml(Path(args.conf))
    target = _resolve_target(cfg, args)

    backend_key = args.backend_ir or cfg.get("backend", "tilelang")
    variant = args.variant or cfg.get("variant", "auto")
    spec = _build_attention_config(cfg)

    kernel = build_attention_vtm(spec, target, variant=variant)

    ensure_backends_registered()
    backend = get_backend("triton" if backend_key == "triton_tle" else backend_key)
    source = backend.lower(kernel, target)

    cache = AOTArtifactCache()
    record = cache.record_for(kernel, target, backend_key)
    ext = "py"
    artifact_rel = f"artifacts/{record.semantic_id}.{ext}"
    artifact_path = cache.root / artifact_rel
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(source, encoding="utf-8")
    record.artifact = artifact_rel
    cache.store(record)

    if args.dump_ir:
        out = Path(args.dump_ir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(backend.dump_ir(kernel, target), encoding="utf-8")
        print(f"[tilegen] backend IR ({backend.name}) written to {out}")

    if args.emit:
        out = Path(args.emit)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(source, encoding="utf-8")
        print(f"[tilegen] lowered source ({backend.name}) written to {out}")

    print(f"[tilegen] TileGen v{__version__}")
    print(f"[tilegen] target      : {target.name} ({target.arch})")
    print(f"[tilegen] backend     : {backend_key}")
    print(f"[tilegen] variant     : {variant}")
    print(f"[tilegen] kernel      : {kernel.name}")
    print(f"[tilegen] semantic_id : {record.semantic_id}")
    print(f"[tilegen] quant path  : fp4 -> {target.quant_path('fp4')}")
    print(f"[tilegen] artifact    : {artifact_path}")
    if args.verbose:
        print()
        print(kernel.serialize())
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Run the built-in smoke checks (no third-party test runner needed)."""
    from tilegen._selftest import run_selftest

    ok, failures = run_selftest()
    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print(f"[tilegen] selftest passed: {ok} checks")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    try:
        import pytest  # type: ignore  # noqa: F401
    except ImportError:
        print("[tilegen] pytest not installed; falling back to built-in selftest", file=sys.stderr)
        return cmd_selftest(args)
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"])
    return result.returncode


def cmd_phased(args: argparse.Namespace) -> int:
    command = args.command
    msg = _PHASE_MSG.get(command)
    print(f"[tilegen] {msg or ('unknown command: ' + command)}", file=sys.stderr)
    return 2


def cmd_evolve(args: argparse.Namespace) -> int:
    """Run a pluggable self-evolution campaign (requires CUDA)."""
    cfg = _load_yaml(Path(args.conf)) if args.conf else {}
    target = _resolve_target(cfg, args)
    spec = _build_attention_config(cfg)
    strategy = args.strategy or cfg.get("strategy", "forgekernel")
    budget = cfg.get("budget") or {}

    from tilegen.evolve.evaluator import TritonAttentionEvaluator, knobs_to_params
    from tilegen.evolve.kb_manager import FileKnowledgeStore
    from tilegen.evolve.strategies import get_engine
    from tilegen.ops.attention import build_attention_vtm

    initial_kernel = build_attention_vtm(spec, target, variant=cfg.get("variant", "fa2_prefill"))

    if not _require_cuda():
        return 1

    import torch

    B, H, N, D = spec.batch, spec.num_heads, spec.seq_len, spec.head_dim
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    evaluator = TritonAttentionEvaluator(
        q, k, v, is_causal=True, sm_scale=spec.scale,
        reference_params=knobs_to_params(initial_kernel.topology.knobs),
    )
    kb = FileKnowledgeStore(args.kb or "conf/kb")
    engine = get_engine(strategy)

    print(f"[tilegen] evolve: strategy={strategy} target={target.name} budget={budget}")
    best = engine.run_campaign(initial_kernel, target, evaluator, kb, budget=budget)
    print(f"[tilegen] best: {best.candidate_id} "
          f"knobs={best.vtm_ir.topology.knobs.to_dict()} ({best.rationale})")
    return 0


def _require_cuda() -> bool:
    try:
        import torch
    except ImportError:
        print("[tilegen] this command requires PyTorch with CUDA", file=sys.stderr)
        return False
    if not torch.cuda.is_available():
        print(
            "[tilegen] no GPU visible in this process; run on the GPU node, e.g.\n"
            "  ssh root@eco4 'cd ~/workspace/tilegen && "
            "PYTHONPATH=src python -m tilegen.cli <subcommand> …'",
            file=sys.stderr,
        )
        return False
    return True


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the 4-stage utilization & regression gate (stages 3-4 need CUDA)."""
    cfg = _load_yaml(Path(args.conf)) if args.conf else {}
    target = _resolve_target(cfg, args)
    spec = _build_attention_config(cfg)

    from tilegen.ops.verify import verify

    report = verify(spec, target, speed_gate=args.speed_gate)
    print(report.summary())
    return 0 if report.all_passed else 1


def cmd_autotune(args: argparse.Namespace) -> int:
    """Constraint-pruned autotune: prune, profile on GPU, emit best recipe."""
    cfg = _load_yaml(Path(args.conf)) if args.conf else {}
    target = _resolve_target(cfg, args)
    spec = _build_attention_config(cfg)

    from tilegen.autotune import SearchSpace, autotune, prune_configs, save_recipe
    from tilegen.autotune.tuner import TuneConfig

    space = SearchSpace.from_dict(cfg.get("search_space", {}))
    configs = space.expand()
    pruned = prune_configs(configs, spec, target, quant=spec.kv_quant)
    print(f"[tilegen] autotune: {len(configs)} candidates -> {len(pruned)} pruned")

    if not _require_cuda():
        return 1

    import torch

    from tilegen.ops.flash import flash_attention_forward

    B, H, N, D = args.batch, args.heads, args.seq_len, args.head_dim
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def profile(cfg: TuneConfig) -> float:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        flash_attention_forward(
            q, k, v, is_causal=True,
            block_m=cfg.tile_m, block_n=cfg.tile_n,
            num_warps=cfg.num_warps, num_stages=cfg.pipeline_stages,
        )
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / 1e3  # seconds

    results = autotune(pruned, profile, warmup=2, iters=5)
    for r in results[:5]:
        print(f"[tilegen]   {r.config.tile_m}x{r.config.tile_n}x{r.config.tile_k} "
              f"stages={r.config.pipeline_stages} warps={r.config.num_warps}: "
              f"{r.device_time_us:.1f} us")

    best = results[0]
    out = Path(args.out) if args.out else Path("conf/kb/autotuned.json")
    save_recipe(out, target, best, extra={"seq_len": N, "head_dim": D})
    print(f"[tilegen] best: {best.config.to_dict()} ({best.device_time_us:.1f} us)")
    print(f"[tilegen] recipe written to {out}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Run the Generation-0 FA2 benchmark (requires CUDA)."""
    try:
        import torch
    except ImportError:
        print("[tilegen] bench requires PyTorch with CUDA", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print(
            "[tilegen] no GPU visible in this process; run on the GPU node, e.g.\n"
            "  ssh root@eco4 'cd ~/workspace/tilegen && "
            "PYTHONPATH=src python -m tilegen.cli bench'",
            file=sys.stderr,
        )
        return 1

    import time

    import torch.nn.functional as F

    from tilegen.ops.flash import RECIPE, flash_attention_forward

    torch.manual_seed(0)
    B, H, N, D = args.batch, args.heads, args.seq_len, args.head_dim
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=args.causal)
    out, _, _ = flash_attention_forward(q, k, v, is_causal=args.causal)
    torch.cuda.synchronize()
    err = (out - ref).abs().max().item()

    for _ in range(5):
        flash_attention_forward(q, k, v, is_causal=args.causal)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        flash_attention_forward(q, k, v, is_causal=args.causal)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters
    factor = 0.5 if args.causal else 1.0
    tf = factor * 4.0 * B * H * N * N * D / dt / 1e12

    print(f"[tilegen] device   : {torch.cuda.get_device_name(0)}")
    print(f"[tilegen] shape    : B={B} H={H} N={N} D={D} causal={args.causal}")
    print(f"[tilegen] recipe   : {RECIPE}")
    print(f"[tilegen] correct  : max_abs_err={err:.2e}")
    print(f"[tilegen] perf     : {dt * 1e3:.3f} ms/iter  {tf:.1f} TFLOPS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tilegen", description="TileGen CLI")
    p.add_argument("--version", action="version", version=f"tilegen {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile", help="AOT-compile a VTM kernel & lower to backend source")
    c.add_argument("--conf", required=True, help="YAML config (target + attention spec)")
    c.add_argument("--target", help="override target name")
    c.add_argument("--backend-ir", choices=["tilelang", "triton_tle", "triton"], help="lowering backend")
    c.add_argument("--variant", choices=["auto", "fa1", "fa2_prefill", "fa2_decode", "fa3", "fa4"], help="FA generation")
    c.add_argument("--dump-ir", help="write backend intermediate IR to this path")
    c.add_argument("--emit", help="write lowered backend source to this path")
    c.add_argument("--verbose", action="store_true")

    sub.add_parser("test", help="run the pytest suite (falls back to selftest)")
    sub.add_parser("selftest", help="run built-in smoke checks")

    e = sub.add_parser("evolve", help="pluggable self-evolution (CUDA)")
    e.add_argument("--conf", help="YAML config (target + attention spec + budget)")
    e.add_argument("--target", help="override target name")
    e.add_argument("--strategy", help="synthesis strategy (forgekernel/evolutionary/goal_loop)")
    e.add_argument("--kb", help="knowledge-store directory (default conf/kb)")

    a = sub.add_parser("autotune", help="constraint-pruned autotune (CUDA)")
    a.add_argument("--conf", help="YAML config (search space + target)")
    a.add_argument("--target", help="override target name")
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--heads", type=int, default=32)
    a.add_argument("--seq-len", type=int, default=2048)
    a.add_argument("--head-dim", type=int, default=64)
    a.add_argument("--out", help="recipe output path")

    v = sub.add_parser("verify", help="4-stage utilization & regression gate")
    v.add_argument("--conf", help="YAML config (target + attention spec)")
    v.add_argument("--target", help="override target name")
    v.add_argument("--speed-gate", type=float, default=0.9, help="min speedup vs native SDPA")

    b = sub.add_parser("bench", help="benchmark the Generation-0 FA2 kernel (CUDA)")
    b.add_argument("--causal", action="store_true")
    b.add_argument("--batch", type=int, default=4)
    b.add_argument("--heads", type=int, default=32)
    b.add_argument("--seq-len", type=int, default=4096)
    b.add_argument("--head-dim", type=int, default=64)
    b.add_argument("--iters", type=int, default=100)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            return cmd_compile(args)
        if args.command == "test":
            return cmd_test(args)
        if args.command == "selftest":
            return cmd_selftest(args)
        if args.command == "bench":
            return cmd_bench(args)
        if args.command == "verify":
            return cmd_verify(args)
        if args.command == "autotune":
            return cmd_autotune(args)
        if args.command == "evolve":
            return cmd_evolve(args)
        return cmd_phased(args)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"[tilegen] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
