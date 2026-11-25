# src/tilegen/evolve/evaluator.py
#
# Triton attention kernel evaluator

from __future__ import annotations

from typing import Optional, Sequence
from tilegen.evolve.interface import EvalResult, KernelCandidate, KernelEvaluator
from tilegen.ir.ops import VTMKernel
from tilegen.target.spec import TargetSpec


def backend_for_kernel(kernel: VTMKernel) -> str:
    """Map a VTM kernel to its runnable backend: FA4 (tcgen05) → ``tilelang``, else ``triton``."""
    g = kernel.topology.grid
    k = kernel.topology.knobs
    if g.cta_group == 2 or k.exp_emulation_ratio > 0.0:
        return "tilelang"
    return "triton"


def knobs_to_params(knobs) -> dict:
    """Map VTM ``TacticalKnobs`` to ``flash_attention_forward`` parameters."""
    return {
        "block_m": knobs.tile_m,
        "block_n": knobs.tile_n,
        "num_warps": max(1, knobs.tile_m // 32),
        "num_stages": knobs.pipeline_stages,
    }


def _load_module_from_source(source: str, module_name: str):
    """Write ``source`` to a temp file and import it as a module.

    The temp file is intentionally kept on disk: TileLang/Triton ``@jit``
    kernels compile lazily on first call, so the source must still exist when
    the returned callable is first invoked.
    """
    import importlib.util
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".py", prefix="gen_eval_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_telemetry(knobs) -> dict[str, float]:
    """Heuristic stall profile when no hardware profiler is attached.

    Maps tactical knobs to normalized stall symptoms so the profiler-guided
    mutation loop (``diagnose`` -> ``diagnose_fields`` -> ``mutate_knobs(bias)``)
    has input even without ``ncu``/``rocprof``/``msprof``.  The real profiler
    backends replace this.
    """
    t: dict[str, float] = {}
    if knobs.tile_m <= 32 or knobs.tile_n <= 32:
        t["tensorcore_idle"] = 1.0
    if knobs.tile_m >= 256 or knobs.max_registers_per_thread >= 232:
        t["register_spill"] = 1.0
    if knobs.pipeline_stages <= 1:
        t["mbarrier_wait_stall"] = 1.0
    if knobs.exp_emulation_ratio >= 0.5:
        t["sfu_stall"] = 1.0
    return t


class TritonAttentionEvaluator(KernelEvaluator):
    """Evaluate FA2 attention candidates on CUDA."""

    def __init__(
        self,
        q,
        k,
        v,
        is_causal: bool = True,
        sm_scale=None,
        reference_params: Optional[dict] = None,
        iters: int = 20,
    ):
        import torch
        import torch.nn.functional as F

        self.q = q.contiguous()
        self.k = k.contiguous()
        self.v = v.contiguous()
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.iters = iters
        self.reference_params = reference_params or {}

        # Independent correctness baseline (native SDPA).
        self.ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        torch.cuda.synchronize()

        self._ref_time_us: Optional[float] = None
        # Memoize (variant, params) -> device_time_us so re-encountered configs
        # are not recompiled/re-timed across generations.
        self._timing_cache: dict[tuple, float] = {}

    def _forward_fn(self, variant: str):
        if variant == "sdpa":
            import torch.nn.functional as F

            def _sdpa(q, k, v, is_causal=False, sm_scale=None, **params):
                del params
                return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=sm_scale)

            return _sdpa
        if variant == "fa3_tma":
            from tilegen.ops.flash_tma import flash_attention_tma_forward

            return flash_attention_tma_forward
        if variant == "fa3_ws":
            from tilegen.ops.flash import flash_attention_fa3_forward

            return flash_attention_fa3_forward
        if variant == "fa4_tilelang":
            from tilegen.ops.flash_tilelang import flash_attention_tilelang_forward

            return flash_attention_tilelang_forward
        # fa1 and fa2_bf16 both run the FA2 kernel; the FA1 vs FA2 distinction is
        # carried by the candidate's topology knobs (64x64/stages-1 vs 128x128).
        from tilegen.ops.flash import flash_attention_forward

        return flash_attention_forward

    def _time_config(self, params: dict, variant: str = "fa2_bf16") -> float:
        import time

        import torch

        cache_key = (variant, tuple(sorted(params.items())))
        if cache_key in self._timing_cache:
            return self._timing_cache[cache_key]

        fn = self._forward_fn(variant)
        for _ in range(5):
            fn(self.q, self.k, self.v, is_causal=self.is_causal,
               sm_scale=self.sm_scale, **params)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(self.iters):
            fn(self.q, self.k, self.v, is_causal=self.is_causal,
               sm_scale=self.sm_scale, **params)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / self.iters * 1e6
        self._timing_cache[cache_key] = us
        return us

    def _call(self, fn, params) -> "torch.Tensor":
        """Call a forward and return its output tensor.

        FA2/FA3 return ``(out, L, M)``; FA4 (TileLang) returns ``out`` directly.
        """
        result = fn(self.q, self.k, self.v, is_causal=self.is_causal,
                    sm_scale=self.sm_scale, **params)
        return result[0] if isinstance(result, (tuple, list)) else result

    def evaluate(
        self,
        candidates: Sequence[KernelCandidate],
        target: TargetSpec,
        sealed_ref_id: str | None = None,
    ) -> list[EvalResult]:
        import torch

        if self._ref_time_us is None:
            self.profile_reference(sealed_ref_id or "reference", target)
        ref_time = self._ref_time_us or 1.0

        results: list[EvalResult] = []
        for c in candidates:
            params = knobs_to_params(c.vtm_ir.topology.knobs)
            fn = self._forward_fn(c.variant)
            telemetry = synthetic_telemetry(c.vtm_ir.topology.knobs)
            try:
                out = self._call(fn, params)
                torch.cuda.synchronize()
                err = float((out - self.ref_out).abs().max().item())
                t_us = self._time_config(params, variant=c.variant)
                speedup = ref_time / t_us if t_us > 0 else None
                results.append(
                    EvalResult(
                        candidate_id=c.candidate_id,
                        compiled=True,
                        correct=err < 5e-2,
                        device_time_us=t_us,
                        speedup_vs_ref=speedup,
                        max_rel_error=err,
                        telemetry=telemetry,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    EvalResult(
                        candidate_id=c.candidate_id,
                        compiled=False,
                        correct=False,
                        error_log=str(exc),
                        telemetry=telemetry,
                    )
                )
        return results

    def profile_reference(self, sealed_ref_id: str, target: TargetSpec) -> dict:
        del target
        if self._ref_time_us is None:
            self._ref_time_us = self._time_config(self.reference_params)
        return {"sealed_ref_id": sealed_ref_id, "device_time_us": self._ref_time_us}


class GeneratedKernelEvaluator(KernelEvaluator):
    """Evaluate candidates through the *generic flow*: backend-lowered codegen.

    For each candidate, lowers its ``vtm_ir`` via ``get_backend(backend)`` (the
    backend is derived from the VTM topology via :func:`backend_for_kernel`),
    writes the generated source to a temp file, imports it, and times the
    emitted ``{name}_forward`` launcher.  Compiled kernels are memoized by a
    semantic hash of (VTM, target, backend).  The reference is native SDPA.
    """

    def __init__(self, q, k, v, is_causal: bool = True, sm_scale=None, iters: int = 20):
        import torch
        import torch.nn.functional as F

        self.q = q.contiguous()
        self.k = k.contiguous()
        self.v = v.contiguous()
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.iters = iters
        self.ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        torch.cuda.synchronize()
        self._compiled: dict[str, object] = {}
        self._ref_time_us: Optional[float] = None

    def _compile(self, candidate: KernelCandidate, target: TargetSpec):
        from tilegen.backend import ensure_backends_registered, get_backend
        from tilegen.runtime.hash import semantic_hash

        kernel = candidate.vtm_ir
        backend_name = backend_for_kernel(kernel)   # VTM topology is the source of truth
        if candidate.code_override:
            # External evolvers may supply direct Triton/TileLang source; it
            # takes precedence over lowering from the VTM IR.
            key = semantic_hash({"code_override": candidate.code_override, "target": target.to_dict()})
            if key in self._compiled:
                return self._compiled[key]
            src = candidate.code_override
        else:
            key = semantic_hash({"vtm": kernel.to_dict(), "target": target.to_dict(), "backend": backend_name})
            if key in self._compiled:
                return self._compiled[key]
            ensure_backends_registered()
            src = get_backend(backend_name).lower(kernel, target)

        mod = _load_module_from_source(src, f"gen_eval_{key[:12]}")
        fn = getattr(mod, f"{kernel.name}_forward")
        self._compiled[key] = fn
        return fn

    def evaluate(
        self,
        candidates: Sequence[KernelCandidate],
        target: TargetSpec,
        sealed_ref_id: str | None = None,
    ) -> list[EvalResult]:
        import time

        import torch

        if self._ref_time_us is None:
            self.profile_reference(sealed_ref_id or "native-sdpa", target)
        ref_time = self._ref_time_us or 1.0

        results: list[EvalResult] = []
        for c in candidates:
            telemetry = synthetic_telemetry(c.vtm_ir.topology.knobs)
            try:
                fn = self._compile(c, target)
                out = fn(self.q, self.k, self.v, self.is_causal)
                out = out[0] if isinstance(out, (tuple, list)) else out
                torch.cuda.synchronize()
                err = float((out - self.ref_out).abs().max().item())

                for _ in range(5):
                    fn(self.q, self.k, self.v, self.is_causal)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(self.iters):
                    fn(self.q, self.k, self.v, self.is_causal)
                torch.cuda.synchronize()
                t_us = (time.perf_counter() - t0) / self.iters * 1e6
                speedup = ref_time / t_us if t_us > 0 else None
                results.append(
                    EvalResult(
                        candidate_id=c.candidate_id,
                        compiled=True,
                        correct=err < 5e-2,
                        device_time_us=t_us,
                        speedup_vs_ref=speedup,
                        max_rel_error=err,
                        telemetry=telemetry,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    EvalResult(
                        candidate_id=c.candidate_id,
                        compiled=False,
                        correct=False,
                        error_log=str(exc),
                        telemetry=telemetry,
                    )
                )
        return results

    def profile_reference(self, sealed_ref_id: str, target: TargetSpec) -> dict:
        import time

        import torch
        import torch.nn.functional as F

        del target
        if self._ref_time_us is None:
            for _ in range(5):
                F.scaled_dot_product_attention(self.q, self.k, self.v, is_causal=self.is_causal)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(self.iters):
                F.scaled_dot_product_attention(self.q, self.k, self.v, is_causal=self.is_causal)
            torch.cuda.synchronize()
            self._ref_time_us = (time.perf_counter() - t0) / self.iters * 1e6
        return {"sealed_ref_id": sealed_ref_id, "device_time_us": self._ref_time_us}


__all__ = ["TritonAttentionEvaluator", "GeneratedKernelEvaluator", "backend_for_kernel", "knobs_to_params"]
