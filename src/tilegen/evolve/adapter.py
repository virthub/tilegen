# src/tilegen/evolve/adapter.py
#
# Evolver adaptation layer: bridges external evolvers (LLM agents, remote
# services, third-party search/RL frameworks) onto the in-process evolve
# protocols via JSON proposals, feedback formatting, unified variant names,
# and sync/async remote evaluation

from __future__ import annotations

from typing import Any, Callable, Sequence
from tilegen.evolve.interface import EvalResult, KernelCandidate
from tilegen.evolve.profiler import diagnose
from tilegen.ir.from_dict import vtm_kernel_from_dict
from tilegen.target.spec import TargetSpec

# Canonical variant vocabulary: the schedules namespace plus "sdpa".
CANONICAL_VARIANTS = ("sdpa", "fa1", "fa2_prefill", "fa2_decode", "fa3", "fa4")

# Agent/evaluator aliases -> canonical names (mutation.VARIANT_ORDER etc.).
VARIANT_ALIASES = {
    "sdpa": "sdpa",
    "fa1": "fa1",
    "fa2_bf16": "fa2_prefill",
    "fa2_prefill": "fa2_prefill",
    "fa2_decode": "fa2_decode",
    "fa3": "fa3",
    "fa3_tma": "fa3",
    "fa3_ws": "fa3",
    "fa4": "fa4",
    "fa4_tilelang": "fa4",
}

BACKENDS = ("tilelang", "triton_tle", "triton")


def normalize_variant(variant: str) -> str:
    """Map an alias to the canonical variant name; raise on unknown input."""
    canon = VARIANT_ALIASES.get(variant)
    if canon is None:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(VARIANT_ALIASES)}"
        )
    return canon


def candidate_to_dict(candidate: KernelCandidate) -> dict:
    """Serialize a candidate for transport to an external evolver/evaluator."""
    return {
        "candidate_id": candidate.candidate_id,
        "parent_ids": list(candidate.parent_ids),
        "vtm_ir": candidate.vtm_ir.to_dict(),
        "backend": candidate.backend,
        "code_override": candidate.code_override,
        "rationale": candidate.rationale,
        "variant": candidate.variant,
    }


def candidate_from_dict(data: dict, verify: bool = True) -> KernelCandidate:
    """Build a validated KernelCandidate from a transport payload dict."""
    if not isinstance(data, dict):
        raise TypeError("candidate entry must be a dict")
    variant = normalize_variant(data.get("variant", "fa2_bf16"))
    backend = data.get("backend", "triton")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if "vtm_ir" not in data:
        raise ValueError("candidate entry missing 'vtm_ir'")
    return KernelCandidate(
        candidate_id=data.get("candidate_id", ""),
        parent_ids=list(data.get("parent_ids", [])),
        vtm_ir=vtm_kernel_from_dict(data["vtm_ir"], verify=verify),
        backend=backend,
        code_override=data.get("code_override"),
        rationale=data.get("rationale", ""),
        variant=variant,
    )


def propose_candidates(payload: Any, verify: bool = True) -> list[KernelCandidate]:
    """Accept an external evolver's proposal payload.

    ``payload`` is either a dict like ``{"candidates": [...]}`` or a bare list
    of candidate dicts.  Each entry is validated (variant, backend, VTM IR);
    errors carry the failing candidate index.
    """
    if isinstance(payload, dict):
        items = payload.get("candidates")
    else:
        items = payload
    if not isinstance(items, (list, tuple)):
        raise ValueError("payload must be a dict with 'candidates' or a list of candidate dicts")
    out: list[KernelCandidate] = []
    for i, item in enumerate(items):
        try:
            out.append(candidate_from_dict(item, verify=verify))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"candidate {i} invalid: {exc}") from exc
    return out


def eval_result_from_dict(data: dict) -> EvalResult:
    """Rebuild an EvalResult from a transport payload dict."""
    known = {f for f in EvalResult.__dataclass_fields__}
    payload = {k: v for k, v in data.items() if k in known}
    payload.setdefault("candidate_id", "")
    payload.setdefault("compiled", False)
    payload.setdefault("correct", False)
    return EvalResult(**payload)


def format_feedback(results: Sequence[EvalResult]) -> str:
    """Render evaluation results as agent-readable feedback text.

    Includes the pass/fail status, timing, speedup, error, and profiler-derived
    diagnoses (``profiler.diagnose``) for each candidate.
    """
    lines: list[str] = []
    for r in results:
        status = "PASS" if (r.compiled and r.correct) else "FAIL"
        parts = [f"[{status}] {r.candidate_id}"]
        if r.device_time_us is not None:
            parts.append(f"{r.device_time_us:.1f} us")
        if r.speedup_vs_ref is not None:
            parts.append(f"{r.speedup_vs_ref:.2f}x vs ref")
        if r.max_rel_error is not None:
            parts.append(f"err={r.max_rel_error:.2e}")
        lines.append(" ".join(parts))
        if not (r.compiled and r.correct):
            lines.append(f"    error: {r.error_log or 'evaluation failed'}")
        for d in diagnose(r.telemetry or {}):
            lines.append(f"    diagnosis: {d['symptom']} -> {d['action']}")
    return "\n".join(lines)


def feedback_json(results: Sequence[EvalResult]) -> list[dict]:
    """Structured (JSON-friendly) feedback for programmatic evolvers."""
    out: list[dict] = []
    for r in results:
        out.append(
            {
                "candidate_id": r.candidate_id,
                "compiled": r.compiled,
                "correct": r.correct,
                "device_time_us": r.device_time_us,
                "speedup_vs_ref": r.speedup_vs_ref,
                "max_rel_error": r.max_rel_error,
                "telemetry": dict(r.telemetry or {}),
                "error_log": r.error_log,
                "diagnoses": diagnose(r.telemetry or {}),
            }
        )
    return out


class RemoteKernelEvaluator:
    """A ``KernelEvaluator`` backed by an external transport.

    ``submit`` is a callable ``(candidate_dicts, target_name, sealed_ref_id) ->
    list[EvalResult-compatible dicts]`` — e.g. an HTTP client, a job queue, or a
    subprocess bridge.  Keeps the synchronous protocol expected by the built-in
    engines while the device work happens elsewhere.
    """

    def __init__(
        self,
        submit: Callable[[list[dict], str, str | None], list[dict]],
        profile_reference: Callable[[str, str], dict] | None = None,
    ):
        self._submit = submit
        self._profile_reference = profile_reference

    def evaluate(
        self,
        candidates: Sequence[KernelCandidate],
        target: TargetSpec,
        sealed_ref_id: str | None = None,
    ) -> list[EvalResult]:
        payload = [candidate_to_dict(c) for c in candidates]
        raw = self._submit(payload, target.name, sealed_ref_id)
        return [eval_result_from_dict(r) for r in (raw or [])]

    def profile_reference(self, sealed_ref_id: str, target: TargetSpec) -> dict:
        if self._profile_reference is None:
            return {}
        return self._profile_reference(sealed_ref_id, target.name)


class AsyncRemoteEvaluator:
    """Async variant of :class:`RemoteKernelEvaluator`.

    ``submit_async`` is an awaitable transport; use ``await evaluator.evaluate(...)``
    from agent harnesses and RL loops that cannot block on device work.
    """

    def __init__(self, submit_async: Callable[[list[dict], str, str | None], Any]):
        self._submit = submit_async

    async def evaluate(
        self,
        candidates: Sequence[KernelCandidate],
        target: TargetSpec,
        sealed_ref_id: str | None = None,
    ) -> list[EvalResult]:
        payload = [candidate_to_dict(c) for c in candidates]
        raw = await self._submit(payload, target.name, sealed_ref_id)
        return [eval_result_from_dict(r) for r in (raw or [])]


__all__ = [
    "CANONICAL_VARIANTS",
    "VARIANT_ALIASES",
    "BACKENDS",
    "normalize_variant",
    "candidate_to_dict",
    "candidate_from_dict",
    "propose_candidates",
    "eval_result_from_dict",
    "format_feedback",
    "feedback_json",
    "RemoteKernelEvaluator",
    "AsyncRemoteEvaluator",
]
