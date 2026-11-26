# src/tilegen/evolve/interface.py
#
# Common synthesis & self-evolution protocol

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence
from tilegen.ir.ops import VTMKernel
from tilegen.target.spec import TargetSpec

BackendName = Literal["tilelang", "triton_tle", "triton"]

# Strategy plugin identifiers (``./run.sh --evolve --strategy <name>``).
STRATEGIES = (
    "forgekernel", # default: reference-guided self-evolution
    "evolutionary", # population + crossover/mutation (AlphaEvolve/AVO)
    "world_model", # K-Search intrinsic transition/value model
    "goal_loop", # single-episode patch loop (Claude /goal, STARK)
    "rl_rollout", # batched reward environment (Kevin, AutoTriton)
    "schedule_codesign", # typed schedule transforms (CAKE)
    "zero", # the evolved evolver artifact (evolve/zero.py)
    "meta_evolver", # the evolver of the evolver (evolve/meta_evolver.py)
)


@dataclass
class KernelCandidate:
    """Unified candidate payload exchanged between any SynthesisEngine and KernelEvaluator."""

    candidate_id: str
    parent_ids: list[str] # single parent (ForgeKernel/RL) or multiple (crossover)
    vtm_ir: VTMKernel # structured topology + tactical levers
    backend: BackendName = "triton"
    code_override: str | None = None # optional direct Triton/TileLang escape hatch
    rationale: str = "" # curriculum row, mutation operator, or policy metadata
    variant: str = "fa2_bf16" # kernel structure variant (sdpa / fa1 / fa2_bf16 / fa3_tma / fa3_ws / fa4_tilelang)


@dataclass
class EvalResult:
    """Standardized compilation, correctness, timing & telemetry feedback."""

    candidate_id: str
    compiled: bool
    correct: bool
    device_time_us: float | None = None
    speedup_vs_ref: float | None = None # same-job paired ratio g(K,R) = t_R / t_K
    max_rel_error: float | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    error_log: str = ""
    # Robust confirmation stats (noise.py); None when not measured.
    m_c: float | None = None
    sigma_c: float | None = None
    lcb_speedup: float | None = None


class KernelEvaluator(Protocol):
    """Shared compilation, paired timing, correctness judge, and hardware profiler."""

    def evaluate(
        self,
        candidates: Sequence[KernelCandidate], # batched for evolutionary/RL rollouts
        target: TargetSpec,
        sealed_ref_id: str | None = None,
    ) -> list[EvalResult]: ...

    def profile_reference(self, sealed_ref_id: str, target: TargetSpec) -> dict[str, Any]: ...


class KnowledgeStore(Protocol):
    """Shared persistent memory & reference promotion registry across campaigns."""

    def query_recipes(self, target: TargetSpec, symptoms: list[str]) -> list[dict[str, Any]]: ...

    def record_certified(self, recipe: dict[str, Any], result: EvalResult) -> None: ...

    def promote_reference(
        self,
        candidate: KernelCandidate,
        result: EvalResult,
        expected_reference_id: str | None = None,
    ) -> str: ...


class SynthesisEngine(Protocol):
    """Common pluggable strategy interface for ForgeKernel, Evolutionary, RL & /goal agents."""

    def run_campaign(
        self,
        initial_kernel: VTMKernel,
        target: TargetSpec,
        evaluator: KernelEvaluator,
        kb: KnowledgeStore,
        sealed_ref_id: str | None = None,
        budget: dict[str, int] | None = None,
    ) -> KernelCandidate: ...


__all__ = [
    "BackendName",
    "STRATEGIES",
    "KernelCandidate",
    "EvalResult",
    "KernelEvaluator",
    "KnowledgeStore",
    "SynthesisEngine",
]
