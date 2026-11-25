# src/tilegen/evolve/strategies/goal_loop.py
#
# Single-episode goal-loop strategy (Claude /goal / STARK style)

from __future__ import annotations

import random
from tilegen.evolve.interface import (
    KernelCandidate,
    KernelEvaluator,
    KnowledgeStore,
    SynthesisEngine,
)
from tilegen.evolve.mutation import mutate_kernel
from tilegen.ir.ops import VTMKernel
from tilegen.target.spec import TargetSpec


class GoalLoopEngine(SynthesisEngine):
    def __init__(self, seed: int = 0):
        self._seed = seed

    def run_campaign(
        self,
        initial_kernel: VTMKernel,
        target: TargetSpec,
        evaluator: KernelEvaluator,
        kb: KnowledgeStore,
        sealed_ref_id: str | None = None,
        budget: dict[str, int] | None = None,
    ) -> KernelCandidate:
        budget = budget or {"iterations": 10}
        iterations = budget.get("iterations", 10)
        rng = random.Random(self._seed)

        best_kernel = initial_kernel
        best_time = float("inf")

        for it in range(iterations):
            cand = KernelCandidate(
                candidate_id=f"goal_{it}",
                parent_ids=[],
                vtm_ir=mutate_kernel(best_kernel, rng),
                rationale="single-episode patch",
            )
            results = evaluator.evaluate([cand], target, sealed_ref_id)
            r = results[0]
            if r.compiled and r.correct and r.device_time_us is not None and r.device_time_us < best_time:
                best_kernel = cand.vtm_ir
                best_time = r.device_time_us

        best_candidate = KernelCandidate(
            candidate_id="goal_best", parent_ids=[], vtm_ir=best_kernel,
            rationale="goal loop best",
        )
        from tilegen.evolve.interface import EvalResult

        kb.promote_reference(
            best_candidate, EvalResult("goal_best", True, True, device_time_us=best_time)
        )
        return best_candidate


__all__ = ["GoalLoopEngine"]
