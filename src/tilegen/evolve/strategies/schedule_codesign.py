# src/tilegen/evolve/strategies/schedule_codesign.py
#
# Schedule co-design strategy (CAKE)

from __future__ import annotations

import random
from tilegen.evolve.interface import (
    EvalResult,
    KernelCandidate,
    KernelEvaluator,
    KnowledgeStore,
    SynthesisEngine,
)
from tilegen.evolve.mutation import kernel_state, mutate_kernel
from tilegen.ir.ops import VTMKernel
from tilegen.target.spec import TargetSpec


class ScheduleCodesignEngine(SynthesisEngine):
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
        budget = budget or {"generations": 4, "candidates_per_generation": 8}
        generations = budget.get("generations", 4)
        n_per_gen = budget.get("candidates_per_generation", 8)
        rng = random.Random(self._seed)

        best_kernel = initial_kernel
        best = KernelCandidate("seed", [], initial_kernel, rationale="seed")
        best_time = float("inf")

        for gen in range(generations):
            candidates = [
                KernelCandidate(
                    candidate_id=f"cake{gen}_{i}",
                    parent_ids=[best.candidate_id],
                    vtm_ir=mutate_kernel(best_kernel, rng),
                    rationale="typed schedule transform",
                )
                for i in range(n_per_gen)
            ]
            results = evaluator.evaluate(candidates, target, sealed_ref_id)
            for c, r in zip(candidates, results):
                if not (r.compiled and r.correct and r.device_time_us is not None):
                    continue
                if r.device_time_us < best_time:
# Persist the verified rewrite rule (before -> after) plus the
                    # winning knobs config so other strategies can warm-start.
                    kb.record_certified(
                        {
                            "target": target.name,
                            "config": c.vtm_ir.topology.knobs.to_dict(),
                            "rewrite": {
                                "before": kernel_state(best_kernel),
                                "after": kernel_state(c.vtm_ir),
                            },
                        },
                        r,
                    )
                    best_time = r.device_time_us
                    best_kernel = c.vtm_ir
                    best = c

        return best


__all__ = ["ScheduleCodesignEngine"]
