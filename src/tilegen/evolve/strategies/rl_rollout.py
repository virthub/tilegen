# src/tilegen/evolve/strategies/rl_rollout.py
#
# RL rollout strategy (batched reward environment)

from __future__ import annotations

import random
from tilegen.evolve.interface import (
    EvalResult,
    KernelCandidate,
    KernelEvaluator,
    KnowledgeStore,
    SynthesisEngine,
)
from tilegen.evolve.mutation import mutate_kernel
from tilegen.ir.ops import VTMKernel
from tilegen.target.spec import TargetSpec


def reward(result: EvalResult) -> float:
    """Scalar reward: speedup when correct, heavily penalized otherwise."""
    if result.compiled and result.correct and result.speedup_vs_ref is not None:
        return result.speedup_vs_ref
    return -1.0


class RLRolloutEngine(SynthesisEngine):
    def __init__(self, seed: int = 0, epsilon: float = 0.1):
        self._seed = seed
        self.epsilon = epsilon

    def run_campaign(
        self,
        initial_kernel: VTMKernel,
        target: TargetSpec,
        evaluator: KernelEvaluator,
        kb: KnowledgeStore,
        sealed_ref_id: str | None = None,
        budget: dict[str, int] | None = None,
    ) -> KernelCandidate:
        budget = budget or {"rollouts": 3, "batch": 8}
        rollouts = budget.get("rollouts", 3)
        batch = budget.get("batch", 8)
        rng = random.Random(self._seed)

        incumbent = initial_kernel
        best = KernelCandidate("seed", [], initial_kernel, rationale="seed")
        best_reward = float("-inf")

        for rollout in range(rollouts):
            candidates = [
                KernelCandidate(
                    candidate_id=f"rl{rollout}_{i}",
                    parent_ids=[best.candidate_id],
                    vtm_ir=mutate_kernel(incumbent, rng),
                    rationale="rl rollout",
                )
                for i in range(batch)
            ]
            results = evaluator.evaluate(candidates, target, sealed_ref_id)
            scored = sorted(
                zip(candidates, results),
                key=lambda cr: reward(cr[1]),
                reverse=True,
            )
            # ε-greedy: exploit the best with prob 1-ε, explore a random rollout
            # with prob ε (escaping local optima that pure argmax would settle on).
            idx = rng.randrange(len(scored)) if rng.random() < self.epsilon else 0
            top_candidate, top_result = scored[idx]
            top_reward = reward(top_result)
            incumbent = top_candidate.vtm_ir
            if top_reward > best_reward:
                best_reward = top_reward
                best = top_candidate

        kb.promote_reference(best, EvalResult(best.candidate_id, True, True, speedup_vs_ref=max(best_reward, 0.0)))
        return best


__all__ = ["RLRolloutEngine", "reward"]
