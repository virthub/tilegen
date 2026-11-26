# src/tilegen/evolve/strategies/world_model.py
#
# World-model tree search strategy (K-Search)

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


class NearestNeighborValueModel:
    """Distance-weighted k-NN value function over kernel states.

    ``update(state, value)`` records an observation (keeping the best value for
    each exact state); ``estimate(state)`` returns a weighted average of the
    ``k`` closest observed states, so nearby unseen states inherit value.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self._obs: list[tuple[tuple, float]] = []

    def update(self, state: tuple, value: float) -> None:
        for i, (s, v) in enumerate(self._obs):
            if s == state:
                if value > v:
                    self._obs[i] = (s, value)
                return
        self._obs.append((state, value))

    def estimate(self, state: tuple) -> float:
        if not self._obs:
            return 0.0
        # An exact observed state returns its recorded value.
        for s, v in self._obs:
            if s == state:
                return v
        scored = sorted(((self._distance(state, s), v) for s, v in self._obs), key=lambda x: x[0])
        top = scored[: self.k]
        weights = [1.0 / (1.0 + d) for d, _ in top]
        total = sum(weights) or 1.0
        return sum(w * v for w, (_, v) in zip(weights, top)) / total

    @staticmethod
    def _distance(a: tuple, b: tuple) -> float:
        d = 0.0
        for x, y in zip(a, b):
            if isinstance(x, str) or isinstance(y, str):
                d += 0.0 if x == y else 1.0
            else:
                d += abs(x - y) / 100.0
        return d

    def __len__(self) -> int:
        return len(self._obs)


class WorldModelEngine(SynthesisEngine):
    """K-Search: a generalizing value model over kernel states guides expansion."""

    def __init__(self, seed: int = 0, k: int = 3):
        self._seed = seed
        self.value_model = NearestNeighborValueModel(k=k)

    def run_campaign(
        self,
        initial_kernel: VTMKernel,
        target: TargetSpec,
        evaluator: KernelEvaluator,
        kb: KnowledgeStore,
        sealed_ref_id: str | None = None,
        budget: dict[str, int] | None = None,
    ) -> KernelCandidate:
        budget = budget or {"generations": 4, "beam_width": 4}
        generations = budget.get("generations", 4)
        beam_width = budget.get("beam_width", 4)
        rng = random.Random(self._seed)

        beam: list[VTMKernel] = [initial_kernel]
        best = KernelCandidate("seed", [], initial_kernel, rationale="seed")
        best_speedup = 0.0

        for gen in range(generations):
# Expand the current beam; prioritize states with the highest
            # *estimated* value (which generalizes to unseen near-neighbors).
            ordered = sorted(
                beam, key=lambda k: self.value_model.estimate(kernel_state(k)), reverse=True
            )
            candidates: list[KernelCandidate] = []
            for i in range(beam_width):
                parent = ordered[i % len(ordered)]
                candidates.append(
                    KernelCandidate(
                        candidate_id=f"wm{gen}_{i}",
                        parent_ids=[best.candidate_id],
                        vtm_ir=mutate_kernel(parent, rng),
                        rationale="world-model expansion",
                    )
                )

            results = evaluator.evaluate(candidates, target, sealed_ref_id)
            scored: list[tuple[float, KernelCandidate]] = []
            for c, r in zip(candidates, results):
                if r.compiled and r.correct and r.speedup_vs_ref is not None:
                    self.value_model.update(kernel_state(c.vtm_ir), r.speedup_vs_ref)
                    scored.append((r.speedup_vs_ref, c))

            if not scored:
                continue
            scored.sort(key=lambda x: -x[0])
            beam = [c.vtm_ir for _, c in scored[:beam_width]]
            if scored[0][0] > best_speedup:
                best_speedup = scored[0][0]
                best = scored[0][1]

        kb.promote_reference(
            best, EvalResult(best.candidate_id, True, True, speedup_vs_ref=best_speedup)
        )
        return best


__all__ = ["WorldModelEngine", "NearestNeighborValueModel"]
