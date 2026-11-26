# src/tilegen/evolve/strategies/evolutionary.py
#
# Evolutionary search strategy (population + crossover + mutation)

from __future__ import annotations

import random
from dataclasses import replace
from tilegen.evolve.interface import (
    KernelCandidate,
    KernelEvaluator,
    KnowledgeStore,
    SynthesisEngine,
)
from tilegen.evolve.mutation import crossover_knobs, mutate_kernel, mutate_variant
from tilegen.ir.ops import VTMKernel
from tilegen.target.spec import TargetSpec


class EvolutionaryEngine(SynthesisEngine):
    def __init__(self, seed: int = 0, population_size: int = 8):
        self._seed = seed
        self.population_size = population_size

    def run_campaign(
        self,
        initial_kernel: VTMKernel,
        target: TargetSpec,
        evaluator: KernelEvaluator,
        kb: KnowledgeStore,
        sealed_ref_id: str | None = None,
        budget: dict[str, int] | None = None,
    ) -> KernelCandidate:
        budget = budget or {"generations": 4}
        from tilegen.evolve.noise import campaign_noise, gate_value

        noise_floor, _ref_stats = campaign_noise(evaluator, initial_kernel)
        generations = budget.get("generations", 4)
        rng = random.Random(self._seed)

# Seed population with the initial kernel plus independent mutations of
        # *the seed* (not a chain drifting away from it).
        population: list[KernelCandidate] = [
            KernelCandidate("pop0", [], initial_kernel, rationale="seed")
        ]
        while len(population) < self.population_size:
            population.append(
                KernelCandidate(
                    candidate_id=f"pop{len(population)}",
                    parent_ids=["pop0"],
                    vtm_ir=mutate_kernel(initial_kernel, rng),
                    rationale="seed mutation",
                )
            )

        best_candidate = population[0]
        best_speedup = 0.0

        for gen in range(generations):
            parents = rng.sample(population, min(len(population), self.population_size))
            offspring: list[KernelCandidate] = []
            for i in range(0, len(parents) - 1, 2):
                crossed = crossover_knobs(
                    parents[i].vtm_ir.topology.knobs, parents[i + 1].vtm_ir.topology.knobs, rng
                )
                child_vtm = replace(
                    parents[i].vtm_ir,
                    topology=replace(parents[i].vtm_ir.topology, knobs=crossed),
                )
                child_vtm = mutate_kernel(child_vtm, rng)
                variant = parents[i].variant
                if rng.random() < 0.15:
                    variant = mutate_variant(variant, rng)
                offspring.append(
                    KernelCandidate(
                        candidate_id=f"gen{gen}_o{i // 2}",
                        parent_ids=[parents[i].candidate_id, parents[i + 1].candidate_id],
                        vtm_ir=child_vtm,
                        rationale="crossover+mutation",
                        variant=variant,
                    )
                )

            results = evaluator.evaluate(offspring, target, sealed_ref_id)
            ranked = sorted(
                (r for r in results if r.compiled and r.correct and r.speedup_vs_ref is not None),
                key=lambda r: r.speedup_vs_ref or 0.0,
                reverse=True,
            )
            for r in ranked[: self.population_size]:
                idx = next(i for i, c in enumerate(offspring) if c.candidate_id == r.candidate_id)
                population.append(offspring[idx])
            population = population[-self.population_size :]

            if ranked:
                top = ranked[0]
                idx = next(i for i, c in enumerate(offspring) if c.candidate_id == top.candidate_id)
                if (top.speedup_vs_ref or 0.0) > best_speedup:
                    best_speedup = top.speedup_vs_ref or 0.0
                    best_candidate = offspring[idx]

        kb.promote_reference(best_candidate, _placeholder(best_candidate, best_speedup))
        return best_candidate


def _placeholder(candidate: KernelCandidate, speedup: float):
    from tilegen.evolve.interface import EvalResult

    return EvalResult(
        candidate_id=candidate.candidate_id, compiled=True, correct=True,
        speedup_vs_ref=speedup,
    )


__all__ = ["EvolutionaryEngine"]
