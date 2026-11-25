# src/tilegen/evolve/strategies/forgekernel.py
#
# ForgeKernel-style reference-guided self-evolution strategy (default)

from __future__ import annotations

import random
from tilegen.evolve.interface import (
    EvalResult,
    KernelCandidate,
    KernelEvaluator,
    KnowledgeStore,
    SynthesisEngine,
)
from tilegen.evolve.mutation import mutate_kernel, mutate_variant, reopen_m0
from tilegen.evolve.profiler import diagnose_fields
from tilegen.ir.ops import VTMKernel
from tilegen.target.spec import TargetSpec


class ForgeKernelEngine(SynthesisEngine):
    """Reference-guided self-evolution with plateau-triggered M0 reopening."""

    def __init__(self, seed: int = 0, variant_mutation_prob: float = 0.15):
        self._seed = seed
        self.variant_mutation_prob = variant_mutation_prob

    def run_campaign(
        self,
        initial_kernel: VTMKernel,
        target: TargetSpec,
        evaluator: KernelEvaluator,
        kb: KnowledgeStore,
        sealed_ref_id: str | None = None,
        budget: dict[str, int] | None = None,
    ) -> KernelCandidate:
        budget = budget or {"generations": 5, "candidates_per_generation": 8}
        generations = budget.get("generations", 5)
        n_per_gen = budget.get("candidates_per_generation", 8)
        plateau_window = budget.get("plateau_window", 2)

        rng = random.Random(self._seed)
        ref = evaluator.profile_reference(sealed_ref_id or "generation-0", target)
        best_time = float(ref.get("device_time_us", float("inf")))

        best_kernel = initial_kernel
        best_result = EvalResult(
            candidate_id="generation-0", compiled=True, correct=True,
            device_time_us=best_time, speedup_vs_ref=1.0,
        )
        best_candidate = KernelCandidate(
            candidate_id="generation-0", parent_ids=[], vtm_ir=initial_kernel,
            rationale="sealed reference",
        )

        # Warm-start: apply the most recent certified recipe for this target
        # (cross-run knowledge transfer through the KnowledgeStore).
        for recipe in kb.query_recipes(target, []):
            config = recipe.get("config")
            if isinstance(config, dict) and "tile_m" in config:
                from dataclasses import replace as _replace

                from tilegen.ir.topology import TacticalKnobs

                try:
                    valid = {k: v for k, v in config.items()
                             if k in TacticalKnobs.__dataclass_fields__}
                    warm = TacticalKnobs(**valid)
                    best_kernel = _replace(
                        best_kernel,
                        topology=_replace(best_kernel.topology, knobs=warm),
                    )
                    best_candidate = KernelCandidate(
                        candidate_id="generation-0-warm", parent_ids=[],
                        vtm_ir=best_kernel, rationale="KB warm-start",
                    )
                except (TypeError, ValueError):
                    pass
                break  # only the most recent recipe

        plateau = 0
        for gen in range(generations):
            # Profiler-guided bias: incumbent telemetry -> preferred knob fields.
            bias = diagnose_fields(best_result.telemetry or {})

            candidates = []
            for i in range(n_per_gen):
                vtm = mutate_kernel(best_kernel, rng, bias=bias)
                variant = best_candidate.variant
                if rng.random() < self.variant_mutation_prob:
                    variant = mutate_variant(variant, rng)   # FA2 <-> FA4 discovery
                candidates.append(
                    KernelCandidate(
                        candidate_id=f"gen{gen}_c{i}",
                        parent_ids=[best_candidate.candidate_id],
                        vtm_ir=vtm,
                        rationale=f"profiler-guided mutation (bias={bias})",
                        variant=variant,
                    )
                )

            results = evaluator.evaluate(candidates, target, sealed_ref_id)
            gen_best = min(
                (r for r in results if r.compiled and r.correct and r.device_time_us is not None),
                key=lambda r: r.device_time_us,
                default=None,
            )
            if gen_best is not None and gen_best.device_time_us < best_time:
                idx = next(i for i, r in enumerate(results) if r.candidate_id == gen_best.candidate_id)
                best_kernel = candidates[idx].vtm_ir
                best_candidate = candidates[idx]
                best_result = gen_best
                best_time = gen_best.device_time_us
                plateau = 0
            else:
                plateau += 1
                if plateau >= plateau_window:
                    # Stall-anchored plateau: reopen the M0 topology on the
                    # incumbent only; the best *proven* candidate is unchanged.
                    best_kernel = reopen_m0(best_kernel, rng)
                    plateau = 0

        if best_candidate.candidate_id != "generation-0":
            kb.promote_reference(best_candidate, best_result)
        return best_candidate


__all__ = ["ForgeKernelEngine"]
