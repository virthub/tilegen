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
    """Reference-guided self-evolution with plateau-triggered structural reopening."""

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
        from tilegen.evolve.curriculum import Curriculum
        from tilegen.evolve.triggers import TriggerDetector

        curriculum = Curriculum.load(budget)
        thresholds = dict(curriculum.thresholds)
        if budget.get("plateau_window") is not None:
            thresholds["N_stall"] = int(budget["plateau_window"])
        detector = TriggerDetector(thresholds)
        from tilegen.evolve.attempt_history import AttemptHistory
        from tilegen.evolve.metrics import CampaignMetrics
        from tilegen.evolve.triggers import RevisionGuard

        history = AttemptHistory()
        guard = RevisionGuard()
        metrics = CampaignMetrics()
        budget_consumed = float(budget.get("budget_consumed_fraction", 0.0))

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
                break # only the most recent recipe

        # Noise floor + reference robust stats (measurement layer).
        from dataclasses import replace as _replace

        from tilegen.evolve.evaluator import knobs_to_params
        from tilegen.evolve.noise import (
            campaign_noise,
            gate_value,
            lc_bound,
            robust_stats,
        )

        noise_floor, ref_stats = campaign_noise(evaluator, initial_kernel)
        sampler = getattr(evaluator, "sample_times", None)

        for gen in range(generations):
            # Profiler-guided bias: incumbent telemetry -> preferred knob fields.
            bias = diagnose_fields(best_result.telemetry or {})

            candidates = []
            for i in range(n_per_gen):
                vtm = mutate_kernel(
                    best_kernel, rng, bias=bias,
                    exclude=history.failed_fields())
                variant = best_candidate.variant
                if rng.random() < self.variant_mutation_prob:
                    variant = mutate_variant(variant, rng) # FA2 <-> FA4 discovery
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
            metrics.submissions += len(candidates)
            gen_best = min(
                (r for r in results if r.compiled and r.correct and r.device_time_us is not None),
                key=lambda r: r.device_time_us,
                default=None,
            )
            idx = next((i for i, r in enumerate(results)
                        if r.candidate_id == gen_best.candidate_id), None) \
                if gen_best is not None else None
            accepted = False
            if gen_best is not None and idx is not None \
                    and gen_best.device_time_us < best_time:
# Unified bootstrap gate: speedup >= max(1 + floor, bootstrap
# multiplier); 1.05 for the first generation, 1.0 afterwards.
# Accepted candidates get a small-sample LCB confirmation
                # when the evaluator can sample.
                speedup = gen_best.speedup_vs_ref or 0.0
                gate = gate_value(
                    noise_floor,
                    curriculum.active_bootstrap_multiplier if gen == 0 else 1.0)
                accepted = speedup >= gate
                if accepted and sampler is not None and ref_stats is not None:
                    c_params = knobs_to_params(
                        candidates[idx].vtm_ir.topology.knobs)
                    c_runs = sampler(c_params, candidates[idx].variant, iters=5)
                    m_c, _, s_c = robust_stats(c_runs)
                    lcb = lc_bound(m_c, s_c, ref_stats[0], ref_stats[2],
                                   k_c=1.0, k_r=2.0)
                    accepted = lcb is not None and lcb >= 1.0 + noise_floor
                    if accepted:
                        gen_best = _replace(
                            gen_best, m_c=m_c, sigma_c=s_c, lcb_speedup=lcb)
            if accepted:
                best_kernel = candidates[idx].vtm_ir
                best_candidate = candidates[idx]
                best_result = gen_best
                best_time = gen_best.device_time_us
                detector.update(improved=True)
                guard.on_promotion()
                metrics.effective_promotions += 1
            else:
                gate_failed = (
                    gen_best is not None and gen_best.compiled
                    and gen_best.correct and idx is not None
                    and gen_best.device_time_us < best_time
                    and speedup < gate
                )
                fired = detector.update(
                    improved=False, gate_failed=gate_failed,
                    m0_localized=True,
                )
                metrics.rollbacks += 1
                if gate_failed:
                    metrics.noise_rejections += 1
                if idx is not None:
                    cand_knobs = candidates[idx].vtm_ir.topology.knobs.to_dict()
                    inc_knobs = best_kernel.topology.knobs.to_dict()
                    changed = [k for k in cand_knobs
                               if cand_knobs[k] != inc_knobs.get(k)]
                    history.record(
                        changed, "below_noise" if gate_failed else "stall")
                if "stall_without_new_best" in fired or "gate_impasse" in fired:
# Trigger-anchored structural revision: reopen the structural topology
# on the incumbent only; the best *proven* candidate is
                    # unchanged. Bounded by the budget-modulated K_max cap.
                    if guard.allow(budget_consumed):
                        from tilegen.evolve.pricing import (
                            apply_priced_revision, price_revision)

                        curriculum.m0_epoch += 1
                        guard.record_revision()
                        history.advance_epoch()
                        metrics.m0_revisions += 1
                        records = price_revision(
                            best_result.telemetry or {}, noise_floor=noise_floor)
                        best_kernel = apply_priced_revision(
                            best_kernel, records, rng)
                    else:
                        metrics.m0_revisions_blocked += 1

        if best_candidate.candidate_id != "generation-0":
            kb.promote_reference(best_candidate, best_result,
                                  expected_reference_id=sealed_ref_id)
        from pathlib import Path

        metrics.record(Path(kb.root) if hasattr(kb, "root") else Path("conf/kb"))
        return best_candidate


__all__ = ["ForgeKernelEngine"]
