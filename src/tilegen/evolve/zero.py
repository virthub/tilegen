# src/tilegen/evolve/zero.py
#
# The Zero evolver: the latest evolver produced by the Meta-Evolver
# (meta_evolver.py). This file is an *artifact* — the Meta-Evolver mutates
# the POLICY block below (AST-level), validates the result with static and
# smoke gates, archives accepted versions under the knowledge-store meta
# directory, and atomically promotes the best one to this path. ZeroEngine is
# therefore a `SynthesisEngine` whose search policy is itself evolved, one
# level up from the kernels it evolves. Select it with
# ``./run.sh --evolve --strategy zero``.

from __future__ import annotations

import random
from typing import Optional

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

POLICY = {
    "plateau_window": 2,
    "variant_mutation_prob": 0.15,
    "warm_start": True,
    "bias_strength": 1.0,
    "default_generations": 5,
    "default_candidates_per_generation": 8,
}


class ZeroEngine(SynthesisEngine):
    """Reference-guided kernel evolver whose policy is evolved by the Meta-Evolver.

    Behavior mirrors the built-in ForgeKernel strategy, but every policy knob
    lives in the module-level ``POLICY`` block so the Meta-Evolver can mutate
    this file without touching the framework contracts.
    """

    def __init__(self, seed: int = 0, policy: Optional[dict] = None):
        self._seed = seed
        self.policy = dict(POLICY if policy is None else policy)

    def run_campaign(
        self,
        initial_kernel: VTMKernel,
        target: TargetSpec,
        evaluator: KernelEvaluator,
        kb: KnowledgeStore,
        sealed_ref_id: Optional[str] = None,
        budget: Optional[dict] = None,
    ) -> KernelCandidate:
        budget = budget or {}
        generations = budget.get("generations", self.policy["default_generations"])
        n_per_gen = budget.get(
            "candidates_per_generation",
            self.policy["default_candidates_per_generation"],
        )
        from tilegen.evolve.curriculum import Curriculum
        from tilegen.evolve.triggers import TriggerDetector

        curriculum = Curriculum.load(budget)
        thresholds = dict(curriculum.thresholds)
        thresholds["N_stall"] = int(
            budget.get("plateau_window", self.policy["plateau_window"]))
        detector = TriggerDetector(thresholds)
        from tilegen.evolve.attempt_history import AttemptHistory
        from tilegen.evolve.metrics import CampaignMetrics
        from tilegen.evolve.triggers import RevisionGuard

        history = AttemptHistory()
        guard = RevisionGuard()
        metrics = CampaignMetrics()
        budget_consumed = float(budget.get("budget_consumed_fraction", 0.0))
        warm_start = budget.get("warm_start", self.policy["warm_start"])
        variant_prob = budget.get(
            "variant_mutation_prob", self.policy["variant_mutation_prob"]
        )
        bias_strength = budget.get("bias_strength", self.policy["bias_strength"])

        rng = random.Random(self._seed)
        ref = evaluator.profile_reference(sealed_ref_id or "generation-0", target)
        best_time = float(ref.get("device_time_us", float("inf")))

        best_kernel = initial_kernel
        best_result = EvalResult(
            candidate_id="generation-0",
            compiled=True,
            correct=True,
            device_time_us=best_time,
            speedup_vs_ref=1.0,
        )
        best_candidate = KernelCandidate(
            candidate_id="generation-0",
            parent_ids=[],
            vtm_ir=initial_kernel,
            rationale="sealed reference",
        )

        if warm_start:
            for recipe in kb.query_recipes(target, []):
                config = recipe.get("config")
                if isinstance(config, dict) and "tile_m" in config:
                    from dataclasses import replace as _replace

                    from tilegen.ir.topology import TacticalKnobs

                    try:
                        valid = {
                            k: v
                            for k, v in config.items()
                            if k in TacticalKnobs.__dataclass_fields__
                        }
                        warm = TacticalKnobs(**valid)
                        best_kernel = _replace(
                            best_kernel,
                            topology=_replace(best_kernel.topology, knobs=warm),
                        )
                        best_candidate = KernelCandidate(
                            candidate_id="generation-0-warm",
                            parent_ids=[],
                            vtm_ir=best_kernel,
                            rationale="KB warm-start",
                        )
                    except (TypeError, ValueError):
                        pass
                    break

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

        for _gen in range(generations):
            bias = diagnose_fields(best_result.telemetry or {})
            candidates = []
            for i in range(n_per_gen):
                use_bias = bias and rng.random() < bias_strength
                vtm = mutate_kernel(
                    best_kernel, rng, bias=bias if use_bias else None,
                    exclude=history.failed_fields())
                variant = best_candidate.variant
                if rng.random() < variant_prob:
                    variant = mutate_variant(variant, rng)
                candidates.append(
                    KernelCandidate(
                        candidate_id=f"gen{_gen}_c{i}",
                        parent_ids=[best_candidate.candidate_id],
                        vtm_ir=vtm,
                        rationale=f"profiler-guided mutation (bias={bias if use_bias else None})",
                        variant=variant,
                    )
                )

            results = evaluator.evaluate(candidates, target, sealed_ref_id)
            metrics.submissions += len(candidates)
            gen_best = min(
                (
                    r
                    for r in results
                    if r.compiled and r.correct and r.device_time_us is not None
                ),
                key=lambda r: r.device_time_us,
                default=None,
            )
            idx = next((i for i, r in enumerate(results)
                        if r.candidate_id == gen_best.candidate_id), None) \
                if gen_best is not None else None
            accepted = False
            if gen_best is not None and idx is not None \
                    and gen_best.device_time_us < best_time:
# Unified bootstrap gate + small-sample LCB confirmation,
                # mirroring ForgeKernel.
                speedup = gen_best.speedup_vs_ref or 0.0
                gate = gate_value(
                    noise_floor,
                    curriculum.active_bootstrap_multiplier if _gen == 0 else 1.0)
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


class ZeroEvolveEngine(SynthesisEngine):
    """The ``zero`` strategy entry point.

    Runs the Meta-Evolver to evolve this file's POLICY until ZeroEngine
    outperforms the project's built-in evolvers (forgekernel, evolutionary,
    world_model, goal_loop, rl_rollout, schedule_codesign) on the given task,
    or until the meta budget / evidence-gap stop.  On victory the returned
    candidate is the winning variant's best kernel; otherwise the best kernel
    any engine produced (champion baseline).
    """

    def __init__(self, seed: int = 0):
        self._seed = seed

    def run_campaign(
        self,
        initial_kernel: VTMKernel,
        target: TargetSpec,
        evaluator: KernelEvaluator,
        kb: KnowledgeStore,
        sealed_ref_id: Optional[str] = None,
        budget: Optional[dict] = None,
    ) -> KernelCandidate:
        from tilegen.evolve.meta_evolver import MetaEvolverEngine

        budget = dict(budget or {})
        budget.setdefault("beat_builtin_engines", True)
        return MetaEvolverEngine(seed=self._seed).run_campaign(
            initial_kernel, target, evaluator, kb, sealed_ref_id, budget
        )


__all__ = ["ZeroEngine", "ZeroEvolveEngine", "POLICY"]
