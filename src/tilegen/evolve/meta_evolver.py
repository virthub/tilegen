# src/tilegen/evolve/meta_evolver.py
#
# The Meta-Evolver: the evolver of the evolver. It mutates the POLICY block
# of `zero.py` (the Zero evolver), validates each variant with static and
# smoke gates, scores it by running ZeroEngine campaigns against sealed
# reference tasks, archives accepted versions under the knowledge store's
# meta directory, and atomically promotes the best one back to zero.py.
# It never touches kernels and never mutates itself — only zero.py's search
# policy is on the mutation surface. Select it with
# ``./run.sh --evolve --strategy meta_evolver``.

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from tilegen.evolve.interface import (
    EvalResult,
    KernelCandidate,
    KernelEvaluator,
    KnowledgeStore,
    SynthesisEngine,
)
from tilegen.evolve.kb_manager import FileKnowledgeStore
from tilegen.ir.ops import VTMKernel
from tilegen.target.spec import TargetSpec

ZERO_PATH = Path(__file__).with_name("zero.py")

_ALLOWED_TOP_MODULES = {
    "__future__", "dataclasses", "json", "math", "pathlib", "random",
    "tilegen", "time", "typing",
}
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}
_DYNAMIC_ATTR_CALLS = {"getattr", "setattr"}

_POLICY_KINDS = {
    "plateau_window": "int",
    "variant_mutation_prob": "float",
    "warm_start": "bool",
    "bias_strength": "float",
    "default_generations": "int",
    "default_candidates_per_generation": "int",
}
_MUTATION_RATE = 0.5
_COMPLEXITY_WEIGHT = 0.01 # speedup penalty per 100 added characters

# Symptom -> policy-dim direction map: aggregated inner-campaign telemetry
# guides which POLICY dims the next generation mutates.
_POLICY_BIAS: dict[str, dict[str, int]] = {
    "tensorcore_idle": {"default_candidates_per_generation": +1,
                        "bias_strength": +1},
    "mbarrier_wait_stall": {"variant_mutation_prob": -1},
    "register_spill": {"default_candidates_per_generation": +1},
    "sfu_stall": {"variant_mutation_prob": -1},
    "cta_boundary_stall": {"plateau_window": -1},
}


def policy_bias(telemetry: dict[str, float]) -> dict[str, int]:
    """Map aggregated symptom telemetry to (policy dim -> direction)."""
    out: dict[str, int] = {}
    for symptom, dims in _POLICY_BIAS.items():
        if float(telemetry.get(symptom, 0.0)) > 0.0:
            for dim, direction in dims.items():
                out.setdefault(dim, direction)
    return out


def static_gate(source: str) -> list[str]:
    """Return a list of violations; empty means the source passes."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax: {exc}"]
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_TOP_MODULES:
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in _ALLOWED_TOP_MODULES:
                violations.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                violations.append(f"call {node.func.id}")
            elif node.func.id in _DYNAMIC_ATTR_CALLS and node.args:
                second = node.args[1] if len(node.args) > 1 else None
                if not (isinstance(second, ast.Constant)
                        and isinstance(second.value, str)):
                    violations.append(f"dynamic {node.func.id}")
    return violations


def smoke_gate(source: str, timeout: int = 60) -> Optional[str]:
    """Import the source in a subprocess and instantiate ZeroEngine.

    Returns None on success, or an error description.
    """
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "zero_smoke.py"
        path.write_text(source, encoding="utf-8")
        script = (
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('zero_smoke', {str(path)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "engine = mod.ZeroEngine(seed=0)\n"
            "assert hasattr(engine, 'run_campaign'), 'missing run_campaign'\n"
            "print('SMOKE_OK')\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "timeout"
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "nonzero exit")[-400:]
            return f"import failed: {detail}"
        if "SMOKE_OK" not in proc.stdout:
            return "smoke marker missing"
    return None


def _policy_block(tree: ast.Module) -> ast.Assign:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "POLICY":
                    if isinstance(node.value, ast.Dict):
                        return node
    raise ValueError("zero.py has no module-level POLICY dict")


def _render_policy(policy: dict) -> str:
    lines = ["POLICY = {"]
    for key in sorted(policy):
        lines.append(f"    {key!r}: {policy[key]!r},")
    lines.append("}")
    return "\n".join(lines)


def _policy_values(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    block = _policy_block(tree)
    policy: dict[str, Any] = {}
    for key, val in zip(block.value.keys, block.value.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str) \
                and isinstance(val, ast.Constant):
            policy[key.value] = val.value
    return policy


def _set_policy(source: str, policy: dict[str, Any]) -> str:
    """Rewrite the POLICY block values, preserving the header verbatim."""
    tree = ast.parse(source)
    block = _policy_block(tree)
    current = _policy_values(source)
    merged = {**current, **{k: v for k, v in policy.items() if k in current}}
    lines = source.splitlines()
    header = "\n".join(lines[: block.lineno - 1])
    footer = "\n".join(lines[block.end_lineno:])
    return header + "\n" + _render_policy(merged) + "\n" + footer


def mutate_source(
    source: str,
    rng: random.Random,
    bias: Optional[dict[str, int]] = None,
) -> tuple[str, list[str]]:
    """Mutate zero.py's POLICY literals; returns (new_source, diff_notes).

    ``bias`` maps policy dims to a preferred direction (+1 raise, -1 lower,
    0 random); only biased dims are mutated when provided.  The file header
    and all non-POLICY code are preserved verbatim.
    """
    policy = _policy_values(source)
    notes: list[str] = []
    if bias is not None:
        targets = [k for k in bias if k in _POLICY_KINDS]
        if not targets:
            return source, notes
    else:
        targets = list(_POLICY_KINDS)
    mutated = False
    while not mutated:
        for key in list(policy):
            if key not in targets or rng.random() > _MUTATION_RATE:
                continue
            kind = _POLICY_KINDS[key]
            old = policy[key]
            direction = (bias or {}).get(key, 0)
            if kind == "int":
                delta = direction if direction != 0 else rng.choice([-1, 1])
                policy[key] = max(1, int(old) + delta)
            elif kind == "float":
                if direction > 0:
                    factor = rng.choice([1.2, 1.5])
                elif direction < 0:
                    factor = rng.choice([0.5, 0.8])
                else:
                    factor = rng.choice([0.5, 1.5, 2.0])
                policy[key] = min(1.0, max(0.05, round(float(old) * factor, 3)))
            elif kind == "bool":
                policy[key] = not bool(old)
            notes.append(f"{key}: {old} -> {policy[key]}")
            mutated = True
    return _set_policy(source, policy), notes


def _load_source(source: str):
    """Load generated source in-process (gates must have passed)."""
    import importlib.util

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "zero_cand.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("zero_cand", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


BUILTIN_ENGINES = (
    "forgekernel",
    "evolutionary",
    "world_model",
    "goal_loop",
    "rl_rollout",
    "schedule_codesign",
)


def _task_key(kernel: VTMKernel, target: TargetSpec) -> str:
    from tilegen.runtime.hash import semantic_hash

    return semantic_hash({
        "kernel": kernel.name,
        "serialized": kernel.serialize(),
        "target": target.name,
        "sm": target.sm,
    })[:16]


def champion_baseline(
    task: tuple[VTMKernel, TargetSpec, KernelEvaluator],
    kb_root: Path,
    inner_budget: dict,
    seeds: list[int],
    recompute: bool = False,
) -> dict:
    """Best (speedup, candidate) across the built-in engines on the task.

    Cached per task under ``kb_root/meta/champion.json``.
    """
    from tilegen.evolve.strategies import ENGINES

    kernel, target, evaluator = task
    meta = _meta_dir(kb_root)
    cache_path = meta / "champion.json"
    if cache_path.exists() and not recompute:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("task_key") == _task_key(kernel, target):
                result = dict(cached.get("result") or {})
                cand_dict = result.pop("candidate", None)
                if cand_dict:
                    from tilegen.evolve.adapter import candidate_from_dict

                    result["candidate"] = candidate_from_dict(
                        cand_dict, verify=False)
                return result
        except (json.JSONDecodeError, OSError):
            pass

    best: dict[str, Any] = {"speedup": 0.0, "engine": None}
    for name in BUILTIN_ENGINES:
        engine_cls = ENGINES.get(name)
        if engine_cls is None:
            continue
        for seed in seeds:
            kb = FileKnowledgeStore(kb_root / f"champ_{name}_seed{seed}")
            try:
                if name in ("forgekernel", "rl_rollout", "schedule_codesign"):
                    engine = engine_cls(seed=seed)
                else:
                    engine = engine_cls()
                cand = engine.run_campaign(
                    kernel, target, evaluator, kb, None, dict(inner_budget)
                )
                res = evaluator.evaluate([cand], target, None)
                speedup = float(res[0].speedup_vs_ref or 1.0) if res else 1.0
            except Exception:
                speedup = 0.0
                cand = None
            if speedup > best["speedup"]:
                best = {"speedup": speedup, "engine": name, "seed": seed,
                        "candidate": cand}
    from tilegen.evolve.adapter import candidate_to_dict

    serializable = dict(best)
    if best.get("candidate") is not None:
        serializable["candidate"] = candidate_to_dict(best["candidate"])
    cache_path.write_text(
        json.dumps({"task_key": _task_key(kernel, target),
                    "result": serializable}, indent=2),
        encoding="utf-8",
    )
    return best


def _score_source(
    source: str,
    task: tuple[VTMKernel, TargetSpec, KernelEvaluator],
    kb_root: Path,
    seeds: list[int],
    inner_budget: dict,
) -> Optional[dict]:
    """Score a zero.py variant on one task; median speedup across seeds.

    Returns None if the variant fails any gate.
    """
    if static_gate(source) or smoke_gate(source):
        return None
    try:
        mod = _load_source(source)
        engine_cls = mod.ZeroEngine
    except Exception:
        return None

    kernel, target, evaluator = task
    seed_runs: list[tuple[float, Any]] = []
    telemetry_sum: dict[str, float] = {}
    for seed in seeds:
        kb = FileKnowledgeStore(kb_root / f"seed{seed}")
        collector = _CollectingEvaluator(evaluator)
        try:
            engine = engine_cls(seed=seed)
            cand = engine.run_campaign(
                kernel, target, collector, kb, None, dict(inner_budget)
            )
            res = evaluator.evaluate([cand], target, None)
            speedup = float(res[0].speedup_vs_ref or 1.0) if res else 1.0
            for t in collector.telemetry:
                for k, v in t.items():
                    telemetry_sum[k] = telemetry_sum.get(k, 0.0) + float(v)
        except Exception:
            speedup = 0.0
            cand = None
        seed_runs.append((speedup, cand))
    ordered = sorted(seed_runs, key=lambda rc: rc[0])
    median = ordered[len(ordered) // 2][0]
    return {
        "median_speedup": median,
        "seed_scores": [rc[0] for rc in ordered],
        "best_candidate": ordered[-1][1],
        "telemetry_agg": telemetry_sum,
        "source_len": len(source),
    }


_LLM_SYSTEM = (
    "You are tuning the search policy of a kernel evolver.  Reply with ONLY "
    "a single Python dict literal for the new POLICY values, using exactly "
    "these keys: plateau_window (int), variant_mutation_prob (float 0..1), "
    "warm_start (bool), bias_strength (float 0..1), default_generations "
    "(int), default_candidates_per_generation (int).  No prose."
)


def llm_propose_policy(
    incumbent_source: str,
    context: dict,
    client,
    timeout: Optional[float] = None,
) -> Optional[tuple[str, list[str]]]:
    """Ask the LLM for a new POLICY block; return (source, notes) or None.

    The reply is parsed strictly: only known POLICY keys survive, values are
    coerced to their declared kinds, and the header is preserved verbatim.
    """
    import ast as _ast

    current = _policy_values(incumbent_source)
    user = (
        "Current POLICY:\n" + json.dumps(current, indent=2)
        + "\n\nCampaign context:\n" + json.dumps(context, indent=2, default=str)
        + "\n\nPropose one improved POLICY dict."
    )
    try:
        reply = client.complete(user, timeout=timeout)
    except Exception:
        return None
    if not reply:
        return None
    # Find the first brace-delimited dict in the reply (code fences allowed).
    start = reply.find("{")
    end = reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        proposed = _ast.literal_eval(reply[start:end + 1])
    except (ValueError, SyntaxError):
        return None
    if not isinstance(proposed, dict):
        return None
    clean: dict[str, Any] = {}
    for key, value in proposed.items():
        if key not in _POLICY_KINDS:
            continue
        kind = _POLICY_KINDS[key]
        try:
            if kind == "int":
                clean[key] = max(1, int(value))
            elif kind == "float":
                clean[key] = min(1.0, max(0.05, round(float(value), 3)))
            elif kind == "bool":
                clean[key] = bool(value)
        except (TypeError, ValueError):
            continue
    if not clean:
        return None
    notes = [f"{k}: {current[k]} -> {clean[k]}" for k in sorted(clean)]
    return _set_policy(incumbent_source, clean), notes


def _llm_client(budget: dict):
    """Resolve the LLM client: injected for tests, else lazily constructed."""
    injected = budget.get("llm_client")
    if injected is not None:
        return injected
    try:
        from tilegen.llm import LLMClient

        return LLMClient()
    except Exception:
        return None


def _confirm_promotion(
    src: str,
    task: tuple[VTMKernel, TargetSpec, KernelEvaluator],
    kb_root: Path,
    inner_budget: dict,
    incumbent_scores: list[float],
    seeds: list[int],
    budget: dict,
) -> bool:
    """Meta-level confirmation: re-score the winning variant on fresh
    seeds and require the latency LCB to beat the noise floor before the
    promotion record is finalized.  Speedups are converted to normalized
    times first (lc_bound is direction-correct for latency, lower-is-better).
    """
    from tilegen.evolve.noise import DEFAULT_MIN_FLOOR, lc_bound, robust_stats

    runs = int(budget.get("meta_confirm_runs", 5))
    if runs <= 0:
        return True # confirmation disabled by budget
    base = max(seeds) if seeds else 0
    confirm_seeds = [base + i + 1 for i in range(runs)]
    fit = _score_source(src, task, kb_root / "meta_work", confirm_seeds,
                        inner_budget)
    if fit is None:
        return False

    def _times(scores: list[float]) -> list[float]:
        return [1.0 / s for s in scores if s and s > 0]

    cand_times = _times(fit.get("seed_scores", []))
    inc_times = _times(list(incumbent_scores) or [1.0])
    if not cand_times or not inc_times:
        return False
    m_c, _, s_c = robust_stats(cand_times)
    m_r, _, s_r = robust_stats(inc_times)
    floor = max(DEFAULT_MIN_FLOOR, 2.0 * s_r / m_r) if m_r > 0 else DEFAULT_MIN_FLOOR
    lcb = lc_bound(m_c, s_c, m_r, s_r, k_c=1.0, k_r=2.0)
    return lcb is not None and lcb >= 1.0 + floor


def _fd_variants(
    source: str, rng: random.Random
) -> list[tuple[str, list[str]]]:
    """One directional probe per POLICY dim (finite-difference exploration).

    Each probe perturbs a single dim in its feasible direction (away from
    the cap), so the meta loop can attribute fitness deltas to dimensions.
    """
    values = _policy_values(source)
    out: list[tuple[str, list[str]]] = []
    for dim in sorted(_POLICY_KINDS):
        kind = _POLICY_KINDS[dim]
        cur = values.get(dim)
        if cur is None:
            continue
        if kind == "bool":
            direction = 0
        elif kind == "int":
            direction = -1 if int(cur) > 1 else +1
        else:
            direction = -1 if float(cur) >= 1.0 else +1
        new, notes = mutate_source(source, rng, bias={dim: direction})
        if new != source and notes:
            out.append((new, notes))
    return out


class _CollectingEvaluator:
    """Evaluator proxy that records candidate telemetry (inner campaigns)."""

    def __init__(self, inner: KernelEvaluator):
        self.inner = inner
        self.telemetry: list[dict] = []

    def profile_reference(self, sealed_ref_id: str, target: TargetSpec):
        return self.inner.profile_reference(sealed_ref_id, target)

    def evaluate(
        self, candidates, target: TargetSpec, sealed_ref_id: Optional[str] = None
    ):
        results = self.inner.evaluate(candidates, target, sealed_ref_id)
        for r in results:
            if getattr(r, "telemetry", None):
                self.telemetry.append(r.telemetry)
        return results


def _meta_dir(kb_root: Path) -> Path:
    d = kb_root / "meta"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ledger_path(kb_root: Path) -> Path:
    return _meta_dir(kb_root) / "ledger.json"


def _promote(
    source: str,
    fitness: dict,
    kb_root: Path,
    notes: list[str],
    task_key: Optional[str] = None,
) -> str:
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    archive = _meta_dir(kb_root) / f"zero_{sha}.py"
    archive.write_text(source, encoding="utf-8")

    ledger = _ledger_path(kb_root)
    if ledger.exists():
        try:
            entries = json.loads(ledger.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            entries = []
    else:
        entries = []
    entries.append({
        "sha": sha,
        "fitness": fitness,
        "notes": notes,
        "promoted_at": time.time(),
    })
    ledger.write_text(json.dumps(entries, indent=2), encoding="utf-8")

# Cross-task policy memory: remember the best POLICY per
    # task signature so similar tasks warm-start instead of re-searching.
    if task_key:
        (_meta_dir(kb_root) / f"policy_{task_key}.json").write_text(
            json.dumps({
                "task_key": task_key,
                "policy": _policy_values(source),
                "fitness": fitness,
            }, indent=2),
            encoding="utf-8",
        )

    tmp = ZERO_PATH.with_suffix(".py.tmp")
    tmp.write_text(source, encoding="utf-8")
    os.replace(tmp, ZERO_PATH)
    return sha


def _park_review(
    source: str, notes: list[str], fitness: dict, kb_root: Path,
    task_key: Optional[str],
) -> str:
    """Park an LLM proposal for human review: archive + record."""
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    _meta_dir(kb_root)
    (kb_root / "meta" / f"review_pending_{sha}.py").write_text(
        source, encoding="utf-8")
    (kb_root / "meta" / f"review_pending_{sha}.json").write_text(
        json.dumps({
            "sha": sha, "task_key": task_key, "notes": notes,
            "fitness": fitness, "parked_at": time.time(),
        }, indent=2), encoding="utf-8")
    return sha


def load_best_policy(kb_root: Path, task_key: str) -> Optional[dict[str, Any]]:
    """Best known POLICY for a task signature, or None."""
    path = _meta_dir(kb_root) / f"policy_{task_key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("policy") if isinstance(data.get("policy"), dict) else None


class MetaEvolverEngine(SynthesisEngine):
    """Evolves zero.py (the Zero evolver); never touches kernels directly."""

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
        del sealed_ref_id # the meta loop re-seals per seed via fresh KB roots
        budget = budget or {}
        meta_generations = budget.get("meta_generations", 3)
        meta_candidates = budget.get("meta_candidates", 4)
        seeds = budget.get("meta_seeds", 3)
        seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
# Note: no default inner budget. When the caller pins no keys, each
# zero.py variant falls back to its POLICY defaults, so budget-knob
        # mutations actually change behavior and can be selected for.
        inner_budget = dict(budget.get("inner_budget") or {})
        smoke_budget = {"generations": 1, "candidates_per_generation": 2}
        kb_root = Path(kb.root) if isinstance(kb, FileKnowledgeStore) \
            else Path("conf/kb")
        campaign_ts = time.time()

        rng = random.Random(self._seed)
        task = (initial_kernel, target, evaluator)
        incumbent = ZERO_PATH.read_text(encoding="utf-8") if ZERO_PATH.exists() \
            else _bootstrap_source()
        task_key = _task_key(initial_kernel, target)
        if budget.get("warm_start_policy", True):
            stored = load_best_policy(kb_root, task_key)
            if stored:
                incumbent = _set_policy(incumbent, stored)
        best_fit = _score_source(incumbent, task, kb_root / "meta_work", seeds,
                                 smoke_budget)
        best_fit = best_fit or {"median_speedup": 0.0, "seed_scores": [],
                                "best_candidate": None, "telemetry_agg": {},
                                "source_len": len(incumbent)}
        mutation_bias = policy_bias(best_fit.get("telemetry_agg", {}))
        best_sha = hashlib.sha256(incumbent.encode("utf-8")).hexdigest()[:12]
        best_notes = ["bootstrap"]
        history: list[dict] = []

        champion: dict[str, Any] = {}
        if budget.get("champion_speedup") is not None:
            champion = {"speedup": float(budget["champion_speedup"]),
                        "engine": "override"}
        elif budget.get("beat_builtin_engines"):
            champion = champion_baseline(
                task, kb_root, inner_budget, seeds,
                recompute=bool(budget.get("recompute_champion")),
            )
        victory = False

        fd_first_gen = budget.get("meta_strategy", "hybrid") in (
            "finite_difference", "hybrid")
        for gen in range(meta_generations):
# Stage A (successive halving): screen every variant
# at one seed with the smoke budget. The first generation uses
# finite-difference probes (one per POLICY dim);
            # later generations use telemetry-biased random mutation.
            screened: list[tuple[str, list[str], dict]] = []
            llm_on = budget.get("llm_mutations", False) and (
                gen == 0 or budget.get("llm_every_generation", False))
            variants: list[tuple[str, list[str], str]] = []
            if llm_on:
                client = _llm_client(budget)
                if client is not None:
                    context = {
                        "best_median_speedup": best_fit["median_speedup"],
                        "champion_speedup": champion.get("speedup"),
                        "champion_engine": champion.get("engine"),
                        "mutation_bias": mutation_bias,
                        "recent_history": history[-6:],
                    }
                    proposed = llm_propose_policy(incumbent, context, client)
                    if proposed is not None:
                        variants.append((proposed[0], proposed[1], "llm"))
            if not variants:
                if fd_first_gen and gen == 0:
                    variants = [(src, notes, "fd")
                                for src, notes in _fd_variants(incumbent, rng)]
                else:
                    variants = [
                        (src, notes, "rule")
                        for src, notes in (
                            (mutate_source(incumbent, rng,
                                           bias=mutation_bias or None)
                             for _ in range(meta_candidates))
                        )
                    ]
            for src, notes, origin in variants:
                if static_gate(src):
                    history.append({"gen": gen, "stage": "static_gate", "notes": notes, "ok": False})
                    continue
                if smoke_gate(src):
                    history.append({"gen": gen, "stage": "smoke_gate", "notes": notes, "ok": False})
                    continue
                fit = _score_source(src, task, kb_root / "meta_work", seeds[:1],
                                    smoke_budget)
                if fit is None:
                    history.append({"gen": gen, "stage": "screen",
                                    "origin": origin, "notes": notes,
                                    "ok": False})
                    continue
                screened.append((src, notes, fit, origin))
                history.append({"gen": gen, "stage": "screen", "tier": "screen",
                                "origin": origin, "notes": notes, "ok": True})

            if not screened:
                continue
            # Stage B: keep the top half and confirm at all seeds.
            screened.sort(key=lambda t: (-t[2]["median_speedup"], len(t[0])))
            keep = max(1, (len(screened) + 1) // 2)
            survivors: list[tuple[str, list[str], dict, str]] = []
            for src, notes, fit, origin in screened[:keep]:
                fit_full = _score_source(src, task, kb_root / "meta_work",
                                         seeds, inner_budget)
                if fit_full is None:
                    history.append({"gen": gen, "stage": "full", "tier": "full",
                                    "origin": origin, "notes": notes,
                                    "ok": False})
                    continue
                survivors.append((src, notes, fit_full, origin))
                history.append({"gen": gen, "stage": "full", "tier": "full",
                                "origin": origin, "notes": notes, "ok": True})
            for src, notes, fit, origin in survivors:
                penalty = _COMPLEXITY_WEIGHT * max(
                    len(src) - len(incumbent), 0) / 100.0
                adjusted = fit["median_speedup"] - penalty
                bar = max(best_fit["median_speedup"],
                          champion.get("speedup", 0.0))
                if origin == "llm" and budget.get("require_review"):
# LLM proposals require human review — park the
                    # best one instead of promoting it.
                    _park_review(src, notes, fit, kb_root, task_key)
                    history.append({"gen": gen, "stage": "review_required",
                                    "notes": notes, "ok": True})
                    continue
                if adjusted > bar:
                    # Confirm on fresh seeds before finalizing.
                    if not _confirm_promotion(
                        src, task, kb_root, inner_budget,
                        best_fit.get("seed_scores") or [], seeds, budget,
                    ):
                        history.append({"gen": gen, "stage": "confirm",
                                        "notes": notes, "ok": False})
                        continue
                    best_fit = fit
                    best_sha = _promote(src, fit, kb_root, notes, task_key)
                    best_notes = notes
                    incumbent = src
                    mutation_bias = policy_bias(fit.get("telemetry_agg", {}))
                    if champion and adjusted > champion["speedup"]:
                        victory = True
            if victory:
                break

        record = {
            "campaign_ts": campaign_ts,
            "task_key": task_key,
            "best_sha": best_sha,
            "promoted": best_notes[0] != "bootstrap" if best_notes else False,
            "champion_speedup": champion.get("speedup"),
            "champion_engine": champion.get("engine"),
            "victory": victory,
            "history": history,
        }
        _meta_dir(kb_root)
        (_meta_dir(kb_root) / f"campaign_{campaign_ts:.0f}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

        winner = best_fit.get("best_candidate")
        if champion.get("candidate") is not None and not victory:
            winner = champion["candidate"]
        if winner is None:
            winner = KernelCandidate(
                candidate_id=f"zero-{best_sha}", parent_ids=["meta_evolver"],
                vtm_ir=initial_kernel,
                rationale="; ".join(best_notes) + (" [victory]" if victory else ""),
            )
        return winner


def _bootstrap_source() -> str:
    """Fallback when zero.py is missing: refuse clearly rather than fabricate."""
    raise RuntimeError(
        f"{ZERO_PATH} is missing; the Zero evolver artifact must exist before "
        "the Meta-Evolver can run (it is shipped with the repository)."
    )


__all__ = [
    "MetaEvolverEngine",
    "ZERO_PATH",
    "static_gate",
    "smoke_gate",
    "mutate_source",
    "llm_propose_policy",
    "_fd_variants",
    "_confirm_promotion",
    "policy_bias",
    "load_best_policy",
    "_promote",
    "_policy_values",
    "_set_policy",
]
