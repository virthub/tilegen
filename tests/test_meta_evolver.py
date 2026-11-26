# tests/test_meta_evolver.py
#
# Tests for the Meta-Evolver (the evolver of the evolver) and the Zero
# evolver artifact it develops.

import json
import random
from pathlib import Path

from tilegen.evolve.interface import EvalResult
from tilegen.evolve.kb_manager import FileKnowledgeStore
from tilegen.evolve.meta_evolver import (
    MetaEvolverEngine,
    ZERO_PATH,
    _confirm_promotion,
    _fd_variants,
    _policy_values,
    _promote,
    _set_policy,
    champion_baseline,
    llm_propose_policy,
    load_best_policy,
    mutate_source,
    policy_bias,
    smoke_gate,
    static_gate,
)
from tilegen.evolve.strategies import get_engine
from tilegen.evolve.zero import POLICY, ZeroEngine, ZeroEvolveEngine
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec, build_attention_vtm
from tilegen.target.spec import get_target

_ORIGINAL_ZERO = Path(__file__).resolve().parent.parent / "src" / "tilegen" / "evolve" / "zero.py"


def _initial_kernel():
    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    return build_attention_vtm(spec, get_target("sm_90a"), variant="fa2_prefill")


class _FlatEvaluator:
    """Every correct candidate times identically (slightly better than ref)."""

    def __init__(self, time_us=90.0, ref_time=100.0):
        self.time_us = time_us
        self.ref_time = ref_time

    def profile_reference(self, sealed_ref_id, target):
        del sealed_ref_id, target
        return {"device_time_us": self.ref_time}

    def evaluate(self, candidates, target, sealed_ref_id=None):
        del target, sealed_ref_id
        return [
            EvalResult(
                c.candidate_id, True, True, device_time_us=self.time_us,
                speedup_vs_ref=self.ref_time / self.time_us,
            )
            for c in candidates
        ]


class TestZeroEvolver:
    def test_campaign_runs_and_promotes(self, tmp_path):
        kb = FileKnowledgeStore(tmp_path)
        cand = ZeroEngine(seed=0).run_campaign(
            _initial_kernel(), get_target("sm_90a"), _FlatEvaluator(), kb,
            budget={"generations": 1, "candidates_per_generation": 2},
        )
        assert cand.candidate_id.startswith("gen0_c")
        assert any(p.name.startswith("reference_") for p in tmp_path.glob("*.json"))

    def test_policy_kwargs_override(self):
        engine = ZeroEngine(policy={**POLICY, "plateau_window": 99})
        assert engine.policy["plateau_window"] == 99
        assert engine.policy["variant_mutation_prob"] == POLICY["variant_mutation_prob"]


class TestChampionAndVictory:
    def test_champion_baseline_and_cache(self, tmp_path):
        kb_root = tmp_path / "kb"
        task = (_initial_kernel(), get_target("sm_90a"), _FlatEvaluator())
        best = champion_baseline(task, kb_root, {"generations": 1, "candidates_per_generation": 2}, [0])
        assert best["engine"] in ("forgekernel", "evolutionary", "world_model",
                                  "goal_loop", "rl_rollout", "schedule_codesign")
        assert best["speedup"] > 1.0
        cached = champion_baseline(task, kb_root, {"generations": 1, "candidates_per_generation": 2}, [0])
        assert cached["engine"] == best["engine"]
        assert (kb_root / "meta" / "champion.json").exists()

    def test_zero_evolve_engine_records_champion(self, tmp_path, monkeypatch):
        kb = FileKnowledgeStore(tmp_path / "kb")
        zero = tmp_path / "zero.py"
        zero.write_text(_ORIGINAL_ZERO.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr("tilegen.evolve.meta_evolver.ZERO_PATH", zero)

        engine = ZeroEvolveEngine(seed=0)
        cand = engine.run_campaign(
            _initial_kernel(), get_target("sm_90a"), _FlatEvaluator(), kb,
            budget={
                "meta_generations": 1,
                "meta_candidates": 1,
                "meta_seeds": 1,
                "champion_speedup": 1.5, # explicit bar above all variant scores
                "inner_budget": {"generations": 1, "candidates_per_generation": 2},
            },
        )
        records = list((tmp_path / "kb" / "meta").glob("campaign_*.json"))
        assert records
        record = json.loads(records[0].read_text(encoding="utf-8"))
        assert record["champion_speedup"] == 1.5
        assert record["victory"] is False
        assert cand is not None


class TestSmartMetaLoop:
    def test_policy_bias_mapping(self):
        bias = policy_bias({"tensorcore_idle": 1.0, "sfu_stall": 0.0})
        assert bias == {"default_candidates_per_generation": 1,
                        "bias_strength": 1}
        assert policy_bias({}) == {}

    def test_mutate_source_bias_restricts_dims(self):
        src = ZERO_PATH.read_text(encoding="utf-8")
        rng = random.Random(0)
        for _ in range(10):
            new, notes = mutate_source(
                src, rng, bias={"default_generations": +1})
            assert notes
            for n in notes:
                assert n.startswith("default_generations:")

    def test_policy_values_roundtrip(self):
        src = ZERO_PATH.read_text(encoding="utf-8")
        values = _policy_values(src)
        assert values["plateau_window"] == 2
        new = _set_policy(src, {"plateau_window": 9, "unknown_key": 3})
        assert _policy_values(new)["plateau_window"] == 9
        assert "unknown_key" not in _policy_values(new)
        assert "# src/tilegen/evolve/zero.py" in new
        assert static_gate(new) == []

    def test_policy_memory_store_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tilegen.evolve.meta_evolver.ZERO_PATH",
                            tmp_path / "zero.py")
        src = _ORIGINAL_ZERO.read_text(encoding="utf-8")
        _promote(src, {"median_speedup": 1.1}, tmp_path / "kb", ["t"],
                 task_key="tk42")
        stored = load_best_policy(tmp_path / "kb", "tk42")
        assert stored is not None and stored["plateau_window"] == 2
        assert load_best_policy(tmp_path / "kb", "missing") is None

    def test_warm_start_policy_applied(self, tmp_path, monkeypatch):
        from tilegen.evolve.meta_evolver import _task_key

        kb = FileKnowledgeStore(tmp_path / "kb")
        zero = tmp_path / "zero.py"
        zero.write_text(_ORIGINAL_ZERO.read_text(encoding="utf-8"),
                        encoding="utf-8")
        monkeypatch.setattr("tilegen.evolve.meta_evolver.ZERO_PATH", zero)
        kernel = _initial_kernel()
        target = get_target("sm_90a")
        tk = _task_key(kernel, target)
        (tmp_path / "kb" / "meta").mkdir(parents=True, exist_ok=True)
        import json as _json

        (tmp_path / "kb" / "meta" / f"policy_{tk}.json").write_text(
            _json.dumps({"task_key": tk,
                         "policy": {"plateau_window": 7}, "fitness": {}}))
        engine = MetaEvolverEngine(seed=0)
        engine.run_campaign(
            kernel, target, _FlatEvaluator(), kb,
            budget={"meta_generations": 1, "meta_candidates": 1,
                    "meta_seeds": 1,
                    "inner_budget": {"generations": 1, "candidates_per_generation": 2}},
        )
        # The campaign record names the warm-start task key.
        records = list((tmp_path / "kb" / "meta").glob("campaign_*.json"))
        record = _json.loads(records[0].read_text(encoding="utf-8"))
        assert record.get("task_key") == tk

    def test_fd_variants_probe_each_dim(self):
        from tilegen.evolve.meta_evolver import _POLICY_KINDS

        src = ZERO_PATH.read_text(encoding="utf-8")
        variants = _fd_variants(src, random.Random(0))
        dims = []
        for new, notes in variants:
            assert notes and len(notes) == 1
            dims.append(notes[0].split(":")[0])
            assert new != src
            assert static_gate(new) == []
        assert sorted(dims) == sorted(_POLICY_KINDS)

    def test_fd_first_generation_screens_all_dims(self, tmp_path, monkeypatch):
        import json as _json

        kb = FileKnowledgeStore(tmp_path / "kb")
        zero = tmp_path / "zero.py"
        zero.write_text(_ORIGINAL_ZERO.read_text(encoding="utf-8"),
                        encoding="utf-8")
        monkeypatch.setattr("tilegen.evolve.meta_evolver.ZERO_PATH", zero)
        engine = MetaEvolverEngine(seed=0)
        engine.run_campaign(
            _initial_kernel(), get_target("sm_90a"), _FlatEvaluator(), kb,
            budget={"meta_generations": 1, "meta_candidates": 1,
                    "meta_seeds": 1,
                    "inner_budget": {"generations": 1, "candidates_per_generation": 2}},
        )
        records = list((tmp_path / "kb" / "meta").glob("campaign_*.json"))
        record = _json.loads(records[0].read_text(encoding="utf-8"))
        screened = [h for h in record["history"]
                    if h.get("stage") == "screen" and h.get("ok")]
        dims = {h["notes"][0].split(":")[0] for h in screened}
        assert len(dims) == 6 # one finite-difference probe per POLICY dim


class TestMetaConfirmation:
    def test_accepts_clear_winner(self, monkeypatch):
        import tilegen.evolve.meta_evolver as me

        # Variant scores 1.2x, incumbent 1.0x -> latency LCB 1.2 >= 1.015.
        monkeypatch.setattr(
            me, "_score_source",
            lambda src, task, kb_root, seeds, inner_budget: {
                "seed_scores": [1.2] * len(seeds),
                "median_speedup": 1.2,
            },
        )
        assert me._confirm_promotion(
            "src", (_initial_kernel(), get_target("sm_90a"), _FlatEvaluator()),
            __import__("pathlib").Path("/tmp"), {"generations": 1},
            [1.0, 1.0, 1.0], [0], {"meta_confirm_runs": 5},
        ) is True

    def test_rejects_below_floor(self, monkeypatch):
        import tilegen.evolve.meta_evolver as me

        monkeypatch.setattr(
            me, "_score_source",
            lambda src, task, kb_root, seeds, inner_budget: {
                "seed_scores": [1.0] * len(seeds),
                "median_speedup": 1.0,
            },
        )
        assert me._confirm_promotion(
            "src", (_initial_kernel(), get_target("sm_90a"), _FlatEvaluator()),
            __import__("pathlib").Path("/tmp"), {"generations": 1},
            [1.2, 1.2, 1.2], [0], {"meta_confirm_runs": 5},
        ) is False

    def test_disabled_by_budget(self, monkeypatch):
        import tilegen.evolve.meta_evolver as me

        assert me._confirm_promotion(
            "src", (_initial_kernel(), get_target("sm_90a"), _FlatEvaluator()),
            __import__("pathlib").Path("/tmp"), {"generations": 1},
            [1.0], [0], {"meta_confirm_runs": 0},
        ) is True


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompt = None

    def complete(self, prompt, timeout=None):
        self.prompt = prompt
        return self.reply


class TestLLMProposals:
    def test_parses_code_fenced_dict(self):
        src = ZERO_PATH.read_text(encoding="utf-8")
        fake = _FakeLLM("```python\nPOLICY = {\"plateau_window\": 7, "
                        "\"bias_strength\": 0.6}\n```")
        new, notes = llm_propose_policy(src, {"x": 1}, fake)
        assert new is not None and notes
        from tilegen.evolve.meta_evolver import _policy_values

        values = _policy_values(new)
        assert values["plateau_window"] == 7
        assert values["bias_strength"] == 0.6
        assert "# src/tilegen/evolve/zero.py" in new
        assert static_gate(new) == []

    def test_rejects_garbage_and_unknown_keys(self):
        src = ZERO_PATH.read_text(encoding="utf-8")
        assert llm_propose_policy(src, {}, _FakeLLM("no dict here")) is None
        assert llm_propose_policy(src, {}, _FakeLLM('{"nope": 1}')) is None

    def test_client_failure_returns_none(self):
        class _Boom:
            def complete(self, prompt, timeout=None):
                raise RuntimeError("no network")

        assert llm_propose_policy(
            ZERO_PATH.read_text(encoding="utf-8"), {}, _Boom()) is None

    def test_review_flag_parks_llm_variant(self, tmp_path, monkeypatch):
        import json as _json

        import tilegen.evolve.meta_evolver as me

        kb = FileKnowledgeStore(tmp_path / "kb")
        zero = tmp_path / "zero.py"
        original = _ORIGINAL_ZERO.read_text(encoding="utf-8")
        zero.write_text(original, encoding="utf-8")
        monkeypatch.setattr(me, "ZERO_PATH", zero)
# Make every scored source a clear winner so the LLM variant reaches
        # the promotion stage.
        monkeypatch.setattr(
            me, "_score_source",
            lambda src, task, kb_root, seeds, inner_budget: {
                "seed_scores": [2.0] * len(seeds),
                "median_speedup": 2.0,
            })
        fake = _FakeLLM('{"plateau_window": 9}')
        engine = MetaEvolverEngine(seed=0)
        engine.run_campaign(
            _initial_kernel(), get_target("sm_90a"), _FlatEvaluator(), kb,
            budget={"meta_generations": 1, "meta_candidates": 1,
                    "meta_seeds": 1,
                    "inner_budget": {"generations": 1, "candidates_per_generation": 2},
                    "llm_mutations": True, "require_review": True,
                    "llm_client": fake,
                    "meta_confirm_runs": 0},
        )
        # The parked proposal exists; zero.py was NOT overwritten.
        parked = list((tmp_path / "kb" / "meta").glob("review_pending_*.py"))
        assert len(parked) == 1
        assert zero.read_text(encoding="utf-8") == original
        records = list((tmp_path / "kb" / "meta").glob("campaign_*.json"))
        record = _json.loads(records[0].read_text(encoding="utf-8"))
        assert any(h.get("stage") == "review_required"
                   for h in record["history"])

    def test_llm_variant_promotes_without_review(self, tmp_path, monkeypatch):
        import tilegen.evolve.meta_evolver as me

        kb = FileKnowledgeStore(tmp_path / "kb")
        zero = tmp_path / "zero.py"
        original = _ORIGINAL_ZERO.read_text(encoding="utf-8")
        zero.write_text(original, encoding="utf-8")
        monkeypatch.setattr(me, "ZERO_PATH", zero)
# The incumbent scores 1.0; the LLM variant (plateau_window 9) scores
        # 9.0, so it strictly beats the promotion bar.
        def _fake_score(src, task, kb_root, seeds, inner_budget):
            # _render_policy uses repr(), so the literal is single-quoted.
            value = 9.0 if "'plateau_window': 9" in src else 1.0
            return {"seed_scores": [value] * len(seeds),
                    "median_speedup": value}

        monkeypatch.setattr(me, "_score_source", _fake_score)
        fake = _FakeLLM('{"plateau_window": 9}')
        engine = MetaEvolverEngine(seed=0)
        engine.run_campaign(
            _initial_kernel(), get_target("sm_90a"), _FlatEvaluator(), kb,
            budget={"meta_generations": 1, "meta_candidates": 1,
                    "meta_seeds": 1,
                    "inner_budget": {"generations": 1, "candidates_per_generation": 2},
                    "llm_mutations": True, "require_review": False,
                    "llm_client": fake,
                    "meta_confirm_runs": 0},
        )
        from tilegen.evolve.meta_evolver import _policy_values

        assert _policy_values(zero.read_text(encoding="utf-8"))[
            "plateau_window"] == 9


class TestChampionCacheRoundtrip:
    def test_candidate_roundtrips_through_cache(self, tmp_path):
        import json as _json

        from tilegen.evolve.adapter import candidate_to_dict
        from tilegen.evolve.interface import KernelCandidate
        from tilegen.evolve.meta_evolver import _task_key

        kernel = _initial_kernel()
        target = get_target("sm_90a")
        task = (kernel, target, _FlatEvaluator())
        tk = _task_key(kernel, target)
        cand = KernelCandidate("champ_c", [], kernel, rationale="champion")
        # Write a cache entry carrying a candidate, then read it back.
        meta_dir = tmp_path / "kb" / "meta"
        meta_dir.mkdir(parents=True)
        (meta_dir / "champion.json").write_text(
            _json.dumps({"task_key": tk,
                         "result": {"speedup": 1.2, "engine": "rl_rollout",
                                    "candidate": candidate_to_dict(cand)}}))
        best = champion_baseline(task, tmp_path / "kb",
                                 {"generations": 1}, [0])
        assert best["candidate"] is not None
        assert best["candidate"].candidate_id == "champ_c"
        assert best["candidate"].rationale == "champion"
        assert best["speedup"] == 1.2


class TestMetaEvolver:
    def test_get_engine_registration(self):
        assert isinstance(get_engine("zero"), ZeroEvolveEngine)
        assert isinstance(get_engine("meta_evolver"), MetaEvolverEngine)

    def test_static_gate(self):
        assert static_gate(ZERO_PATH.read_text(encoding="utf-8")) == []
        assert static_gate("import os\n") == ["import os"]
        assert static_gate("x = eval('1')\n") == ["call eval"]
        assert static_gate("from subprocess import run\n") == ["from subprocess import ..."]

    def test_smoke_gate(self):
        assert smoke_gate(ZERO_PATH.read_text(encoding="utf-8")) is None
        assert smoke_gate("import os\n") is not None
        assert smoke_gate("def broken(:\n") is not None

    def test_mutate_source_preserves_header(self):
        src = ZERO_PATH.read_text(encoding="utf-8")
        new, notes = mutate_source(src, random.Random(0))
        assert notes
        assert "# src/tilegen/evolve/zero.py" in new # header preserved verbatim
        assert "class ZeroEngine" in new
        assert static_gate(new) == []

    def test_promote_writes_archive_ledger_and_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tilegen.evolve.meta_evolver.ZERO_PATH", tmp_path / "zero.py")
        src = _ORIGINAL_ZERO.read_text(encoding="utf-8")
        sha = _promote(src, {"median_speedup": 1.1, "seed_scores": [1.1]},
                       tmp_path / "kb", ["t: a -> b"])
        assert (tmp_path / "zero.py").read_text(encoding="utf-8") == src
        assert (tmp_path / "kb" / "meta" / f"zero_{sha}.py").exists()
        ledger = json.loads((tmp_path / "kb" / "meta" / "ledger.json").read_text(encoding="utf-8"))
        assert ledger[-1]["sha"] == sha
        assert ledger[-1]["notes"] == ["t: a -> b"]

    def test_campaign_completes_with_artifact(self, tmp_path, monkeypatch):
        kb = FileKnowledgeStore(tmp_path / "kb")
        zero = tmp_path / "zero.py"
        zero.write_text(_ORIGINAL_ZERO.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr("tilegen.evolve.meta_evolver.ZERO_PATH", zero)

        engine = MetaEvolverEngine(seed=0)
        cand = engine.run_campaign(
            _initial_kernel(), get_target("sm_90a"), _FlatEvaluator(), kb,
            budget={
                "meta_generations": 1,
                "meta_candidates": 2,
                "meta_seeds": 1,
                "inner_budget": {"generations": 1, "candidates_per_generation": 2},
            },
        )
        assert cand is not None and cand.vtm_ir is not None
        # Whatever was promoted (or the bootstrap copy) must still parse.
        assert static_gate(zero.read_text(encoding="utf-8")) == []
        # The campaign record is always written, even without a promotion.
        records = list((tmp_path / "kb" / "meta").glob("campaign_*.json"))
        assert records
        record = json.loads(records[0].read_text(encoding="utf-8"))
        assert record["best_sha"]
        assert isinstance(record["history"], list)
