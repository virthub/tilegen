# tests/test_cli_branches.py
#
# Table-driven coverage for the CLI command branches (§24.2.2 of
# tilegen.md): dispatch wiring, compile, and evolve paths with the heavy
# backends/evaluators mocked — no GPU required.

import pytest

from tilegen import cli

_CONF = "conf/benchmark/llama3_70b.yaml"


class _FakeBackend:
    name = "fake"

    def lower(self, kernel, target):
        del target
        return f"# lowered {kernel.name}\n"

    def dump_ir(self, kernel, target):
        del kernel, target
        return "ir-text"


class _FakeEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    def profile_reference(self, sealed_ref_id, target):
        del sealed_ref_id, target
        return {"device_time_us": 100.0}

    def evaluate(self, candidates, target, sealed_ref_id=None):
        del target, sealed_ref_id
        from tilegen.evolve.interface import EvalResult

        return [EvalResult(c.candidate_id, True, True, device_time_us=100.0,
                           speedup_vs_ref=1.0) for c in candidates]


class _FakeEngine:
    def run_campaign(self, initial_kernel, target, evaluator, kb,
                     sealed_ref_id=None, budget=None):
        del target, evaluator, kb, sealed_ref_id, budget
        from tilegen.evolve.interface import KernelCandidate

        return KernelCandidate("fake_best", [], initial_kernel,
                               rationale="table-driven cli test")


class TestDispatch:
    def test_required_subcommand(self):
        with pytest.raises(SystemExit):
            cli.main([])

    def test_unknown_command_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["definitely_not_a_command"])
        assert exc.value.code == 2


class TestCompileBranch:
    def test_compile_writes_artifact(self, tmp_path, monkeypatch,
                                     capsys):
        monkeypatch.setenv("TILEGEN_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(cli, "_require_cuda", lambda: True)
        import tilegen.backend

        monkeypatch.setattr(tilegen.backend, "get_backend",
                            lambda name: _FakeBackend())
        rc = cli.main(["compile", "--conf", _CONF, "--emit",
                       str(tmp_path / "out.py")])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[tilegen] semantic_id" in out
        assert (tmp_path / "out.py").read_text(encoding="utf-8").startswith(
            "# lowered")


class _FakeVerifyReport:
    def __init__(self, passed):
        self.all_passed = passed

    def summary(self):
        return f"verify summary (passed={self.all_passed})"


class _FakeAutotuneResult:
    def __init__(self, tile_m, tile_n, tile_k, stages, warps, us):
        self.tile_m, self.tile_n, self.tile_k = tile_m, tile_n, tile_k
        self.pipeline_stages, self.num_warps = stages, warps
        self.device_time_us = us
        self.config = self

    def to_dict(self):
        return {"tile_m": self.tile_m, "tile_n": self.tile_n,
                "tile_k": self.tile_k, "pipeline_stages": self.pipeline_stages,
                "num_warps": self.num_warps}


class TestVerifyBranch:
    def test_verify_passes(self, monkeypatch, capsys):
        import tilegen.ops.verify as verify_mod

        monkeypatch.setattr(verify_mod, "verify",
                            lambda spec, target, speed_gate=None:
                            _FakeVerifyReport(True))
        rc = cli.main(["verify", "--conf", _CONF])
        assert rc == 0
        assert "verify summary" in capsys.readouterr().out

    def test_verify_fails_gate(self, monkeypatch):
        import tilegen.ops.verify as verify_mod

        monkeypatch.setattr(verify_mod, "verify",
                            lambda spec, target, speed_gate=None:
                            _FakeVerifyReport(False))
        assert cli.main(["verify", "--conf", _CONF]) == 1


class TestAutotuneBranch:
    def test_no_cuda_path(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_require_cuda", lambda: False)
        rc = cli.main(["autotune", "--conf", _CONF])
        assert rc == 1
        assert "pruned" in capsys.readouterr().out

    def test_full_path_with_mocks(self, tmp_path, monkeypatch, capsys):
        import torch

        import tilegen.autotune as autotune_mod
        import tilegen.ops.flash as flash_mod

        monkeypatch.setattr(cli, "_require_cuda", lambda: True)
        monkeypatch.setattr(
            autotune_mod, "autotune",
            lambda configs, profile, warmup=2, iters=5: [
                _FakeAutotuneResult(64, 64, 32, 2, 4, 12.3)])
        monkeypatch.setattr(autotune_mod, "save_recipe",
                            lambda out, target, best, extra=None: None)
        monkeypatch.setattr(
            flash_mod, "flash_attention_forward",
            lambda q, k, v, is_causal=True, **kw: (torch.zeros_like(q), None, None))
        out = tmp_path / "recipe.json"
        rc = cli.main(["autotune", "--conf", _CONF, "--batch", "1",
                       "--heads", "2", "--seq-len", "32", "--head-dim", "32",
                       "--out", str(out)])
        assert rc == 0
        captured = capsys.readouterr().out
        assert "best:" in captured


class TestBenchBranch:
    def test_no_cuda_path(self, monkeypatch, capsys):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        rc = cli.main(["bench", "--batch", "1", "--heads", "2",
                       "--seq-len", "32", "--head-dim", "32"])
        assert rc == 1
        assert "no GPU visible" in capsys.readouterr().err

    def test_full_path_with_mocks(self, monkeypatch, capsys):
        import torch

        import tilegen.ops.flash as flash_mod

        monkeypatch.setattr(
            flash_mod, "flash_attention_forward",
            lambda q, k, v, is_causal=True, **kw: (torch.zeros_like(q), None, None))
        rc = cli.main(["bench", "--batch", "1", "--heads", "2",
                       "--seq-len", "32", "--head-dim", "32", "--iters", "2"])
        assert rc == 0
        captured = capsys.readouterr().out
        assert "TFLOPS" in captured


class TestEvolveBranch:
    def test_evolve_with_mocks(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("TILEGEN_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(cli, "_require_cuda", lambda: True)
        import tilegen.evolve.evaluator as evaluator_mod
        import tilegen.evolve.strategies as strategies_mod

        monkeypatch.setattr(evaluator_mod, "TritonAttentionEvaluator",
                            _FakeEvaluator)
        monkeypatch.setattr(strategies_mod, "get_engine",
                            lambda name: _FakeEngine())
        rc = cli.main(["evolve", "--strategy", "forgekernel",
                       "--conf", _CONF, "--kb", str(tmp_path / "kb")])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[tilegen] best: fake_best" in out
