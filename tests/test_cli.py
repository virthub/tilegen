# tests/test_cli.py
#
# Unit tests for the unified tilegen.cli command-line interface

import textwrap
import pytest
import tilegen.cli as cli
from pathlib import Path


def _write_conf(tmp_path: Path) -> Path:
    conf = tmp_path / "compile.yaml"
    conf.write_text(
        textwrap.dedent(
            """
            target: sm_100a
            backend: tilelang
            attention:
              batch: 1
              num_heads: 8
              num_kv_heads: 8
              head_dim: 128
              kv_head_dim: 64
              seq_len: 2048
              num_kv_blocks: 256
              page_size: 32
              dtype: bf16
              kv_quant:
                dtype: fp4_e2m1
                granularity: block
                block_size: 16
                scale_dtype: fp8_e4m3
                pack_factor: 2
            """
        ),
        encoding="utf-8",
    )
    return conf


class TestCompile:
    def test_compile_end_to_end(self, tmp_path, monkeypatch, capsys):
        conf = _write_conf(tmp_path)
        monkeypatch.setenv("TILEGEN_CACHE_DIR", str(tmp_path / "cache"))
        ir = tmp_path / "kernel.tilelang.py"

        rc = cli.main(["compile", "--conf", str(conf), "--backend-ir", "tilelang", "--dump-ir", str(ir)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "semantic_id" in out
        assert "paged_flash_attn_fp4" in out

        # Backend IR was dumped (TileLang source by default).
        assert ir.exists() and "import tilelang" in ir.read_text()
        # A lowered artifact was written next to the manifest.
        artifacts = list((tmp_path / "cache" / "artifacts").glob("*.py"))
        assert len(artifacts) == 1
        manifests = list((tmp_path / "cache" / "manifests").glob("*.json"))
        assert len(manifests) == 1

    def test_compile_emit_and_variant(self, tmp_path, monkeypatch):
        conf = _write_conf(tmp_path)
        monkeypatch.setenv("TILEGEN_CACHE_DIR", str(tmp_path / "cache"))
        src = tmp_path / "out.py"
        rc = cli.main(["compile", "--conf", str(conf), "--backend-ir", "tilelang",
                       "--variant", "fa4", "--emit", str(src)])
        assert rc == 0
        assert src.exists() and "T.gemm" in src.read_text()

    def test_compile_missing_conf(self):
        assert cli.main(["compile", "--conf", "/nonexistent.yaml"]) != 0


class TestSelftest:
    def test_selftest_passes(self, capsys):
        assert cli.main(["selftest"]) == 0
        assert "selftest passed" in capsys.readouterr().out


class TestEvolveCli:
    @pytest.mark.skipif(
        __import__("torch").cuda.is_available(), reason="CUDA available; no-GPU path not applicable"
    )
    def test_evolve_requires_cuda(self, monkeypatch, capsys):
        # With a valid conf but no GPU, --evolve reports and returns non-zero.
        monkeypatch.setenv("TILEGEN_CACHE_DIR", "/tmp/tg-evolve-cache")
        rc = cli.main(["evolve", "--strategy", "forgekernel",
                       "--conf", "conf/benchmark/llama3_70b.yaml"])
        assert rc == 1
