# tests/test_profiler_backend.py
#
# Tests for the real-telemetry NCU profiler backend: CSV parsing,
# normalization, per-config caching, and failure fallback. Uses a fake
# `ncu` executable that prints a canned CSV and never runs the profiling
# script, so no GPU or Nsight Compute is required.

import json
import os
import stat

import pytest

from tilegen.evolve.profiler import NcuProfiler

_CANNED_CSV = '''"ID","Process","Kernel Name","Metric Name","Metric Value"
"0","python","k1","sm__warps_active.avg.pct_of_peak_sustained_active","45.0"
"0","python","k1","l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum","100.0"
"0","python","k1","smsp__inst_executed_pipe_xu.sum","12.5"
"0","python","k1","l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum","7.0"
"1","python","k1","sm__warps_active.avg.pct_of_peak_sustained_active","55.0"
'''


@pytest.fixture()
def fake_ncu(tmp_path):
    counter = tmp_path / "calls"
    counter.write_text("0", encoding="utf-8")
    script = tmp_path / "fake_ncu"
    script.write_text(
        "#!/bin/bash\n"
        f"echo $(( $(cat {counter}) + 1 )) > {counter}\n"
        f"cat <<'CSV'\n{_CANNED_CSV}CSV\n",
        encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, counter


def _tensors():
    import torch

    q = torch.zeros(1, 2, 32, 16)
    return q, q, q


class TestNcuProfiler:
    def test_parse_normalize_and_cache(self, tmp_path, fake_ncu):
        ncu_bin, counter = fake_ncu
        backend = NcuProfiler(ncu_bin=str(ncu_bin),
                              cache_dir=str(tmp_path / "cache"))
        out = backend.telemetry(
            "fa2_bf16:tile_m=128", "fa2_bf16", {"tile_m": 128},
            _tensors(), True, 1.0)
        # Two launches of the same metric averaged: (45 + 55) / 2 = 50.
        assert out["tensorcore_idle"] == pytest.approx(50.0)
        assert out["mbarrier_wait_stall"] == pytest.approx(100.0)
        assert out["sfu_stall"] == pytest.approx(12.5)
        assert out["register_spill"] == pytest.approx(7.0)
        assert out["cta_boundary_stall"] == 0.0 # unmapped symptom stays 0
        # Cache written; a second call is served from it (no ncu invocation).
        assert int(counter.read_text(encoding="utf-8")) == 1
        again = backend.telemetry(
            "fa2_bf16:tile_m=128", "fa2_bf16", {"tile_m": 128},
            _tensors(), True, 1.0)
        assert again == out
        assert int(counter.read_text(encoding="utf-8")) == 1

    def test_failure_returns_empty(self, tmp_path):
        bad = tmp_path / "bad_ncu"
        bad.write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
        bad.chmod(bad.stat().st_mode | stat.S_IEXEC)
        backend = NcuProfiler(ncu_bin=str(bad),
                              cache_dir=str(tmp_path / "cache"))
        assert backend.telemetry(
            "k", "fa2_bf16", {}, _tensors(), True, 1.0) == {}
        assert not list((tmp_path / "cache").glob("*.json"))

    def test_cache_roundtrip_json(self, tmp_path, fake_ncu):
        ncu_bin, _ = fake_ncu
        backend = NcuProfiler(ncu_bin=str(ncu_bin),
                              cache_dir=str(tmp_path / "cache"))
        backend.telemetry("k", "fa2_bf16", {}, _tensors(), True, 1.0)
        files = list((tmp_path / "cache" / "profiles").glob("*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["tensorcore_idle"] == pytest.approx(50.0)
