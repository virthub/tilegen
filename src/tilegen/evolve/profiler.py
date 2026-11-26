# src/tilegen/evolve/profiler.py
#
# Unified profiler adapter: normalized telemetry → VTM IR diagnostics

from __future__ import annotations

from typing import Mapping

_CTA_BOUNDARY = "cta_boundary_stall"
_MBARRIER_WAIT = "mbarrier_wait_stall"
_SFU_STALL = "sfu_stall"
_REGISTER_SPILL = "register_spill"
_TENSORCORE_IDLE = "tensorcore_idle"


def diagnose(telemetry: Mapping[str, float]) -> list[dict[str, str]]:
    """Map normalized stall metrics to ``(symptom, action)`` VTM IR diagnoses.

    Returns a list of ``{"symptom": ..., "action": ...}`` records ordered by
    severity (highest stall first).
    """
    out: list[dict[str, str]] = []
    ordered = [
        (_CTA_BOUNDARY, "Reopen GridTopology -> PERSISTENT_SM"),
        (_MBARRIER_WAIT, "Tune pipeline_stages / async_wait_depth / producer_warpgroups"),
        (_SFU_STALL, "Enable PhaseSchedule.PING_PONG or tune exp_emulation_ratio / rescale_threshold"),
        (_REGISTER_SPILL, "Shift RecomputeBoundary -> REGISTER_FUSED or tighten max_registers_per_thread"),
        (_TENSORCORE_IDLE, "Overlap Vector/Cube sub-tiling or increase pipeline_stages"),
    ]
    for key, action in ordered:
        value = float(telemetry.get(key, 0.0))
        if value > 0.0:
            out.append({"symptom": key, "action": action})
    return out


def normalize_ncu_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Map raw NVIDIA Nsight-Compute metric names to normalized symptom keys.

    Accepts the common ``ncu`` metric names; unknown keys are passed through.
    """
    mapping = {
        "sm__warps_active.avg.pct_of_peak_sustained_active": _TENSORCORE_IDLE,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": _MBARRIER_WAIT,
        "smsp__inst_executed_pipe_xu.sum": _SFU_STALL,
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": _REGISTER_SPILL,
    }
    out: dict[str, float] = {}
    for key, value in metrics.items():
        target = mapping.get(key)
        if target is None:
# Isolation audit: unknown metric names (which may carry kernel/
            # function/file symbols) are hashed before entering meta context.
            import hashlib

            target = "anon_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        out[target] = float(value)
    return out


# Symptom -> tactical-knob fields a mutation should bias toward.
SYMPTOM_FIELDS: dict[str, list[str]] = {
    _CTA_BOUNDARY: ["grid_mode"],
    _MBARRIER_WAIT: ["pipeline_stages", "async_wait_depth", "producer_warpgroups"],
    _SFU_STALL: ["exp_emulation_ratio", "rescale_threshold", "phase_mode"],
    _REGISTER_SPILL: ["max_registers_per_thread"],
    _TENSORCORE_IDLE: ["pipeline_stages", "tile_m", "tile_n"],
}


def diagnose_fields(telemetry: Mapping[str, float]) -> list[str]:
    """Map telemetry to the knob fields a mutation should prefer.

    Returns the ordered (most-severe-first) list of biased fields, which a
    strategy passes to ``mutate_knobs(..., bias=...)``.  This is the feedback
    edge that turns the evolver from a random walk into a profiler-guided one.
    """
    fields: list[str] = []
    for d in diagnose(telemetry):
        for f in SYMPTOM_FIELDS.get(d["symptom"], []):
            if f not in fields:
                fields.append(f)
    return fields


class ProfilerBackend:
    """Real-telemetry source: normalized symptom dict per config key."""

    def telemetry(self, config_key, variant, params, tensors, is_causal,
                  sm_scale) -> dict[str, float]:
        raise NotImplementedError


_NCU_METRICS = {
    "tensorcore_idle": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "mbarrier_wait_stall": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "sfu_stall": "smsp__inst_executed_pipe_xu.sum",
    "register_spill": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
}


class NcuProfiler(ProfilerBackend):
    """Nsight-Compute subprocess profiler with per-config caching.

    Each distinct ``config_key`` is profiled once (``launch_count`` kernel
    launches), parsed from ``ncu --csv``, normalized through
    :func:`normalize_ncu_metrics`, and cached under ``cache_dir``.
    """

    def __init__(self, ncu_bin=None, cache_dir=None, launch_count=3,
                 metrics=None):
        import shutil

        if ncu_bin is None:
            ncu_bin = shutil.which("ncu")
        if not ncu_bin:
            from pathlib import Path

            candidates = []
            for base in ("/usr/local/cuda/bin/ncu",
                         "/usr/local/cuda-12.8/bin/ncu",
                         "/usr/local/cuda-12/bin/ncu"):
                candidates.append(Path(base))
            candidates.extend(sorted(Path("/opt/nvidia").glob(
                "nsight-compute*/ncu")) if Path("/opt/nvidia").exists() else [])
            ncu_bin = next((str(c) for c in candidates if c.exists()), "ncu")
        self.ncu_bin = ncu_bin
        self.launch_count = launch_count
        self.metrics = metrics or dict(_NCU_METRICS)
        self.cache_dir = cache_dir

    def _cache_path(self, config_key: str):
        import hashlib
        from pathlib import Path

        import os

        root = Path(self.cache_dir) if self.cache_dir else Path(
            os.environ.get("TILEGEN_CACHE_DIR", ".tilegen_cache"))
        d = root / "profiles"
        d.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(config_key.encode("utf-8")).hexdigest()[:16]
        return d / f"profile_{digest}.json"

    def _write_payload(self, tensors, params, variant, is_causal, sm_scale):
        import json
        import tempfile

        import torch

        _, tpath = tempfile.mkstemp(suffix=".pt", prefix="ncu_tensors_")
        _, spath = tempfile.mkstemp(suffix=".py", prefix="ncu_script_")
        torch.save(tensors, tpath)
        script = (
            "import json, sys, torch\n"
            "from tilegen.evolve.evaluator import forward_fn_for_variant\n"
            "q, k, v = torch.load(sys.argv[1], map_location='cuda')\n"
            "params = json.loads(sys.argv[2])\n"
            "fn = forward_fn_for_variant(sys.argv[3])\n"
            "sm = float(sys.argv[4]) if sys.argv[4] != 'None' else None\n"
            "for _ in range(int(sys.argv[5])):\n"
            "    fn(q, k, v, is_causal=bool(int(sys.argv[6])), sm_scale=sm, **params)\n"
            "torch.cuda.synchronize()\n"
        )
        with open(spath, "w", encoding="utf-8") as f:
            f.write(script)
        return tpath, spath

    def telemetry(self, config_key, variant, params, tensors, is_causal,
                  sm_scale):
        import json
        import subprocess
        import sys

        cache = self._cache_path(config_key)
        if cache.exists():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        tpath, spath = self._write_payload(
            tensors, params, variant, is_causal, sm_scale)
        metric_list = ",".join(self.metrics.values())
        cmd = [
            self.ncu_bin, "--csv", "--metrics", metric_list,
            "--launch-count", str(self.launch_count),
            sys.executable, spath, tpath,
            json.dumps(params), variant,
            "None" if sm_scale is None else str(sm_scale),
            str(self.launch_count), "1" if is_causal else "0",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300)
        except (subprocess.TimeoutExpired, OSError):
            return {}
        raw = self._parse_csv(proc.stdout)
        if not raw:
            return {}
        normalized = normalize_ncu_metrics(raw)
        out = {k: float(normalized.get(k, 0.0)) for k in
               ("tensorcore_idle", "mbarrier_wait_stall", "sfu_stall",
                "register_spill", "cta_boundary_stall")}
        try:
            cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
        except OSError:
            pass
        return out

    @staticmethod
    def _parse_csv(stdout: str) -> dict[str, float]:
        """Average the CSV metric columns across profiled launches."""
        import csv
        import io

        reader = csv.reader(io.StringIO(stdout))
        rows = list(reader)
        if not rows:
            return {}
# ncu prefixes the CSV with ==PROF== status lines; locate the real
        # header row (the first one carrying the Metric Name column).
        header_idx = next(
            (i for i, row in enumerate(rows) if "Metric Name" in row), None)
        if header_idx is None:
            return {}
        header = rows[header_idx]
        name_idx = header.index("Metric Name")
        val_idx = header.index("Metric Value")
        sums: dict[str, list[float]] = {}
        for row in rows[header_idx + 1:]:
            if len(row) <= max(name_idx, val_idx):
                continue
            name = row[name_idx]
            try:
                value = float(row[val_idx])
            except ValueError:
                continue
            sums.setdefault(name, []).append(value)
        return {name: sum(v) / len(v) for name, v in sums.items()}


__all__ = ["diagnose", "normalize_ncu_metrics", "diagnose_fields",
           "SYMPTOM_FIELDS", "ProfilerBackend", "NcuProfiler"]
