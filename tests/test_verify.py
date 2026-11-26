# tests/test_verify.py
#
# Tests for the 4-stage verification pipeline (stages 3-4 are GPU-gated)

import pytest
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec
from tilegen.ops.verify import verify
from tilegen.target.spec import get_target


def test_stages_1_and_2_no_gpu():
    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    report = verify(spec, get_target("sm_90a"))
    by_name = {s.name: s for s in report.stages}
    # Stage 1 and 2 always run and pass without a GPU.
    assert by_name["host boundary guards"].passed
    assert by_name["zero-Python dispatch"].passed


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="requires CUDA"
)
def test_full_verify_on_gpu():
    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    report = verify(spec, get_target("sm_90a"), speed_gate=0.5)
    by_name = {s.name: s for s in report.stages}
    assert by_name["host boundary guards"].passed
    assert by_name["zero-Python dispatch"].passed
    assert by_name["CUDA-graph compatibility"].passed
    # Parity + dispatch must hold; with a lenient speed gate the stage passes.
    assert by_name["dispatch + regression"].passed


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="requires CUDA"
)
def test_verify_strict_speed_gate_reports():
    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    report = verify(spec, get_target("sm_90a"), speed_gate=1.0)
    summary = report.summary()
    assert "speedup=" in summary # speedup is always measured and reported
