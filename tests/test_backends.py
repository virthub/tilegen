# tests/test_backends.py
#
# Unit tests for the TileLang and Triton lowering backends

import ast
import pytest
from tilegen.backend import (
    available_backends,
    ensure_backends_registered,
    get_backend,
)
from tilegen.layout.quant import QuantSpec
from tilegen.ops.attention import AttentionSpec, build_attention_vtm
from tilegen.target.spec import get_target


@pytest.fixture(scope="module")
def kernels():
    ensure_backends_registered()
    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    return {
        "blackwell": build_attention_vtm(spec, get_target("sm_100a")),
        "hopper": build_attention_vtm(spec, get_target("sm_90a")),
    }


class TestRegistry:
    def test_backends_registered(self):
        ensure_backends_registered()
        assert set(available_backends()) == {"tilelang", "triton"}

    def test_unknown_backend(self):
        with pytest.raises(KeyError):
            get_backend("nope")


class TestTileLang:
    def test_lower_is_valid_python(self, kernels):
        src = get_backend("tilelang").lower(kernels["blackwell"], get_target("sm_100a"))
        ast.parse(src)
        assert "T.gemm" in src
        assert "T.alloc_shared" in src
        assert "T.Pipelined" in src

    def test_tmem_on_blackwell(self, kernels):
        src = get_backend("tilelang").lower(kernels["blackwell"], get_target("sm_100a"))
        assert "T.alloc_tmem" in src
        src_h = get_backend("tilelang").lower(kernels["hopper"], get_target("sm_90a"))
        assert "T.alloc_tmem" not in src_h


class TestTriton:
    def test_lower_is_valid_python(self, kernels):
        src = get_backend("triton").lower(kernels["blackwell"], get_target("sm_100a"))
        ast.parse(src)
        assert "@triton.jit" in src
        assert "tl.dot" in src

    def test_tier_selection(self, kernels):
        # Persistent + ping_pong => TLE-Struct.
        src = get_backend("triton").lower(kernels["blackwell"], get_target("sm_100a"))
        assert "TLE-Struct" in src
        # FA1 (data-parallel, serial) => TLE-Lite.
        fa1 = build_attention_vtm(AttentionSpec(kv_quant=QuantSpec.int8_kv()), get_target("sm_80"), variant="fa1")
        src = get_backend("triton").lower(fa1, get_target("sm_80"))
        assert "TLE-Lite" in src

    def test_native_fp8_uses_dot_scaled(self):
        spec = AttentionSpec(kv_quant=QuantSpec.fp8_e4m3())
        fa2 = build_attention_vtm(spec, get_target("sm_90a"), variant="fa2_prefill")
        src = get_backend("triton").lower(fa2, get_target("sm_90a"))
        assert "tl.dot_scaled" in src

    def test_matmul_transpose_operand(self, kernels):
        src = get_backend("triton").lower(kernels["hopper"], get_target("sm_90a"))
        assert "tl.dot(q_smem, tl.trans(k_reg))" in src


def test_tilelang_codegen_target_flag():
    """The TileLang backend derives its toolchain target from TargetSpec.device."""
    from tilegen.ops.attention import build_dense_attention_vtm

    ensure_backends_registered()
    spec = AttentionSpec(batch=1, num_heads=2, num_kv_heads=2, head_dim=64,
                         seq_len=128, dtype="bf16")
    cases = {
        "sm_120": "target='cuda'",
        "sm_90a": "target='cuda'",
        "gfx942": "target='hip'",
        "ascend_950": "target='ascend'",
    }
    for target_name, expected in cases.items():
        kernel = build_dense_attention_vtm(spec, get_target(target_name))
        src = get_backend("tilelang").lower(kernel, get_target(target_name))
        assert expected in src, f"{target_name}: {expected} not in generated source"


def test_tilelang_dense_codegen_is_runnable_shape():
    """Dense attention lowers to the runnable FA4 TileLang factory (not a sketch)."""
    from tilegen.ops.attention import build_dense_attention_vtm

    ensure_backends_registered()
    spec = AttentionSpec(batch=2, num_heads=4, num_kv_heads=4, head_dim=64,
                         seq_len=256, dtype="bf16", causal=True)
    kernel = build_dense_attention_vtm(spec, get_target("sm_120"))
    src = get_backend("tilelang").lower(kernel, get_target("sm_120"))
    assert "def main(" in src and "T.gemm(" in src
    assert "T.Pipelined(" in src and "T.exp2(" in src
    assert "return main" in src and "_forward(" in src


def test_tilelang_variant_detection():
    """_variant_of maps the $M_0$ structural levers to fa1/fa2/fa3/fa4."""
    from tilegen.backend.tilelang.emitter import _variant_of
    from tilegen.ir.ops import VTMKernel
    from tilegen.ops.schedules import attention_topology

    def k(variant):
        return VTMKernel(name=variant, topology=attention_topology(variant, get_target("sm_120")))

    assert _variant_of(k("fa1")) == "fa1"
    assert _variant_of(k("fa2_prefill")) == "fa2"
    assert _variant_of(k("fa3")) == "fa3"
    assert _variant_of(k("fa4")) == "fa4"


def test_tilelang_emits_variant_label():
    """The generated source is labeled with the inferred FA variant."""
    from tilegen.ops.attention import build_dense_attention_vtm

    ensure_backends_registered()
    spec = AttentionSpec(batch=2, num_heads=4, num_kv_heads=4, head_dim=64,
                         seq_len=256, dtype="bf16", causal=True)
    k2 = build_dense_attention_vtm(spec, get_target("sm_120"), variant="fa2_prefill")
    k3 = build_dense_attention_vtm(spec, get_target("sm_120"), variant="fa3")
    assert "runnable FA2" in get_backend("tilelang").lower(k2, get_target("sm_120"))
    assert "runnable FA3" in get_backend("tilelang").lower(k3, get_target("sm_120"))


def test_triton_variant_detection_and_label():
    """Triton infers the variant and labels the source (FA4 -> FA3 fallback)."""
    from tilegen.backend.triton.emitter import _variant_of
    from tilegen.ir.ops import VTMKernel
    from tilegen.ops.attention import build_dense_attention_vtm
    from tilegen.ops.schedules import attention_topology

    def k(variant):
        return VTMKernel(name=variant, topology=attention_topology(variant, get_target("sm_120")))

    assert _variant_of(k("fa1")) == "fa1"
    assert _variant_of(k("fa2_prefill")) == "fa2"
    assert _variant_of(k("fa3")) == "fa3"
    assert _variant_of(k("fa4")) == "fa4"

    ensure_backends_registered()
    spec = AttentionSpec(batch=2, num_heads=4, num_kv_heads=4, head_dim=64,
                         seq_len=256, dtype="bf16", causal=True)
    k2 = build_dense_attention_vtm(spec, get_target("sm_120"), variant="fa2_prefill")
    k3 = build_dense_attention_vtm(spec, get_target("sm_120"), variant="fa3")
    assert "runnable FA2" in get_backend("triton").lower(k2, get_target("sm_120"))
    assert "runnable FA3 warp-spec" in get_backend("triton").lower(k3, get_target("sm_120"))


def test_triton_fa3_warp_spec_gated_on_blackwell():
    """FA3 uses warp specialization only on Blackwell; falls back on Hopper/Ampere."""
    from tilegen.ops.attention import build_dense_attention_vtm

    ensure_backends_registered()
    spec = AttentionSpec(batch=1, num_heads=2, num_kv_heads=2, head_dim=64,
                         seq_len=128, dtype="bf16", causal=True)
    # Blackwell sm_120: warp_specialize emitted.
    k = build_dense_attention_vtm(spec, get_target("sm_120"), variant="fa3")
    src = get_backend("triton").lower(k, get_target("sm_120"))
    assert "warp_specialize=True" in src
    # Hopper sm_90a: FA3 degrades to the plain FA2 loop (no warp_specialize).
    k = build_dense_attention_vtm(spec, get_target("sm_90a"), variant="fa3")
    src = get_backend("triton").lower(k, get_target("sm_90a"))
    assert "warp_specialize=True" not in src
    assert "for start_n in range(0, hi, BLOCK_N)" in src


@pytest.mark.parametrize("target_name", [
    "sm_80", "sm_89", "sm_90a", "sm_100a", "sm_120",
    "gfx942", "ascend_910b", "ascend_910c", "ascend_950",
])
def test_codegen_valid_python_across_target_matrix(target_name):
    """Dense FA2 lowers to syntactically-valid Python for every declared target."""
    import ast

    from tilegen.ops.attention import build_dense_attention_vtm

    ensure_backends_registered()
    t = get_target(target_name)
    spec = AttentionSpec(batch=1, num_heads=2, num_kv_heads=2, head_dim=64,
                         seq_len=128, dtype="bf16", causal=True)
    kernel = build_dense_attention_vtm(spec, t, variant="fa2_prefill")
    for backend in ("triton", "tilelang"):
        src = get_backend(backend).lower(kernel, t)
        ast.parse(src)  # must be valid Python
        assert "def main(" in src or "def dense_fa2(" in src


def test_tilelang_target_flag_full_matrix():
    """TileLang target= maps device -> toolchain for the whole matrix."""
    from tilegen.ops.attention import build_dense_attention_vtm

    ensure_backends_registered()
    spec = AttentionSpec(batch=1, num_heads=2, num_kv_heads=2, head_dim=64,
                         seq_len=128, dtype="bf16")
    for target_name in ("sm_80", "sm_90a", "sm_120"):
        k = build_dense_attention_vtm(spec, get_target(target_name))
        assert "target='cuda'" in get_backend("tilelang").lower(k, get_target(target_name))
    for target_name in ("gfx942",):
        k = build_dense_attention_vtm(spec, get_target(target_name))
        assert "target='hip'" in get_backend("tilelang").lower(k, get_target(target_name))
    for target_name in ("ascend_910b", "ascend_950"):
        k = build_dense_attention_vtm(spec, get_target(target_name))
        assert "target='ascend'" in get_backend("tilelang").lower(k, get_target(target_name))
