# src/tilegen/_selftest.py
#
# Built-in smoke checks exercised by tilegen --selftest

from __future__ import annotations

from typing import Callable, List, Tuple


def run_selftest() -> Tuple[int, List[str]]:
    failures: List[str] = []
    checks: List[Tuple[str, Callable[[], None]]] = [
        ("quant spec packing", _check_quant_packing),
        ("quant spec presets", _check_quant_presets),
        ("contiguous layout strides", _check_contiguous),
        ("paged kv offset", _check_paged_kv),
        ("vtm ir serialize", _check_vtm_serialize),
        ("target profiles", _check_targets),
        ("semantic hash stability", _check_semantic_hash),
        ("paged attention build", _check_paged_attention),
        ("backend lowering", _check_backend_lowering),
        ("fa schedule variants", _check_fa_variants),
        ("boundary guards", _check_guards),
    ]
    for name, fn in checks:
        try:
            fn()
        except Exception as exc: # noqa: BLE001 - report and continue
            failures.append(f"{name}: {exc!r}")
    return len(checks), failures


def _check_quant_packing() -> None:
    from tilegen.layout.quant import QuantSpec

    q = QuantSpec.int4_kv()
    assert q.bits_per_element == 4
    assert q.is_sub_byte
    assert q.packed_bytes(128) == 64
    assert q.num_scale_blocks(64) == 1
    assert q.num_scale_blocks(65) == 2


def _check_quant_presets() -> None:
    from tilegen.layout.quant import QuantSpec

    nv = QuantSpec.nvfp4()
    assert nv.dtype == "fp4_e2m1" and nv.block_size == 16 and nv.pack_factor == 2
    mx = QuantSpec.mxfp8()
    assert mx.block_size == 32 and mx.scale_dtype == "ue8m0"
    awq = QuantSpec.awq_gptq_int4()
    assert awq.has_zero_point and awq.interleave == "awq_k8"


def _check_contiguous() -> None:
    from tilegen.layout.layout import ContiguousLayout

    c = ContiguousLayout((2, 3, 4))
    assert c.strides == (12, 4, 1)
    assert c.offset((1, 2, 3)) == 1 * 12 + 2 * 4 + 3 * 1


def _check_paged_kv() -> None:
    from tilegen.layout.layout import PagedKVLayout

    p = PagedKVLayout(block_size=16, num_kv_heads=8, head_dim=64)
    assert p.num_blocks_for(33) == 3
    off = p.in_page_offset(2, 3, 1)
    assert off == 2 * (16 * 8 * 64) + 3 * (8 * 64) + 1 * 64


def _check_vtm_serialize() -> None:
    from tilegen.ir.builder import VTMBuilder
    from tilegen.ir.types import MemoryTier
    from tilegen.layout.quant import QuantSpec

    b = VTMBuilder("minimal")
    b.input("A", (128, 64), "bf16")
    b.output("C", (128, 128), "bf16")
    b.quant("q", QuantSpec.int8_kv())
    b.alloc("a_s", (128, 64), "bf16", MemoryTier.SCRATCHPAD)
    b.alloc("acc", (128, 128), "fp32", MemoryTier.ACCUMULATOR)
    b.load("A", "a_s")
    b.matmul("a_s", "a_s", "acc", transpose_b=True, clear=True)
    b.store("acc", "C")
    k = b.build()
    text = k.serialize()
    assert text.startswith("vtm.kernel @minimal(")
    assert "vtm.matmul" in text


def _check_targets() -> None:
    from tilegen.target.spec import get_target

    h = get_target("sm_90a")
    assert h.supports("wgmma") and h.quant_path("fp8") == "native"
    assert h.quant_path("fp4") == "emulated"
    b = get_target("sm_100a")
    assert b.supports("tcgen05") and b.quant_path("fp4") == "native"
    a = get_target("ascend_950")
    assert a.quant_path("fp4") == "native"


def _check_semantic_hash() -> None:
    from tilegen.runtime.hash import semantic_hash

    a = {"x": 1, "nested": {"b": 2}}
    b = {"nested": {"b": 2}, "x": 1}
    assert semantic_hash(a) == semantic_hash(b)
    assert semantic_hash(a) != semantic_hash({**a, "x": 2})


def _check_paged_attention() -> None:
    from tilegen.layout.quant import QuantSpec
    from tilegen.ops.attention import AttentionSpec, build_paged_attention_vtm
    from tilegen.target.spec import get_target

    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    k = build_paged_attention_vtm(spec, get_target("sm_100a"))
    assert k.name == "paged_flash_attn_fp4"
    assert k.topology.grid.cta_group == 2
    # tile_k (64) must be divisible by fp4 block_size (16) and pack_factor (2)
    assert k.topology.knobs.tile_k % 16 == 0
    assert k.topology.knobs.tile_k % 2 == 0


def _check_backend_lowering() -> None:
    import ast

    from tilegen.backend import ensure_backends_registered, get_backend
    from tilegen.layout.quant import QuantSpec
    from tilegen.ops.attention import AttentionSpec, build_attention_vtm
    from tilegen.target.spec import get_target

    ensure_backends_registered()
    k = build_attention_vtm(AttentionSpec(kv_quant=QuantSpec.nvfp4()), get_target("sm_100a"))
    for name in ("tilelang", "triton"):
        src = get_backend(name).lower(k, get_target("sm_100a"))
        ast.parse(src)


def _check_fa_variants() -> None:
    from tilegen.layout.quant import QuantSpec
    from tilegen.ops.attention import AttentionSpec, build_attention_vtm
    from tilegen.ops.schedules import VARIANTS
    from tilegen.target.spec import get_target

    spec = AttentionSpec(kv_quant=QuantSpec.nvfp4())
    tops = {
        v: build_attention_vtm(spec, get_target("sm_90a"), variant=v).topology.to_dict()
        for v in VARIANTS
    }
    assert len({str(t) for t in tops.values()}) == len(VARIANTS)


def _check_guards() -> None:
    from tilegen.layout.quant import QuantSpec
    from tilegen.ops.guards import BoundaryError, check_quant_alignment

    check_quant_alignment(64, 128, QuantSpec.nvfp4())
    try:
        check_quant_alignment(40, 128, QuantSpec.nvfp4())
    except BoundaryError:
        pass
    else:
        raise AssertionError("misaligned tile_k should raise BoundaryError")


__all__ = ["run_selftest"]
