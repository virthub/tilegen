# tests/test_pspkv_decode_smoke.py
#
# Default-suite smoke coverage for the PSP-KV Triton decode module: the
# full parity tests run in the vllm-pspkv fork venv only, so this file
# guarantees the module stays importable and structurally intact in the
# default environment (coverage blind spot in §24.2.1 of tilegen.md).

import pytest


def test_module_importable():
    mod = pytest.importorskip("tilegen.ops.pspkv_decode")
    assert callable(mod.pspkv_decode_forward)


def test_launcher_signature_and_kernel_present():
    import inspect

    import tilegen.ops.pspkv_decode as mod

    params = list(inspect.signature(mod.pspkv_decode_forward).parameters)
    for required in ("q", "cache", "block_tables", "seq_lens",
                     "precision_table", "block_offsets", "scale_table",
                     "kv_layout", "fmt_k", "fmt_v"):
        assert required in params
    assert hasattr(mod, "_pspkv_decode_kernel")
    assert hasattr(mod, "_load_dequant")


def test_no_lut_tables_remain():
# Regression guard for the shared-memory fix: the arithmetic decode must
    # not reintroduce fp8/fp4 lookup tables.
    import tilegen.ops.pspkv_decode as mod

    for name in dir(mod):
        if "lut" in name.lower():
            assert "lut" not in name.lower() or name.startswith("__")


def test_header_comment_present():
    import tilegen.ops.pspkv_decode as mod

    src = mod.__file__
    assert src
    with open(src, "r", encoding="utf-8") as f:
        head = f.read(400)
    assert "# src/tilegen/ops/pspkv_decode.py" in head
