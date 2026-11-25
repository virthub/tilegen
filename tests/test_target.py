# tests/test_target.py
#
# Unit tests for TargetSpec profiles and target loading

from tilegen.target.spec import BUILTIN_TARGETS, TargetSpec, get_target


class TestBuiltins:
    def test_registry_size(self):
        assert len(BUILTIN_TARGETS) >= 8

    def test_quant_paths(self):
        hopper = get_target("sm_90a")
        assert hopper.quant_path("int8") == "native"
        assert hopper.quant_path("fp8") == "native"
        assert hopper.quant_path("fp4") == "emulated"

        blackwell = get_target("sm_100a")
        assert blackwell.quant_path("fp4") == "native"
        assert blackwell.supports("tcgen05") and blackwell.supports("2cta")

        cdna = get_target("gfx942")
        assert cdna.device == "rocm"
        assert cdna.quant_path("fp8") == "native"

        ascend = get_target("ascend_950")
        assert ascend.device == "ascend"
        assert ascend.quant_path("fp4") == "native"

    def test_unknown_target(self):
        import pytest

        with pytest.raises(KeyError):
            get_target("sm_999")


class TestRoundtrip:
    def test_to_from_dict(self):
        h = get_target("sm_90a")
        h2 = TargetSpec.from_dict(h.to_dict())
        assert h2 == h

    def test_features_is_frozenset(self):
        assert isinstance(get_target("sm_90a").features, frozenset)
