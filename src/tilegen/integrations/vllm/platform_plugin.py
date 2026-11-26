# src/tilegen/integrations/vllm/platform_plugin.py
#
# vLLM out-of-tree plugin entry point (vllm.general_plugins)

from __future__ import annotations


def register() -> None:
    """Activate TileGen in vLLM (idempotent; may be called multiple times)."""
# Importing these modules performs side-effect registration:
# * TileGenAttentionBackend becomes selectable via VLLM_ATTENTION_BACKEND.
# * TileGenRMSNorm overrides vllm's "rms_norm" via CustomOp.register_oot.
# * TileGenPSPKVAttentionBackend (TILEGEN_PSPKV) becomes selectable in
# the vllm-pspkv fork; registration is a no-op failure in upstream
# vLLM, where the fork-specific paths are unavailable at runtime.
    from tilegen.integrations.vllm.attention_backend import ( # noqa: F401
        TileGenAttentionBackend,
    )
    from tilegen.integrations.vllm.custom_ops import TileGenRMSNorm # noqa: F401
    try:
        from tilegen.integrations.vllm_pspkv.attention_backend import ( # noqa: F401
            TileGenPSPKVAttentionBackend,
        )
    except ImportError:
# Upstream vLLM or missing fork hooks: TILEGEN_PSPKV is simply not
        # registered; the rest of the plugin keeps working.
        pass


__all__ = ["register"]
