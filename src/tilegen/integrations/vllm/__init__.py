# src/tilegen/integrations/vllm/__init__.py
#
# vLLM out-of-tree (OOT) plugin (Phase 3)

"""Provides the ``vllm.general_plugins`` entry point (:mod:`platform_plugin`),
the ``TileGenAttentionBackend``, and OOT custom-op overrides.
"""

from tilegen.integrations.vllm.attention_backend import (
    TileGenAttentionBackend,
    TileGenAttentionImpl,
)
from tilegen.integrations.vllm.custom_ops import TileGenRMSNorm

__all__ = [
    "TileGenAttentionBackend",
    "TileGenAttentionImpl",
    "TileGenRMSNorm",
]
