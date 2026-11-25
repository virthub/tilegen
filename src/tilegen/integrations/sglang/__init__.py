# src/tilegen/integrations/sglang/__init__.py
#
# SGLang integration

"""Provides ``TileGenAttentionBackend`` (an SGLang ``AttentionBackend``) routing
prefill/decode to TileGen's FA kernels.  SGLang is imported lazily under
``TYPE_CHECKING`` so ``import tilegen`` stays lightweight.
"""

from tilegen.integrations.sglang.attention_backend import TileGenAttentionBackend

__all__ = ["TileGenAttentionBackend"]
