# src/tilegen/integrations/lmcache/__init__.py
#
# LMCache integration

"""Provides ``LMCacheKVConnector`` — a KV-cache format bridge between TileGen's
paged/token-level KV and LMCache's ``store``/``retrieve``/``lookup``.  LMCache
is duck-typed (not imported), so ``import tilegen`` stays lightweight.
"""

from tilegen.integrations.lmcache.connector import (
    LMCacheKVConnector,
    kv_to_tensor,
    tensor_to_kv,
    token_hash_key,
)

__all__ = ["LMCacheKVConnector", "kv_to_tensor", "tensor_to_kv", "token_hash_key"]
