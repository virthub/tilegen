# src/tilegen/integrations/lmcache/connector.py
#
# LMCache KV connector

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "LMCacheKVConnector",
    "kv_to_tensor",
    "tensor_to_kv",
    "token_hash_key",
]


def kv_to_tensor(k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    """Pack K/V into a single storable tensor.

    Layout: ``[2, *k_cache.shape]`` with ``[0] = K``, ``[1] = V`` (K and V must
    share a shape).  A single contiguous tensor is the natural unit for
    LMCache's ``MemoryObj``.
    """
    assert k_cache.shape == v_cache.shape, "K and V must share a shape"
    return torch.stack([k_cache, v_cache], dim=0).contiguous()


def tensor_to_kv(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a ``kv_to_tensor`` result back into (K, V)."""
    return t[0].contiguous(), t[1].contiguous()


def token_hash_key(token_ids: torch.Tensor) -> str:
    """Deterministic cache key from a prompt's token ids (prefix lookup)."""
    return f"tilegen:{token_ids.cpu().numpy().tobytes().hex()}"


class LMCacheKVConnector:
    """Bridge between TileGen's KV cache and an LMCache-like engine."""

    def __init__(self, engine=None):
        # ``engine`` is an LMCacheEngine (or any object with store/retrieve/lookup).
        self.engine = engine

    def store(self, key: str, k_cache: torch.Tensor, v_cache: torch.Tensor) -> bool:
        """Pack and store K/V.  Returns True if stored (engine present)."""
        if self.engine is None:
            return False
        self.engine.store(key, kv_to_tensor(k_cache, v_cache))
        return True

    def retrieve(self, key: str) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve and unpack K/V.  Returns None on miss or no engine."""
        if self.engine is None:
            return None
        hit = self.engine.retrieve(key)
        if hit is None:
            return None
        # LMCache MemoryObj carries the payload in ``.tensor``.
        t = hit.tensor if hasattr(hit, "tensor") else hit
        return tensor_to_kv(t)

    def lookup(self, key: str) -> bool:
        """True if the KV for ``key`` is cached."""
        if self.engine is None:
            return False
        return bool(self.engine.lookup(key))
