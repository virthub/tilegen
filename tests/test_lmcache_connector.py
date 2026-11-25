# tests/test_lmcache_connector.py
#
# Tests for the LMCache KV connector (tilegen.integrations.lmcache)

import pytest

torch = pytest.importorskip("torch")


class _MockEngine:
    """Duck-typed LMCache engine: store/retrieve/lookup with an in-memory dict."""

    def __init__(self):
        self._cache = {}

    def store(self, key, value):
        self._cache[key] = value

    def retrieve(self, key):
        return self._cache.get(key)

    def lookup(self, key):
        return key in self._cache


def test_kv_tensor_roundtrip():
    from tilegen.integrations.lmcache import kv_to_tensor, tensor_to_kv

    k = torch.randn(8, 2, 64)
    v = torch.randn(8, 2, 64)
    packed = kv_to_tensor(k, v)
    assert packed.shape[0] == 2
    k2, v2 = tensor_to_kv(packed)
    assert torch.equal(k, k2) and torch.equal(v, v2)


def test_connector_store_retrieve_lookup():
    from tilegen.integrations.lmcache import LMCacheKVConnector

    eng = _MockEngine()
    conn = LMCacheKVConnector(eng)
    k = torch.randn(4, 2, 32)
    v = torch.randn(4, 2, 32)
    key = "tilegen:deadbeef"
    assert conn.lookup(key) is False
    assert conn.retrieve(key) is None
    assert conn.store(key, k, v) is True
    assert conn.lookup(key) is True
    k2, v2 = conn.retrieve(key)
    assert torch.equal(k, k2) and torch.equal(v, v2)


def test_connector_no_engine_is_noop():
    from tilegen.integrations.lmcache import LMCacheKVConnector

    conn = LMCacheKVConnector(None)
    k = torch.randn(2, 2, 16)
    v = torch.randn(2, 2, 16)
    assert conn.store("x", k, v) is False
    assert conn.retrieve("x") is None
    assert conn.lookup("x") is False


def test_token_hash_key_is_deterministic():
    from tilegen.integrations.lmcache import token_hash_key

    a = torch.tensor([1, 2, 3])
    b = torch.tensor([1, 2, 3])
    c = torch.tensor([1, 2, 4])
    assert token_hash_key(a) == token_hash_key(b)
    assert token_hash_key(a) != token_hash_key(c)
    assert token_hash_key(a).startswith("tilegen:")
