# tests/test_hash.py
#
# Unit tests for canonicalization and the semantic hash

from tilegen.runtime.hash import canonical_json, canonicalize, semantic_hash


class TestCanonicalize:
    def test_dict_order_independent(self):
        a = {"x": 1, "nested": {"b": 2, "a": [1, 2, 3]}}
        b = {"nested": {"a": [1, 2, 3], "b": 2}, "x": 1}
        assert canonical_json(a) == canonical_json(b)

    def test_enum(self):
        from enum import Enum

        class E(Enum):
            A = "alpha"

        assert canonicalize(E.A) == "alpha"

    def test_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class P:
            x: int
            y: str

        assert canonicalize(P(1, "z")) == {"x": 1, "y": "z"}


class TestSemanticHash:
    def test_stable(self):
        assert semantic_hash({"a": 1}) == semantic_hash({"a": 1})
        assert semantic_hash({"a": 1}) != semantic_hash({"a": 2})

    def test_sha256_known_vector(self):
# canonical_json of "" is '""'; sha256('""')[:40] must equal the digest
        # of the same bytes. Verify against Python's own hashlib.
        import hashlib

        payload = canonical_json("")
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]
        assert semantic_hash("") == expected
