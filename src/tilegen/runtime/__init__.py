# src/tilegen/runtime/__init__.py
#
# Python AOT runtime support: artifact cache, semantic hash & dispatcher
# glue

from tilegen.runtime.cache import AOTArtifactCache, ArtifactRecord, DEFAULT_CACHE_DIR
from tilegen.runtime.hash import canonical_json, canonicalize, semantic_hash

__all__ = [
    "AOTArtifactCache",
    "ArtifactRecord",
    "DEFAULT_CACHE_DIR",
    "canonical_json",
    "canonicalize",
    "semantic_hash",
]
