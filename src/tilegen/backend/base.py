# src/tilegen/backend/base.py
#
# Pluggable backend lowering framework

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar
from tilegen.ir.ops import VTMKernel
from tilegen.target.spec import TargetSpec


class Backend(ABC):
    """Common interface for all VTM lowering backends."""

    name: ClassVar[str]

    @abstractmethod
    def lower(self, kernel: VTMKernel, target: TargetSpec) -> str:
        """Return backend source code for ``kernel`` targeted at ``target``."""

    def dump_ir(self, kernel: VTMKernel, target: TargetSpec) -> str:
        """Return intermediate IR text for ``--dump-ir`` diagnostics.

        Defaults to the lowered source; backends with a distinct intermediate
        representation (e.g. Triton-TLE's intermediate IR) override this.
        """
        return self.lower(kernel, target)


_BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    _BACKENDS[backend.name] = backend


def get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def _import_emitters() -> None:
    """Lazily import the built-in emitters so they self-register."""
    from tilegen.backend.tilelang.emitter import TileLangBackend # noqa: F401
    from tilegen.backend.triton.emitter import TritonBackend # noqa: F401


def ensure_backends_registered() -> None:
    if not _BACKENDS:
        _import_emitters()


__all__ = [
    "Backend",
    "register_backend",
    "get_backend",
    "available_backends",
    "ensure_backends_registered",
]
