from __future__ import annotations

import contextlib
from typing import Iterator

__version__ = "0.1.0"

from tilegen.layout import ( # noqa: F401
    ContiguousLayout,
    PagedKVLayout,
    QuantSpec,
    TensorSpec,
)
from tilegen.ir import ( # noqa: F401
    MemoryTier,
    SymbolicDim,
    TacticalKnobs,
    TopologySpec,
    VTMBuilder,
    VTMKernel,
)
from tilegen.target import TargetSpec, get_target, load_target # noqa: F401
from tilegen.runtime import AOTArtifactCache, semantic_hash # noqa: F401


@contextlib.contextmanager
def sdpa_kernel(*args, **kwargs) -> Iterator[None]:
    """Route ``F.scaled_dot_product_attention`` through ``tilegen::sdpa``.

    Dense CUDA bf16 attention dispatches to the TileGen FA2 kernel; other
    shapes/dtypes delegate to the native PyTorch SDPA backends.  The custom op
    is registered lazily on first use so ``import tilegen`` stays lightweight.
    """
    from tilegen.runtime.sdpa import sdpa_kernel as _impl

    with _impl(*args, **kwargs):
        yield


__all__ = [
    "__version__",
    "sdpa_kernel",
    "ContiguousLayout",
    "PagedKVLayout",
    "QuantSpec",
    "TensorSpec",
    "MemoryTier",
    "SymbolicDim",
    "TacticalKnobs",
    "TopologySpec",
    "VTMBuilder",
    "VTMKernel",
    "TargetSpec",
    "get_target",
    "load_target",
    "AOTArtifactCache",
    "semantic_hash",
]
