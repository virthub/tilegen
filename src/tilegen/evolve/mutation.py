# src/tilegen/evolve/mutation.py
#
# VTM schedule mutation operators shared by synthesis strategies

from __future__ import annotations

import random
from dataclasses import replace
from tilegen.ir.ops import VTMKernel
from tilegen.ir.topology import GridMode, PhaseMode, TacticalKnobs, TopologySpec

TILE_CHOICES = (32, 64, 128, 256)
SWIZZLE_CHOICES = ("NONE", "B32", "B64", "B128")
REGISTER_CHOICES = (128, 168, 200, 232)

# Kernel-structure variants in ascending generation order (SDPA -> FA1 -> FA4).
VARIANT_ORDER = ("sdpa", "fa1", "fa2_bf16", "fa3_tma", "fa3_ws", "fa4_tilelang")

# All tunable tactical-knob fields (in mutate order).
KNOB_FIELDS = (
    "tile_m", "tile_n", "tile_k", "pipeline_stages", "swizzle_mode",
    "async_wait_depth", "max_registers_per_thread",
    "exp_emulation_ratio", "rescale_threshold",
)


def _mutate_field(d: dict, field: str, rng: random.Random) -> None:
    """Apply one field mutation in-place to a knobs dict."""
    if field in ("tile_m", "tile_n", "tile_k"):
        cur = d[field]
        idx = TILE_CHOICES.index(cur) if cur in TILE_CHOICES else 1
        idx = max(0, min(len(TILE_CHOICES) - 1, idx + rng.choice([-1, 1])))
        d[field] = TILE_CHOICES[idx]
    elif field == "pipeline_stages":
        d["pipeline_stages"] = max(1, min(6, d["pipeline_stages"] + rng.choice([-1, 1])))
    elif field == "swizzle_mode":
        d["swizzle_mode"] = rng.choice(SWIZZLE_CHOICES)
    elif field == "async_wait_depth":
        d["async_wait_depth"] = max(0, min(4, d["async_wait_depth"] + rng.choice([-1, 1])))
    elif field == "max_registers_per_thread":
        cur = d["max_registers_per_thread"]
        idx = REGISTER_CHOICES.index(cur) if cur in REGISTER_CHOICES else 1
        idx = max(0, min(len(REGISTER_CHOICES) - 1, idx + rng.choice([-1, 1])))
        d["max_registers_per_thread"] = REGISTER_CHOICES[idx]
    elif field == "exp_emulation_ratio":
        d["exp_emulation_ratio"] = round(max(0.0, min(1.0, d["exp_emulation_ratio"] + rng.choice([-0.25, 0.25]))), 2)
    elif field == "rescale_threshold":
        d["rescale_threshold"] = round(max(0.0, min(1.0, d["rescale_threshold"] + rng.choice([-0.1, 0.1]))), 2)


def mutate_knobs(knobs: TacticalKnobs, rng: random.Random, bias: list[str] | None = None) -> TacticalKnobs:
    """Apply a single $M_1..M_k$ tactical mutation.

    ``bias`` is an ordered list of preferred knob fields (from the profiler's
    ``diagnose_fields``); when given, a biased field is chosen ~70% of the time
    so the mutation follows the hardware diagnosis instead of drifting randomly.
    """
    d = knobs.to_dict()
    if bias:
        biased = [f for f in bias if f in KNOB_FIELDS]
        if biased and rng.random() < 0.7:
            field = rng.choice(biased)
        else:
            field = rng.choice(KNOB_FIELDS)
    else:
        field = rng.choice(KNOB_FIELDS)
    _mutate_field(d, field, rng)
    return TacticalKnobs(**d)


def crossover_knobs(a: TacticalKnobs, b: TacticalKnobs, rng: random.Random) -> TacticalKnobs:
    """Uniform crossover between two ``TacticalKnobs`` configurations."""
    da, db = a.to_dict(), b.to_dict()
    return TacticalKnobs(**{k: rng.choice([da[k], db[k]]) for k in da})


def mutate_variant(variant: str, rng: random.Random) -> str:
    """Step the kernel-structure variant up/down one generation (FA2 <-> FA4).

    ~30% of the time jumps to a random variant instead of an adjacent step, so
    the search can escape the FA3 local minimum (FA3 is slower than FA2) and
    reach FA4 directly.
    """
    if variant not in VARIANT_ORDER:
        return "fa2_bf16"
    if rng.random() < 0.3:
        return rng.choice(VARIANT_ORDER)
    idx = VARIANT_ORDER.index(variant)
    idx = max(0, min(len(VARIANT_ORDER) - 1, idx + rng.choice([-1, 1])))
    return VARIANT_ORDER[idx]


def mutate_kernel(kernel: VTMKernel, rng: random.Random, bias: list[str] | None = None) -> VTMKernel:
    """Return a copy of ``kernel`` with mutated tactical knobs."""
    knobs = mutate_knobs(kernel.topology.knobs, rng, bias=bias)
    return replace(kernel, topology=replace(kernel.topology, knobs=knobs))


def reopen_m0(kernel: VTMKernel, rng: random.Random) -> VTMKernel:
    """Reopen the $M_0$ structural topology (plateau lever).

    Toggles the grid mode between ``PERSISTENT_SM`` and ``DATA_PARALLEL`` (and
    occasionally ``SPLIT_K``), flips the Blackwell 2-CTA group, and toggles the
    producer/consumer phase mode — a structural (not tactical) schedule change.
    """
    grid = kernel.topology.grid
    roles = kernel.topology.roles
    modes = [GridMode.DATA_PARALLEL, GridMode.PERSISTENT_SM]
    if grid.mode not in modes:
        modes.append(grid.mode)
    new_mode = rng.choice([m for m in modes if m != grid.mode])
    new_cta = 2 if (grid.cta_group == 1 and rng.random() < 0.5) else grid.cta_group
    new_phase = (
        PhaseMode.PING_PONG if (roles.phase_mode == PhaseMode.SERIAL and rng.random() < 0.5)
        else roles.phase_mode
    )
    new_grid = replace(grid, mode=new_mode, cta_group=new_cta)
    new_roles = replace(roles, phase_mode=new_phase)
    new_topology = replace(kernel.topology, grid=new_grid, roles=new_roles)
    return replace(kernel, topology=new_topology)


def kernel_state(kernel: VTMKernel) -> tuple:
    """Hashable state fingerprint of a kernel's M0 topology + tactical knobs."""
    g = kernel.topology.grid
    k = kernel.topology.knobs
    return (
        g.mode.value, g.cta_group, kernel.topology.roles.phase_mode.value,
        k.tile_m, k.tile_n, k.tile_k, k.pipeline_stages, k.swizzle_mode,
        k.async_wait_depth, k.max_registers_per_thread,
        k.exp_emulation_ratio, k.rescale_threshold,
    )


__all__ = [
    "TILE_CHOICES",
    "SWIZZLE_CHOICES",
    "REGISTER_CHOICES",
    "VARIANT_ORDER",
    "KNOB_FIELDS",
    "mutate_knobs",
    "crossover_knobs",
    "mutate_variant",
    "mutate_kernel",
    "reopen_m0",
    "kernel_state",
]
