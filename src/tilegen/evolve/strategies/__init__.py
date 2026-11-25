# src/tilegen/evolve/strategies/__init__.py
#
# Synthesis strategy plugins (Phase 4)

from tilegen.evolve.strategies.evolutionary import EvolutionaryEngine
from tilegen.evolve.strategies.forgekernel import ForgeKernelEngine
from tilegen.evolve.strategies.goal_loop import GoalLoopEngine
from tilegen.evolve.strategies.rl_rollout import RLRolloutEngine
from tilegen.evolve.strategies.schedule_codesign import ScheduleCodesignEngine
from tilegen.evolve.strategies.world_model import WorldModelEngine

# Strategy plugin registry (``./run.sh --evolve --strategy <name>``).
ENGINES = {
    "forgekernel": ForgeKernelEngine,
    "evolutionary": EvolutionaryEngine,
    "goal_loop": GoalLoopEngine,
    "world_model": WorldModelEngine,
    "rl_rollout": RLRolloutEngine,
    "schedule_codesign": ScheduleCodesignEngine,
}


def get_engine(name: str):
    if name not in ENGINES:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(ENGINES)}")
    return ENGINES[name]()


def register_engine(name: str, engine_cls) -> None:
    """Register a strategy plugin class (or zero-arg factory returning a SynthesisEngine)."""
    if not isinstance(name, str) or not name:
        raise ValueError("engine name must be a non-empty string")
    if engine_cls is None:
        raise ValueError("engine class/factory must not be None")
    ENGINES[name] = engine_cls


__all__ = [
    "ENGINES",
    "get_engine",
    "register_engine",
    "ForgeKernelEngine",
    "EvolutionaryEngine",
    "GoalLoopEngine",
    "WorldModelEngine",
    "RLRolloutEngine",
    "ScheduleCodesignEngine",
]
