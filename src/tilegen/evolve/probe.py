# src/tilegen/evolve/probe.py
#
# Probe channel: registry-validated diagnostic treatments applied to a
# declared target kernel. Probes are *measurements*, not candidates — the
# probe schema physically contains no promotion fields, and probe results
# can only certify mechanisms into the knowledge base (never become
# references). Treatments are a closed registry of parameterized toggles /
# IR-level passes with typed parameters.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional
from tilegen.evolve.interface import EvalResult, KernelCandidate
from tilegen.ir.ops import VTMKernel
from tilegen.target.spec import TargetSpec

def _set_pipeline_stages(kernel: VTMKernel, params: dict) -> VTMKernel:
    from dataclasses import replace

    knobs = replace(kernel.topology.knobs,
                    pipeline_stages=int(params["stages"]))
    return replace(kernel, topology=replace(kernel.topology, knobs=knobs))


def _set_tile(kernel: VTMKernel, params: dict) -> VTMKernel:
    from dataclasses import replace

    knobs = kernel.topology.knobs
    knobs = replace(knobs, tile_m=int(params["tile_m"]))
    if "tile_n" in params:
        knobs = replace(knobs, tile_n=int(params["tile_n"]))
    return replace(kernel, topology=replace(kernel.topology, knobs=knobs))


def _disable_swizzle(kernel: VTMKernel, params: dict) -> VTMKernel:
    del params
    from dataclasses import replace

    knobs = replace(kernel.topology.knobs, swizzle_mode="NONE")
    return replace(kernel, topology=replace(kernel.topology, knobs=knobs))


def _reopen_m0(kernel: VTMKernel, params: dict) -> VTMKernel:
    del params
    from dataclasses import replace

    from tilegen.ir.topology import GridMode

    grid = kernel.topology.grid
    new_mode = (GridMode.PERSISTENT_SM if grid.mode == GridMode.DATA_PARALLEL
                else GridMode.DATA_PARALLEL)
    return replace(kernel, topology=replace(
        kernel.topology, grid=replace(grid, mode=new_mode)))


TREATMENTS: dict[str, Callable[[VTMKernel, dict], VTMKernel]] = {
    "set_pipeline_stages": _set_pipeline_stages,
    "set_tile": _set_tile,
    "disable_swizzle": _disable_swizzle,
    "reopen_m0": _reopen_m0,
}

_TYPED_PRIMITIVES = (int, float, bool, str)


@dataclass
class ProbeSpec:
    id: str
    hypothesis: str = ""
    mechanism_target: str = ""
    target_kernel_id: str = "candidate_id"
    treatment: str = "set_pipeline_stages"
    treatment_params: dict[str, Any] = field(default_factory=dict)
    verdict_threshold: float = 0.03
    certification: bool = False
    cache_state: str = "warm_l2"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        """Serialize to convention-conforming YAML (the schema-hygiene conventions)."""
        from tilegen.evolve.conventions import dumps

        return dumps(self.to_dict())

    @classmethod
    def from_yaml(cls, text: str) -> "ProbeSpec":
        """Parse convention-conforming YAML back into a ProbeSpec."""
        from tilegen.evolve.conventions import loads

        data = loads(text)
        valid = {"id", "hypothesis", "mechanism_target", "target_kernel_id",
                 "treatment", "treatment_params", "verdict_threshold",
                 "certification", "cache_state"}
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class ProbeResult:
    probe_id: str
    applied: bool
    verdict: str = "not_confirmed" # confirmed / not_confirmed
    control_us: Optional[float] = None
    treatment_us: Optional[float] = None
    delta: Optional[float] = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_spec(spec: ProbeSpec) -> list[str]:
    errors: list[str] = []
    if not spec.id:
        errors.append("probe id is required")
    if spec.treatment not in TREATMENTS:
        errors.append(f"treatment {spec.treatment!r} not in registry "
                      f"{sorted(TREATMENTS)}")
    if not spec.target_kernel_id:
        errors.append("target_kernel_id is required")
    for key, value in spec.treatment_params.items():
        if not isinstance(value, _TYPED_PRIMITIVES):
            errors.append(f"treatment_params[{key}] has untyped value "
                          f"{type(value).__name__}")
    if spec.verdict_threshold <= 0:
        errors.append("verdict_threshold must be positive")
    return errors


def run_probe(
    spec: ProbeSpec,
    kernel: VTMKernel,
    evaluator,
    target: Optional[TargetSpec] = None,
    sealed_ref_id: Optional[str] = None,
) -> ProbeResult:
    """Apply the treatment to ``kernel`` and time control vs treatment.

    Comparative timing only; the result can never promote a reference.
    """
    errors = validate_spec(spec)
    if errors:
        return ProbeResult(probe_id=spec.id, applied=False, errors=errors)
    try:
        treated = TREATMENTS[spec.treatment](kernel, spec.treatment_params)
    except Exception as exc: # noqa: BLE001
        return ProbeResult(probe_id=spec.id, applied=False,
                           errors=[f"treatment failed: {exc}"])
    control = KernelCandidate("probe_control", [], kernel,
                              rationale=f"probe {spec.id} control")
    treatment = KernelCandidate("probe_treatment", [], treated,
                                rationale=f"probe {spec.id} treatment")
    c_res = evaluator.evaluate([control], target, sealed_ref_id)
    t_res = evaluator.evaluate([treatment], target, sealed_ref_id)
    c = c_res[0] if c_res else None
    t = t_res[0] if t_res else None
    if (c is None or t is None or not c.correct or not t.correct
            or c.device_time_us is None or t.device_time_us is None):
        return ProbeResult(probe_id=spec.id, applied=False,
                           errors=["probe evaluations incomplete or incorrect"])
    delta = (c.device_time_us - t.device_time_us) / c.device_time_us
    verdict = "confirmed" if delta >= spec.verdict_threshold else "not_confirmed"
    return ProbeResult(
        probe_id=spec.id, applied=True, verdict=verdict,
        control_us=c.device_time_us, treatment_us=t.device_time_us, delta=delta,
    )


def certify(
    spec: ProbeSpec,
    result: ProbeResult,
    kb,
    target: TargetSpec,
    eval_result: EvalResult,
) -> bool:
    """Certify a confirmed probe's mechanism into the knowledge base.

    Only general mechanisms independently certified by hardware probes may
    enter the KB; the recipe carries the certifying probe id as its
    provenance line.
    """
    if not spec.certification or result.verdict != "confirmed":
        return False
    kb.record_certified({
        "target": target.name,
        "mechanism_target": spec.mechanism_target,
        "config": dict(spec.treatment_params),
        "Certification source": spec.id,
    }, eval_result)
    return True


__all__ = [
    "ProbeSpec",
    "ProbeResult",
    "TREATMENTS",
    "validate_spec",
    "run_probe",
    "certify",
]
