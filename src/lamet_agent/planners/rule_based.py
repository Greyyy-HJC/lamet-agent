"""Rule-based planner used for the initial scaffold implementation."""

from __future__ import annotations

import lamet_agent.stages  # noqa: F401

from lamet_agent.errors import WorkflowResolutionError
from lamet_agent.schemas import Manifest
from lamet_agent.stages.registry import list_stage_names
from lamet_agent.workflows import WorkflowPlan

FULL_PIPELINE = [
    "correlator_analysis",
    "renormalization",
    "fourier_transform",
    "perturbative_matching",
    "physical_limit",
]


class RuleBasedPlanner:
    """Resolve workflows from simple goal- and input-based rules."""

    def resolve(self, manifest: Manifest) -> WorkflowPlan:
        """Return the canonical workflow plan for a manifest."""
        available_stage_names = set(list_stage_names())
        if manifest.goal == "custom":
            stage_names = manifest.workflow.stages
            if not stage_names:
                raise WorkflowResolutionError("Custom goals must provide workflow.stages.")
        else:
            stage_names = list(FULL_PIPELINE)
        unknown = [stage_name for stage_name in stage_names if stage_name not in available_stage_names]
        if unknown:
            raise WorkflowResolutionError(f"Manifest requested unknown stage(s): {unknown}.")
        correlator_kinds = {correlator.kind for correlator in manifest.correlators}
        if "two_point" not in correlator_kinds:
            raise WorkflowResolutionError("At least one two-point correlator input is required for the default workflows.")
        final_observable = (
            "distribution_amplitude"
            if manifest.goal == "distribution_amplitude"
            else "parton_distribution_function"
        )
        return WorkflowPlan(goal=manifest.goal, stage_names=stage_names, final_observable=final_observable)
