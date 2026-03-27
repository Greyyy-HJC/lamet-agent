"""Registry for all workflow stage implementations."""

from __future__ import annotations

from typing import Type

from lamet_agent.stages.base import WorkflowStage

STAGE_REGISTRY: dict[str, Type[WorkflowStage]] = {}


def register_stage(stage_class: Type[WorkflowStage]) -> Type[WorkflowStage]:
    """Register a workflow stage class by its canonical name."""
    STAGE_REGISTRY[stage_class.name] = stage_class
    return stage_class


def get_stage(stage_name: str) -> WorkflowStage:
    """Instantiate a registered stage by name."""
    stage_class = STAGE_REGISTRY[stage_name]
    return stage_class()


def list_stage_names() -> list[str]:
    """Return all registered stage names in insertion order."""
    return list(STAGE_REGISTRY.keys())
