"""Planner interfaces for mapping manifests to workflow plans."""

from __future__ import annotations

from typing import Protocol

from lamet_agent.schemas import Manifest
from lamet_agent.workflows import WorkflowPlan


class WorkflowPlanner(Protocol):
    """Protocol for planner implementations."""

    def resolve(self, manifest: Manifest) -> WorkflowPlan:  # pragma: no cover - protocol only.
        """Resolve a manifest into an executable workflow plan."""
