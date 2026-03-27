"""Base stage interfaces and execution context models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from lamet_agent.loaders import CorrelatorDataset
from lamet_agent.schemas import Manifest


@dataclass(slots=True)
class StageContext:
    """Shared context passed into every stage."""

    manifest: Manifest
    run_directory: Path
    datasets: dict[str, CorrelatorDataset]
    kernel: Any
    stage_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)

    def stage_directory(self, stage_name: str) -> Path:
        """Return the canonical directory for a stage."""
        return self.run_directory / "stages" / stage_name

    def parameters_for(self, stage_name: str) -> dict[str, Any]:
        """Return user-supplied stage parameters."""
        return dict(self.manifest.workflow.stage_parameters.get(stage_name, {}))


class WorkflowStage(Protocol):
    """Protocol implemented by all stage classes."""

    name: str
    description: str

    def run(self, context: StageContext):  # pragma: no cover - protocol signature only.
        """Execute the stage and return a normalized result."""
