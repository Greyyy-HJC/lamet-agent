"""Stage tool-registry resolution for the agent loop."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from .stages import resolve_stage_package


def resolve_plot_save_path(
    raw: str | None,
    *,
    artifacts_dir: Path,
    default_stem: str = "fit_on_data",
) -> str:
    """Map any plot save_path to a stem under ``artifacts_dir``."""
    if raw:
        stem = Path(raw).name
        for suffix in (".png", ".pdf", ".svg"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if not stem:
            stem = default_stem
    else:
        stem = default_stem
    return str(artifacts_dir / stem)


def resolve_stage_tools(stage: str) -> dict[str, Callable[..., dict[str, Any]]]:
    """Return the ``STAGE_TOOLS`` registry for a stage, or an empty dict."""
    package_name = resolve_stage_package(stage)
    if not package_name:
        return {}
    module = import_module(f"lamet_agent.stages.{package_name}.functions")
    return getattr(module, "STAGE_TOOLS", {})


def validate_stage_inputs(stage: str, manifest: Any) -> list[str]:
    """Return a stage's input issues via its ``validate_stage_inputs`` helper."""
    package_name = resolve_stage_package(stage)
    if not package_name:
        return []
    module = import_module(f"lamet_agent.stages.{package_name}.skills")
    validator = getattr(module, "validate_stage_inputs", None)
    return validator(manifest) if callable(validator) else []
