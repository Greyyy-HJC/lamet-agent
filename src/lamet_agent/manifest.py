"""Minimal manifest schema and validation helpers.

Purpose:
- define the only required runtime input contract
- validate manifest JSON and kernel callable references

Expected inputs:
- a manifest JSON file with `correlators` and `kernels`
- kernel function references in `module:function` format

Expected outputs:
- parsed `AnalysisManifest` object used by CLI commands

Example usage:
- from lamet_agent.manifest import validate_manifest_file
- manifest = validate_manifest_file(Path("examples/workflow_smoke_manifest.json"))
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class CorrelatorInput(BaseModel):
    """Single correlator dataset descriptor."""

    dataset_id: str
    kind: Literal["2pt", "3pt"]
    path: str
    format: Literal["txt", "npy", "hdf5", "csv"] = "txt"
    metadata: dict = Field(default_factory=dict)


class KernelInput(BaseModel):
    """Reference to a perturbative kernel callable."""

    kernel_id: str
    function: str
    description: str = ""


class AnalysisManifest(BaseModel):
    """Top-level analysis manifest."""

    run_id: str
    goal: str = "full_lamet_pipeline"
    correlators: list[CorrelatorInput] = Field(default_factory=list)
    kernels: list[KernelInput] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


def resolve_callable(reference: str) -> Callable:
    """Resolve `module:function` into a Python callable."""
    if ":" not in reference:
        raise ValueError(
            f"Invalid callable reference '{reference}'. Use 'module:function'."
        )

    module_name, fn_name = reference.split(":", maxsplit=1)
    module = import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Reference is not a callable: {reference}")
    return fn


def validate_manifest_file(path: Path) -> AnalysisManifest:
    """Parse manifest and validate kernel function references."""
    if not path.exists():
        raise ValueError(f"Manifest does not exist: {path}")

    manifest = AnalysisManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
    for kernel in manifest.kernels:
        resolve_callable(kernel.function)
    return manifest
