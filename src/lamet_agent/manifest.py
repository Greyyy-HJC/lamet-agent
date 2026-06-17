"""Minimal manifest schema and validation helpers.

Purpose:
- define the only required runtime input contract
- validate manifest JSON and kernel callable references

Expected inputs:
- a manifest JSON file with `correlators` and `kernels`
- correlator `path` values resolved relative to the manifest file, with a
  fallback to the lamet-agent project root for repo-style paths such as
  ``examples/fake_data/...``
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

from pydantic import BaseModel, Field, ConfigDict


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    goal: str = "full_lamet_pipeline"
    root_directory: Path | None = None
    correlators: list[CorrelatorInput] = Field(default_factory=list)
    kernels: list[KernelInput] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    manifest_dir: Path | None = Field(default=None, exclude=True)
    project_root: Path | None = Field(default=None, exclude=True)


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


def find_project_root(start: Path) -> Path:
    """Walk upward from ``start`` to locate this package's project root."""
    for candidate in (start, *start.parents):
        pyproject = candidate / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            continue
        if 'name = "lamet-agent"' in text:
            return candidate
    return start


def resolve_root_path(root_directory: Path, path: str) -> str:
    """Resolve an absolute or root-directory-relative path."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((root_directory / candidate).resolve())


def resolve_data_path(project_root: Path, manifest_dir: Path, path: str) -> str:
    """Resolve a correlator path against manifest or project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)

    manifest_relative = (manifest_dir / candidate).resolve()
    if manifest_relative.exists():
        return str(manifest_relative)

    return str((project_root / candidate).resolve())


def validate_manifest_file(path: Path) -> AnalysisManifest:
    """Parse manifest and validate kernel function references."""
    manifest_path = path.resolve()
    if not manifest_path.exists():
        raise ValueError(f"Manifest does not exist: {path}")

    manifest = AnalysisManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    for kernel in manifest.kernels:
        resolve_callable(kernel.function)

    manifest_dir = manifest_path.parent
    declared_root = manifest.root_directory
    if declared_root is not None:
        root_directory = Path(declared_root).expanduser().resolve()
        if not root_directory.is_absolute():
            raise ValueError("root_directory must be an absolute path or '~'-expanded absolute path")
        manifest.root_directory = root_directory
        project_root = root_directory
    else:
        project_root = find_project_root(manifest_dir)
    manifest.manifest_dir = manifest_dir
    manifest.project_root = project_root
    for correlator in manifest.correlators:
        if manifest.root_directory is not None:
            correlator.path = resolve_root_path(manifest.root_directory, correlator.path)
        else:
            correlator.path = resolve_data_path(project_root, manifest_dir, correlator.path)
    return manifest
