"""Manifest schema and validation for job-based LaMET workflows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


StageId = Literal[
    "correlator_analysis",
    "renormalization",
    "fourier_transform",
    "perturbative_matching",
    "extrapolation",
]


class RunMetadata(BaseModel):
    """Settings shared by every job in one run.

    Required: run_id, root_directory, target_observable, parton, resample_mode,
    random_seed, stages.
    Optional: artifacts_directory (default "artifacts"), bin_size (default: no
    binning applied before jackknife/bootstrap resampling).
    Conditional: bs_samples is required when resample_mode == "bs" and has no
    default; it is ignored when resample_mode == "jk".
    """

    model_config = ConfigDict(extra="allow")

    run_id: str
    root_directory: str
    artifacts_directory: str = "artifacts"
    target_observable: Literal["pdf", "da", "gpd"]
    parton: Literal["quark", "gluon"]
    resample_mode: Literal["jk", "bs"]
    random_seed: int
    bs_samples: int | None = Field(default=None, gt=0)
    bin_size: int | None = Field(default=None, gt=0)
    stages: list[StageId]

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @model_validator(mode="after")
    def validate_bootstrap_requirements(self) -> "RunMetadata":
        if self.resample_mode == "bs" and self.bs_samples is None:
            raise ValueError("metadata.bs_samples is required when metadata.resample_mode is 'bs'")
        return self


class CorrelatorInput(BaseModel):
    """One raw 2pt or 3pt correlator dataset."""

    model_config = ConfigDict(extra="allow")

    correlator_id: str
    kind: Literal["2pt", "3pt"]
    data_path: str
    ensemble: str
    hadron: str
    gfix: str
    source_sink: str
    momentum: str
    a_fm: float
    pz_gev: float
    pz_out_gev: float | None = None
    src_gamma: str
    sink_gamma: str
    current_gamma: str | None = None
    z_direction: str | None = None
    eta: str | None = None
    bt: list[int] | None = None
    bz: list[int] | None = None
    tsep: int | None = None


class ArtifactInput(BaseModel):
    """Precomputed stage output that can seed a partial workflow."""

    model_config = ConfigDict(extra="allow")

    id: str
    stage: StageId
    path: str


class KernelInput(BaseModel):
    """Matching-kernel declaration."""

    model_config = ConfigDict(extra="allow")

    stage: str
    kernel_id: str
    kernel_path: str
    scheme: str
    kernel_parameters: dict[str, Any] = Field(default_factory=dict)


class ManifestInputs(BaseModel):
    """Global source-node pools."""

    correlators: list[CorrelatorInput] = Field(default_factory=list)
    artifacts: list[ArtifactInput] = Field(default_factory=list)
    kernels: list[KernelInput] = Field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class StageJob(BaseModel):
    """One independently executed stage job."""

    id: str
    correlator_ids: list[str] = Field(default_factory=list)
    inputs: dict[str, str | list[str]] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class StageConfig(BaseModel):
    """Shared defaults and jobs for one stage."""

    defaults: dict[str, Any] = Field(default_factory=dict)
    jobs: list[StageJob]

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class AnalysisManifest(BaseModel):
    """Top-level job-DAG manifest."""

    metadata: RunMetadata
    inputs: ManifestInputs = Field(default_factory=ManifestInputs)
    stages: dict[StageId, StageConfig]

    _manifest_path: Path | None = PrivateAttr(default=None)
    _root_directory: Path | None = PrivateAttr(default=None)
    _artifacts_directory: Path | None = PrivateAttr(default=None)

    @property
    def run_id(self) -> str:
        return self.metadata.run_id

    @property
    def root_directory(self) -> Path:
        if self._root_directory is None:
            return Path(self.metadata.root_directory).expanduser()
        return self._root_directory

    @property
    def artifacts_directory(self) -> Path:
        if self._artifacts_directory is None:
            return self.root_directory / self.metadata.artifacts_directory
        return self._artifacts_directory

    @property
    def manifest_dir(self) -> Path | None:
        return None if self._manifest_path is None else self._manifest_path.parent

    @property
    def project_root(self) -> Path:
        return self.root_directory

    @property
    def correlators(self) -> list[CorrelatorInput]:
        return self.inputs.correlators

    @property
    def kernels(self) -> list[KernelInput]:
        return self.inputs.kernels

    @model_validator(mode="after")
    def validate_dag(self) -> "AnalysisManifest":
        if len(set(self.metadata.stages)) != len(self.metadata.stages):
            raise ValueError("metadata.stages contains duplicate stage ids")
        missing = [stage for stage in self.metadata.stages if stage not in self.stages]
        if missing:
            raise ValueError(f"metadata.stages has no job configuration for: {missing}")

        correlator_ids = [item.correlator_id for item in self.inputs.correlators]
        source_ids = [item.id for item in self.inputs.artifacts]
        job_ids = [job.id for config in self.stages.values() for job in config.jobs]
        all_ids = correlator_ids + source_ids + job_ids
        duplicates = sorted({value for value in all_ids if all_ids.count(value) > 1})
        if duplicates:
            raise ValueError(f"manifest ids must be globally unique: {duplicates}")

        known = set(source_ids)
        correlator_id_set = set(correlator_ids)
        for stage in self.metadata.stages:
            for job in self.stages[stage].jobs:
                unknown_correlators = sorted(set(job.correlator_ids) - correlator_id_set)
                if unknown_correlators:
                    raise ValueError(f"job {job.id!r} references unknown correlators: {unknown_correlators}")
                for value in job.inputs.values():
                    refs = value if isinstance(value, list) else [value]
                    unknown = [ref for ref in refs if ref not in known]
                    if unknown:
                        raise ValueError(f"job {job.id!r} references unavailable upstream ids: {unknown}")
                known.add(job.id)
        return self


def _resolve_from_root(root: Path, value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (root / path).resolve())


def validate_manifest_file(path: Path) -> AnalysisManifest:
    """Parse a JSON/JSONC manifest, validate its DAG, and resolve input paths."""
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Manifest does not exist: {path}")

    text = manifest_path.read_text(encoding="utf-8")
    if "//" in text or "/*" in text:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"//.*", "", text)
        text = re.sub(r",(\s*[}\]])", r"\1", text)
    manifest = AnalysisManifest.model_validate(json.loads(text))

    declared_root = Path(manifest.metadata.root_directory).expanduser()
    root = declared_root if declared_root.is_absolute() else manifest_path.parent / declared_root
    manifest._manifest_path = manifest_path
    manifest._root_directory = root.resolve()
    artifacts = Path(manifest.metadata.artifacts_directory).expanduser()
    manifest._artifacts_directory = artifacts if artifacts.is_absolute() else (manifest.root_directory / artifacts).resolve()

    for correlator in manifest.inputs.correlators:
        correlator.data_path = _resolve_from_root(manifest.root_directory, correlator.data_path)
    for artifact in manifest.inputs.artifacts:
        artifact.path = _resolve_from_root(manifest.root_directory, artifact.path)
    for kernel in manifest.inputs.kernels:
        kernel.kernel_path = _resolve_from_root(manifest.root_directory, kernel.kernel_path)
    return manifest
