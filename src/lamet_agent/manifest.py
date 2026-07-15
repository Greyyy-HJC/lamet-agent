"""Manifest schema and validation for job-based LaMET workflows."""

from __future__ import annotations

import json
import math
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
    "review",
]
BzDirection = Literal["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]


HBAR_C_GEV_FM = 0.1973269804
_VOLUME_RE = re.compile(r"^S(?P<spatial>[1-9]\d*)T(?P<temporal>[1-9]\d*)$")
_MOMENTUM_RE = re.compile(r"^PX(?P<px>-?\d+)PY(?P<py>-?\d+)PZ(?P<pz>-?\d+)$")


def parse_volume(value: str) -> tuple[int, int]:
    """Return ``(L_s, L_t)`` from a canonical ``S<number>T<number>`` label."""
    match = _VOLUME_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"volume must use the form 'S48T64', got {value!r}")
    return int(match.group("spatial")), int(match.group("temporal"))


def parse_momentum(value: str) -> tuple[int, int, int]:
    """Return integer momentum components from ``PXnPYnPZn``."""
    match = _MOMENTUM_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"momentum must use the form 'PX0PY0PZ0', got {value!r}")
    return int(match.group("px")), int(match.group("py")), int(match.group("pz"))


def physical_momentum_gev(momentum: str, volume: str, lattice_spacing_fm: float) -> float:
    """Return the magnitude of a lattice momentum in GeV."""
    spatial, _temporal = parse_volume(volume)
    components = parse_momentum(momentum)
    norm = math.sqrt(sum(component * component for component in components))
    return 2.0 * math.pi * HBAR_C_GEV_FM * norm / (spatial * float(lattice_spacing_fm))


class RunMetadata(BaseModel):
    """Settings shared by every job in one run.

    Required: run_id, root_directory, target_observable, parton, resample_mode,
    random_seed, stages.
    Optional: artifacts_directory (default "artifacts"), sample_error_mode
    (default "covariance"), workers (default 1), bin_size (default: no
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
    sample_error_mode: Literal["mean", "median", "covariance"] = "covariance"
    random_seed: int
    workers: int = Field(default=1, ge=1, strict=True)
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
        if self.resample_mode == "jk" and self.sample_error_mode == "median":
            raise ValueError("metadata.sample_error_mode='median' is not supported with metadata.resample_mode='jk'")
        return self


class CorrelatorInput(BaseModel):
    """One raw 2pt or 3pt correlator dataset."""

    model_config = ConfigDict(extra="forbid")

    correlator_id: str
    correlator_type: Literal["2pt", "3pt"]
    data_path: str
    ensemble: str
    hadron: str
    gfix: str
    source_operator: str = Field(min_length=1)
    sink_operator: str = Field(min_length=1)
    current_operator: str | None = Field(default=None, min_length=1)
    bz_direction: BzDirection | None = None
    volume: str
    lattice_spacing_fm: float = Field(gt=0)
    momentum: list[str] = Field(min_length=1, strict=True)
    bT: list[int] | None = Field(default=None, strict=True)
    bz: list[int] | None = Field(default=None, strict=True)
    tsep: list[int] | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def validate_correlator_contract(self) -> "CorrelatorInput":
        parse_volume(self.volume)
        for value in self.momentum:
            parse_momentum(value)
        if len(set(self.momentum)) != len(self.momentum):
            raise ValueError("momentum must not contain duplicates")
        if self.correlator_type == "3pt":
            if not self.current_operator:
                raise ValueError("current_operator is required for 3pt correlators")
            if self.bz_direction is None:
                raise ValueError("bz_direction is required for 3pt correlators")
            if not self.tsep:
                raise ValueError("tsep must be a non-empty list for 3pt correlators")
            if len(set(self.tsep)) != len(self.tsep) or any(value <= 0 for value in self.tsep):
                raise ValueError("tsep must contain unique positive integers")
            if not self.bT:
                raise ValueError("bT must be a non-empty list for 3pt correlators")
            if not self.bz:
                raise ValueError("bz must be a non-empty list for 3pt correlators")
            if len(set(self.bT)) != len(self.bT):
                raise ValueError("bT must not contain duplicates")
            if len(set(self.bz)) != len(self.bz):
                raise ValueError("bz must not contain duplicates")
        elif any(value is not None for value in (self.current_operator, self.bz_direction, self.tsep, self.bT, self.bz)):
            raise ValueError("current_operator, bz_direction, tsep, bT, and bz are only valid for 3pt correlators")
        return self

    @property
    def spatial_extent(self) -> int:
        return parse_volume(self.volume)[0]

    @property
    def temporal_extent(self) -> int:
        return parse_volume(self.volume)[1]

    def momentum_gev(self, momentum: str) -> float:
        if momentum not in self.momentum:
            raise ValueError(f"momentum {momentum!r} is not declared by correlator {self.correlator_id!r}")
        return physical_momentum_gev(momentum, self.volume, self.lattice_spacing_fm)


class ArtifactInput(BaseModel):
    """Precomputed stage output that can seed a partial workflow."""

    model_config = ConfigDict(extra="allow")

    id: str
    stage: StageId
    path: str
    momentum: str | None = None
    volume: str | None = None
    lattice_spacing_fm: float | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_kinematics(cls, value: Any) -> Any:
        if isinstance(value, dict):
            removed = sorted(
                {
                    "a_fm",
                    "pz_gev",
                    "pz_out_gev",
                    "momentum_gev",
                    "initial_momentum_gev",
                    "final_momentum_gev",
                }.intersection(value)
            )
            if removed:
                raise ValueError(f"removed or derived artifact kinematics fields are not supported: {removed}")
        return value

    @model_validator(mode="after")
    def validate_kinematics(self) -> "ArtifactInput":
        supplied = (self.momentum, self.volume, self.lattice_spacing_fm)
        if any(value is not None for value in supplied) and not all(value is not None for value in supplied):
            raise ValueError("artifact momentum, volume, and lattice_spacing_fm must be declared together")
        if self.momentum is not None and self.volume is not None:
            parse_momentum(self.momentum)
            parse_volume(self.volume)
        return self

    @property
    def momentum_gev(self) -> float | None:
        if self.momentum is None or self.volume is None or self.lattice_spacing_fm is None:
            return None
        return physical_momentum_gev(self.momentum, self.volume, self.lattice_spacing_fm)


class KernelInput(BaseModel):
    """Stage kernel declaration (renormalization or perturbative matching)."""

    model_config = ConfigDict(extra="allow")

    stage: StageId
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
        for index, kernel in enumerate(self.inputs.kernels):
            if "zs_fm" in kernel.kernel_parameters:
                raise ValueError(
                    f"inputs.kernels[{index}].kernel_parameters.zs_fm is no longer supported; "
                    "use stages.perturbative_matching.defaults.zs_fm or "
                    "stages.perturbative_matching.jobs[].params.zs_fm"
                )

        renormalization = self.stages.get("renormalization")
        if renormalization is not None:
            nested_defaults = renormalization.defaults.get("scheme_parameters")
            if isinstance(nested_defaults, dict) and "zs_fm" in nested_defaults:
                raise ValueError(
                    "stages.renormalization.defaults.scheme_parameters.zs_fm is no longer supported; "
                    "use stages.renormalization.defaults.zs_fm"
                )
            for index, job in enumerate(renormalization.jobs):
                nested_params = job.params.get("scheme_parameters")
                if isinstance(nested_params, dict) and "zs_fm" in nested_params:
                    raise ValueError(
                        f"stages.renormalization.jobs[{index}].params.scheme_parameters.zs_fm is no longer "
                        f"supported; use stages.renormalization.jobs[{index}].params.zs_fm"
                    )

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


def derive_job_kinematics(manifest: AnalysisManifest, job: StageJob) -> dict[str, Any]:
    """Resolve manifest-authoritative discrete and physical kinematics for a job."""

    jobs = {
        candidate.id: (stage_id, candidate)
        for stage_id, config in manifest.stages.items()
        for candidate in config.jobs
    }
    artifacts = {artifact.id: artifact for artifact in manifest.inputs.artifacts}

    def from_reference(reference: str, seen: set[str]) -> dict[str, Any]:
        artifact = artifacts.get(reference)
        if artifact is not None:
            if artifact.momentum is None:
                return {}
            return {
                "momentum": artifact.momentum,
                "volume": artifact.volume,
                "lattice_spacing_fm": artifact.lattice_spacing_fm,
                "momentum_gev": artifact.momentum_gev,
            }
        found = jobs.get(reference)
        if found is None or reference in seen:
            return {}
        return from_job(*found, seen | {reference})

    def from_job(stage_id: str, candidate: StageJob, seen: set[str]) -> dict[str, Any]:
        if stage_id == "correlator_analysis":
            params = {**manifest.stages[stage_id].defaults, **candidate.params}
            momentum = (
                params.get("final_momentum")
                if str(params.get("fitting_form", "Breit")) == "NonBreit"
                else params.get("momentum")
            )
            selected = [
                item
                for item in manifest.correlators
                if item.correlator_id in candidate.correlator_ids
                and item.correlator_type == "2pt"
                and momentum in item.momentum
            ]
            if not selected or momentum is None:
                return {}
            correlator = selected[0]
            result = {
                "momentum": momentum,
                "volume": correlator.volume,
                "lattice_spacing_fm": correlator.lattice_spacing_fm,
                "momentum_gev": correlator.momentum_gev(momentum),
                "hadron": correlator.hadron,
                "gfix": correlator.gfix,
            }
            if str(params.get("fitting_form", "Breit")) == "NonBreit":
                initial = params.get("initial_momentum")
                initial_source = next(
                    (
                        item
                        for item in manifest.correlators
                        if item.correlator_id in candidate.correlator_ids
                        and item.correlator_type == "2pt"
                        and initial in item.momentum
                    ),
                    None,
                )
                result.update(
                    {
                        "initial_momentum": initial,
                        "final_momentum": momentum,
                        "initial_momentum_gev": (
                            initial_source.momentum_gev(initial)
                            if initial_source is not None and initial is not None
                            else None
                        ),
                        "final_momentum_gev": correlator.momentum_gev(momentum),
                    }
                )
                result["momentum_gev"] = result["initial_momentum_gev"]
            return result

        for role in ("input", "quasi", "target", "reference", "denominator", "zR"):
            value = candidate.inputs.get(role)
            references = value if isinstance(value, list) else [value] if value is not None else []
            for reference in references:
                resolved = from_reference(reference, seen)
                if resolved:
                    return resolved
        return {}

    stage = next((stage_id for stage_id, config in manifest.stages.items() if job in config.jobs), None)
    return {} if stage is None else from_job(stage, job, {job.id})


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
