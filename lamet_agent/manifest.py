"""Manifest schema and validation for job-based LaMET workflows."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import gvar as gv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from lamet_agent.manifest_params import resolve_stage_params, validate_stage_parameter_mapping


StageId = Literal[
    "correlator_analysis",
    "renormalization",
    "fourier_transform",
    "perturbative_matching",
    "extrapolation",
    "review",
]
BzDirection = Literal["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]
CoordUnit = Literal["lattice", "fm", "gev_inv", "lambda"]


_VOLUME_RE = re.compile(r"^S(?P<spatial>[1-9]\d*)T(?P<temporal>[1-9]\d*)$")
_MOMENTUM_RE = re.compile(r"^PX(?P<px>-?\d+)PY(?P<py>-?\d+)PZ(?P<pz>-?\d+)$")


class ManifestPathError(ValueError):
    """Raised when a CLI manifest references an invalid filesystem path."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(issues))


def lamet_agent_project_root() -> Path:
    """Return the source checkout root containing the active package."""
    return Path(__file__).resolve().parents[1]


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
    from lamet_agent.core.data import HBAR_C_GEV_FM

    spatial, _temporal = parse_volume(volume)
    components = parse_momentum(momentum)
    norm = math.sqrt(sum(component * component for component in components))
    return 2.0 * math.pi * HBAR_C_GEV_FM * norm / (spatial * float(lattice_spacing_fm))


class RunMetadata(BaseModel):
    """Settings shared by every job in one run.

    Required: run_id, root_directory, target_observable, parton, resample_mode,
    sample_error_mode, random_seed, stages.
    Optional: artifacts_directory (default "artifacts"), workers (default 1), bin_size (default: no
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
    sample_error_mode: Literal["mean", "median", "covariance"]
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
    polarization: Literal["unpolarized", "helicity", "transversity"] | None = None
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
            if self.polarization is None:
                raise ValueError("polarization is required for 3pt correlators")
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
        elif self.current_operator is not None or self.tsep is not None or "polarization" in self.model_fields_set:
            raise ValueError("current_operator, polarization, and tsep are only valid for 3pt correlators")
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

    _resolved_metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    _metadata_resolved: bool = PrivateAttr(default=False)

    id: str
    stage: StageId
    path: str
    momentum: str | None = None
    volume: str | None = None
    lattice_spacing_fm: float | None = Field(default=None, gt=0)
    hadron: str | None = None
    gfix: str | None = None
    bz_direction: BzDirection | None = None
    coord_unit: CoordUnit | None = None

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
        metadata = self.resolved_metadata
        momentum = metadata.get("momentum")
        volume = metadata.get("volume")
        lattice_spacing_fm = metadata.get("lattice_spacing_fm")
        if momentum is None or volume is None or lattice_spacing_fm is None:
            return None
        return physical_momentum_gev(str(momentum), str(volume), float(lattice_spacing_fm))

    @property
    def declared_metadata(self) -> dict[str, Any]:
        """Return user-declared artifact metadata, excluding graph identity fields."""
        return self.model_dump(exclude={"id", "stage", "path"}, exclude_none=True)

    @property
    def resolved_metadata(self) -> dict[str, Any]:
        """Return cached file/manifest metadata, or manifest metadata before resolution."""
        if self._metadata_resolved:
            return dict(self._resolved_metadata)
        return self.declared_metadata


_ARTIFACT_METADATA_FIELDS = (
    "momentum",
    "volume",
    "lattice_spacing_fm",
    "hadron",
    "gfix",
    "bz_direction",
    "coord_unit",
    "observable",
    "parton",
    "current_operator",
    "polarization",
)
_NETCDF_STORAGE_ATTRS = frozenset({"ensemble", "resample", "gvar_encoding"})


def _normalize_artifact_metadata_value(artifact: ArtifactInput, key: str, value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        if key == "lattice_spacing_fm":
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError("must be a finite positive number")
            return normalized
        normalized = str(value)
        if key == "momentum":
            parse_momentum(normalized)
        elif key == "volume":
            parse_volume(normalized)
        elif key == "bz_direction" and normalized not in {"X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"}:
            raise ValueError("must be a canonical axis-set label")
        elif key == "coord_unit" and normalized not in {"lattice", "fm", "gev_inv", "lambda"}:
            raise ValueError("must be one of lattice, fm, gev_inv, lambda")
        return normalized
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"artifact {artifact.id!r} has invalid {key!r} metadata in {artifact.path!r}: {value!r}"
        ) from exc


def resolve_artifact_metadata(artifact: ArtifactInput) -> dict[str, Any]:
    """Merge one artifact's NetCDF attrs with optional manifest fallback metadata."""
    if artifact._metadata_resolved:
        return artifact.resolved_metadata

    declared = artifact.declared_metadata
    file_metadata: dict[str, Any] = {}
    path = Path(artifact.path)
    if path.suffix.lower() == ".nc" and path.is_file():
        from lamet_agent.core.data import read_netcdf_attrs

        file_metadata = {
            key: value
            for key, value in read_netcdf_attrs(path).items()
            if key not in _NETCDF_STORAGE_ATTRS
        }

    resolved = dict(file_metadata)
    for key, value in declared.items():
        resolved.setdefault(key, value)

    for key in _ARTIFACT_METADATA_FIELDS:
        file_value = _normalize_artifact_metadata_value(artifact, key, file_metadata.get(key))
        declared_value = _normalize_artifact_metadata_value(artifact, key, declared.get(key))
        if file_value is not None and declared_value is not None:
            matches = (
                math.isclose(file_value, declared_value, rel_tol=1e-12, abs_tol=1e-12)
                if key == "lattice_spacing_fm"
                else file_value == declared_value
            )
            if not matches:
                raise ValueError(
                    f"artifact {artifact.id!r} metadata conflict for {key!r}: "
                    f"NetCDF attrs={file_value!r}, manifest={declared_value!r} ({artifact.path})"
                )
        selected = file_value if file_value is not None else declared_value
        if selected is not None:
            resolved[key] = selected
        else:
            resolved.pop(key, None)

    artifact._resolved_metadata = resolved
    artifact._metadata_resolved = True
    return artifact.resolved_metadata


def resolve_manifest_artifact_metadata(manifest: "AnalysisManifest") -> None:
    """Resolve and cache metadata for every external artifact in a manifest."""
    for artifact in manifest.inputs.artifacts:
        resolve_artifact_metadata(artifact)


class KernelInput(BaseModel):
    """Stage kernel declaration (renormalization or perturbative matching)."""

    model_config = ConfigDict(extra="allow")

    stage: StageId
    kernel_id: str
    kernel_path: str
    kernel_parameters: dict[str, Any] = Field(default_factory=dict)


class ManifestInputs(BaseModel):
    """Global source-node pools."""

    correlators: list[CorrelatorInput] = Field(default_factory=list)
    artifacts: list[ArtifactInput] = Field(default_factory=list)
    kernels: list[KernelInput] = Field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def is_constant_job_input(value: Any) -> bool:
    """Return True if a job input is a numeric constant rather than an upstream id."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def job_input_refs(value: Any) -> list[str]:
    """Return upstream id strings from a job input, ignoring numeric constants."""
    items = value if isinstance(value, list) else [value]
    return [item for item in items if isinstance(item, str)]


class StageJob(BaseModel):
    """One independently executed stage job."""

    id: str
    correlator_ids: list[str] = Field(default_factory=list)
    inputs: dict[str, str | list[str] | int | float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("inputs")
    @classmethod
    def validate_input_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        for role, item in value.items():
            if isinstance(item, list):
                if not all(isinstance(entry, str) for entry in item):
                    raise ValueError(f"job input {role!r} lists must contain only upstream ids")
                continue
            if isinstance(item, str):
                continue
            if not is_constant_job_input(item):
                raise ValueError(
                    f"job input {role!r} must be an upstream id, a list of ids, or a numeric constant"
                )
            if not math.isfinite(float(item)):
                raise ValueError(f"job input {role!r} constant must be a finite number")
        return value


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
        parameter_issues: list[str] = []
        for stage, config in self.stages.items():
            parameter_issues.extend(
                validate_stage_parameter_mapping(
                    stage,
                    config.defaults,
                    path=f"stages.{stage}.defaults",
                )
            )
            for index, job in enumerate(config.jobs):
                parameter_issues.extend(
                    validate_stage_parameter_mapping(
                        stage,
                        job.params,
                        path=f"stages.{stage}.jobs[{index}].params",
                    )
                )
        if parameter_issues:
            details = "\n".join(f"- {issue}" for issue in parameter_issues)
            raise ValueError(f"Unsupported stage manifest parameters:\n{details}")

        for index, kernel in enumerate(self.inputs.kernels):
            if "scheme" in (kernel.model_extra or {}):
                raise ValueError(
                    f"inputs.kernels[{index}].scheme is no longer supported; "
                    f"move it to stages.{kernel.stage}.defaults.scheme or the "
                    "corresponding jobs[].params.scheme"
                )
            if "zs_fm" in kernel.kernel_parameters:
                raise ValueError(
                    f"inputs.kernels[{index}].kernel_parameters.zs_fm is no longer supported; "
                    "use stages.perturbative_matching.defaults.zs_fm or "
                    "stages.perturbative_matching.jobs[].params.zs_fm"
                )
            if kernel.stage == "renormalization":
                removed_kernel_parameters = {
                    "LambdaQCD": (
                        "was renamed; declare stages.renormalization.defaults.LambdaQCD_gev "
                        "or jobs[].params.LambdaQCD_gev explicitly"
                    ),
                    "LambdaQCD_gev": (
                        "is a hybrid-self-renormalization ansatz parameter; declare it under "
                        "stages.renormalization.defaults.LambdaQCD_gev or jobs[].params.LambdaQCD_gev"
                    ),
                    "alpha_s": "is derived from mu by alphas_nloop and cannot be specified",
                    "b0": "is an internal hybrid-self-renormalization ansatz constant",
                    "cf": "is an internal hybrid-self-renormalization ansatz constant",
                    "f1_extension_zmin_fm": "is no longer supported",
                    "k": "is an internal hybrid-self-renormalization ansatz constant",
                    "lqcd": (
                        "was renamed; use stages.renormalization.defaults.LambdaQCD_gev "
                        "or jobs[].params.LambdaQCD_gev"
                    ),
                    "Nf": "is not configurable for renormalization; self-renormalization uses alphas_nloop(mu)",
                    "order": "is not configurable for renormalization; self-renormalization uses alphas_nloop(mu)",
                    "zms_kind": "is no longer supported; select the kernel_id instead",
                    "zr_zmax_fm": "is no longer supported",
                }
                for key, message in removed_kernel_parameters.items():
                    if key in kernel.kernel_parameters:
                        raise ValueError(
                            f"inputs.kernels[{index}].kernel_parameters.{key} {message}."
                        )

        if len(set(self.metadata.stages)) != len(self.metadata.stages):
            raise ValueError("metadata.stages contains duplicate stage ids")
        missing = [stage for stage in self.metadata.stages if stage not in self.stages]
        if missing:
            raise ValueError(f"metadata.stages has no job configuration for: {missing}")
        unused = [stage for stage in self.stages if stage not in self.metadata.stages]
        if unused:
            raise ValueError(
                f"stages has configuration for unused stages {unused} that are not listed in "
                "metadata.stages. Add them to metadata.stages to run them, or remove the unused "
                "stage blocks."
            )

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
                    unknown = [ref for ref in job_input_refs(value) if ref not in known]
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
            metadata = artifact.resolved_metadata
            momentum = metadata.get("momentum")
            volume = metadata.get("volume")
            lattice_spacing_fm = metadata.get("lattice_spacing_fm")
            result = {
                key: metadata[key]
                for key in (
                    "hadron",
                    "gfix",
                    "bz_direction",
                    "initial_momentum",
                    "final_momentum",
                    "current_operator",
                    "polarization",
                    "observable",
                    "parton",
                )
                if metadata.get(key) is not None
            }
            if momentum is None or volume is None or lattice_spacing_fm is None:
                return result
            result.update(
                {
                    "momentum": str(momentum),
                    "volume": str(volume),
                    "lattice_spacing_fm": float(lattice_spacing_fm),
                    "momentum_gev": artifact.momentum_gev,
                }
            )
            return result
        found = jobs.get(reference)
        if found is None or reference in seen:
            return {}
        return from_job(*found, seen | {reference})

    def from_job(stage_id: str, candidate: StageJob, seen: set[str]) -> dict[str, Any]:
        if stage_id == "correlator_analysis":
            params = resolve_stage_params(stage_id, manifest.stages[stage_id].defaults, candidate.params)
            momentum = (
                params.get("final_momentum")
                if str(params.get("fitting_form")) == "NonBreit"
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
            operator = next(
                (
                    item
                    for item in manifest.correlators
                    if item.correlator_id in candidate.correlator_ids
                    and item.correlator_type == "3pt"
                    and momentum in item.momentum
                ),
                None,
            )
            if operator is not None:
                result.update(
                    {
                        "current_operator": operator.current_operator,
                        "polarization": operator.polarization,
                    }
                )
            if str(params.get("fitting_form")) == "NonBreit":
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
            references = job_input_refs(value)
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


def validate_manifest_paths(
    manifest: AnalysisManifest,
    *,
    project_root: Path | None = None,
) -> None:
    """Require the CLI manifest root and input paths to reference existing files."""
    expected_root = (project_root or lamet_agent_project_root()).expanduser().resolve()
    actual_root = manifest.root_directory.expanduser().resolve()
    if actual_root != expected_root:
        raise ManifestPathError(
            [
                (
                    "metadata.root_directory must resolve to the lamet-agent project root: "
                    f"expected {expected_root}, got {actual_root}"
                )
            ]
        )

    issues: list[str] = []

    def check_file(field: str, value: str) -> None:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = actual_root / path
        resolved = path.resolve()
        if not resolved.exists():
            issues.append(f"{field} does not exist: {resolved}")
        elif not resolved.is_file():
            issues.append(f"{field} is not a file: {resolved}")

    for index, correlator in enumerate(manifest.inputs.correlators):
        check_file(f"inputs.correlators[{index}].data_path", correlator.data_path)
    for index, artifact in enumerate(manifest.inputs.artifacts):
        check_file(f"inputs.artifacts[{index}].path", artifact.path)
    for index, kernel in enumerate(manifest.inputs.kernels):
        check_file(f"inputs.kernels[{index}].kernel_path", kernel.kernel_path)

    if issues:
        raise ManifestPathError(issues)


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
    renorm = manifest.stages.get("renormalization")
    fourier = manifest.stages.get("fourier_transform")
    matching = manifest.stages.get("perturbative_matching")
    extrapolation = manifest.stages.get("extrapolation")
    systematics_requested = (
        renorm is not None
        and fourier is not None
        and matching is not None
        and extrapolation is not None
        and (
            isinstance(renorm.defaults.get("zs_fm"), str)
            or any(isinstance(job.params.get("zs_fm"), str) for job in renorm.jobs)
            or "zmin_shift" in fourier.defaults
            or any("zmin_shift" in job.params for job in fourier.jobs)
            or "r" in matching.defaults
            or any("r" in job.params for job in matching.jobs)
            or any(key in extrapolation.defaults for key in ("allow_order_a_sym", "allow_order_1overp_sym", "allow_order_ap_sym"))
            or any(
                key in job.params
                for job in extrapolation.jobs
                for key in ("allow_order_a_sym", "allow_order_1overp_sym", "allow_order_ap_sym")
            )
        )
    )
    if systematics_requested:
        renorm_defaults = dict(renorm.defaults)
        fourier_defaults = dict(fourier.defaults)
        matching_defaults = dict(matching.defaults)
        extrapolation_defaults = dict(extrapolation.defaults)
        renorm_default_raw_zs = renorm_defaults.get("zs_fm")
        if "zs_fm" in renorm_defaults and isinstance(renorm_defaults["zs_fm"], str):
            zs = gv.gvar(str(renorm_defaults["zs_fm"]))
            renorm_defaults["zs_fm"] = float(gv.mean(zs))
        if "zs_fm" in matching_defaults and isinstance(matching_defaults["zs_fm"], str):
            zs = gv.gvar(str(matching_defaults["zs_fm"]))
            matching_defaults["zs_fm"] = float(gv.mean(zs))
        elif "zs_fm" in renorm_defaults and "zs_fm" not in matching_defaults:
            matching_defaults["zs_fm"] = renorm_defaults["zs_fm"]
        fourier_shift_default = int(fourier_defaults.get("zmin_shift", 0) or 0)
        matching_r_default = float(matching_defaults.get("r", 1.0) or 1.0)
        a_sym_default = list(extrapolation_defaults.get("allow_order_a_sym", []) or [])
        p_sym_default = list(extrapolation_defaults.get("allow_order_1overp_sym", []) or [])
        ap_sym_default = list(extrapolation_defaults.get("allow_order_ap_sym", []) or [])
        renorm.defaults = {key: value for key, value in renorm_defaults.items()}
        fourier.defaults = {key: value for key, value in fourier_defaults.items() if key != "zmin_shift"}
        matching.defaults = {key: value for key, value in matching_defaults.items() if key != "r"}
        extrapolation.defaults = {
            key: value
            for key, value in extrapolation_defaults.items()
            if key not in {"allow_order_a_sym", "allow_order_1overp_sym", "allow_order_ap_sym"}
        }
        renorm_jobs: list[StageJob] = []
        renorm_zs_variant_ids: dict[str, dict[str, str]] = {}
        renorm_variant_zs: dict[str, float] = {}
        for job in renorm.jobs:
            main = job.model_copy(deep=True)
            raw_zs = main.params.get("zs_fm", renorm_default_raw_zs)
            zs_mean = raw_zs
            zs_sdev = 0.0
            if isinstance(raw_zs, str):
                zs = gv.gvar(raw_zs)
                zs_mean = float(gv.mean(zs))
                zs_sdev = float(gv.sdev(zs))
            elif raw_zs is not None:
                zs_mean = float(raw_zs)
            if zs_mean is not None:
                main.params["zs_fm"] = float(zs_mean)
            renorm_jobs.append(main)
            if zs_sdev > 0.0:
                low = main.model_copy(deep=True)
                low.id = f"{main.id}_zs_low"
                low.params["zs_fm"] = float(zs_mean - zs_sdev)
                high = main.model_copy(deep=True)
                high.id = f"{main.id}_zs_high"
                high.params["zs_fm"] = float(zs_mean + zs_sdev)
                renorm_jobs.extend([low, high])
                renorm_zs_variant_ids[main.id] = {"low": low.id, "high": high.id}
                renorm_variant_zs[low.id] = float(zs_mean - zs_sdev)
                renorm_variant_zs[high.id] = float(zs_mean + zs_sdev)
        renorm.jobs = renorm_jobs
        fourier_jobs: list[StageJob] = []
        fourier_variant_ids: dict[str, dict[str, str]] = {}
        fourier_variant_zs: dict[str, float] = {}
        for job in fourier.jobs:
            main = job.model_copy(deep=True)
            raw_shift = int(main.params.get("zmin_shift", fourier_shift_default) or 0)
            main.params.pop("zmin_shift", None)
            fourier_jobs.append(main)
            input_ref = str(main.inputs.get("input", ""))
            variants: dict[str, str] = {}
            if input_ref in renorm_zs_variant_ids:
                for tag in ("low", "high"):
                    clone = main.model_copy(deep=True)
                    clone.id = f"{main.id}_zs_{tag}"
                    clone.inputs["input"] = renorm_zs_variant_ids[input_ref][tag]
                    fourier_jobs.append(clone)
                    variants[f"zs_{tag}"] = clone.id
                    fourier_variant_zs[clone.id] = renorm_variant_zs[clone.inputs["input"]]
            if raw_shift:
                for tag, shift in (("low", -abs(raw_shift)), ("high", abs(raw_shift))):
                    clone = main.model_copy(deep=True)
                    clone.id = f"{main.id}_lambda_{tag}"
                    clone.params["zmin_shift"] = int(shift)
                    fourier_jobs.append(clone)
                    variants[f"lambda_{tag}"] = clone.id
            fourier_variant_ids[main.id] = variants
        fourier.jobs = fourier_jobs
        matching_jobs: list[StageJob] = []
        matching_variant_ids: dict[str, dict[str, str]] = {}
        for job in matching.jobs:
            main = job.model_copy(deep=True)
            raw_r = float(main.params.get("r", matching_r_default) or 1.0)
            main.params.pop("r", None)
            matching_jobs.append(main)
            quasi_ref = str(main.inputs.get("quasi", ""))
            variants: dict[str, str] = {}
            parent_ft = quasi_ref
            if parent_ft in fourier_variant_ids:
                for key, suffix in (("zs_low", "zs_low"), ("zs_high", "zs_high"), ("lambda_low", "lambda_low"), ("lambda_high", "lambda_high")):
                    if key in fourier_variant_ids[parent_ft]:
                        clone = main.model_copy(deep=True)
                        clone.id = f"{main.id}_{suffix}"
                        clone.inputs["quasi"] = fourier_variant_ids[parent_ft][key]
                        if key.startswith("zs_"):
                            clone.params["zs_fm"] = fourier_variant_zs[clone.inputs["quasi"]]
                        matching_jobs.append(clone)
                        variants[key] = clone.id
            if raw_r != 1.0:
                mu = float(resolve_stage_params("perturbative_matching", matching.defaults, main.params)["mu"])
                low = main.model_copy(deep=True)
                low.id = f"{main.id}_mu_low"
                low.params["mu"] = mu / raw_r
                high = main.model_copy(deep=True)
                high.id = f"{main.id}_mu_high"
                high.params["mu"] = mu * raw_r
                matching_jobs.extend([low, high])
                variants["mu_low"] = low.id
                variants["mu_high"] = high.id
            matching_variant_ids[main.id] = variants
        matching.jobs = matching_jobs
        extrapolation_jobs: list[StageJob] = []
        original_jobs = list(extrapolation.jobs)
        for job in original_jobs:
            main = job.model_copy(deep=True)
            main.params.pop("allow_order_a_sym", None)
            main.params.pop("allow_order_1overp_sym", None)
            main.params.pop("allow_order_ap_sym", None)
            extrapolation_jobs.append(main)
            refs = [str(ref) for ref in list(main.inputs.get("lightcone", []))]
            zs_low = [matching_variant_ids.get(ref, {}).get("zs_low") for ref in refs]
            zs_high = [matching_variant_ids.get(ref, {}).get("zs_high") for ref in refs]
            lambda_low = [matching_variant_ids.get(ref, {}).get("lambda_low") for ref in refs]
            lambda_high = [matching_variant_ids.get(ref, {}).get("lambda_high") for ref in refs]
            mu_low = [matching_variant_ids.get(ref, {}).get("mu_low") for ref in refs]
            mu_high = [matching_variant_ids.get(ref, {}).get("mu_high") for ref in refs]
            budget_inputs: dict[str, Any] = {"main": main.id}
            if all(zs_low):
                clone = main.model_copy(deep=True)
                clone.id = f"{main.id}_zs_low" if main.id != "ex_main" else "ex_zs_low"
                clone.inputs["lightcone"] = list(zs_low)
                extrapolation_jobs.append(clone)
                budget_inputs["zs"] = [clone.id]
            if all(zs_high):
                clone = main.model_copy(deep=True)
                clone.id = f"{main.id}_zs_high" if main.id != "ex_main" else "ex_zs_high"
                clone.inputs["lightcone"] = list(zs_high)
                extrapolation_jobs.append(clone)
                budget_inputs.setdefault("zs", []).append(clone.id)
            if all(lambda_low):
                clone = main.model_copy(deep=True)
                clone.id = f"{main.id}_lambda_low" if main.id != "ex_main" else "ex_lambda_low"
                clone.inputs["lightcone"] = list(lambda_low)
                extrapolation_jobs.append(clone)
                budget_inputs["lambda_extrapolation"] = [clone.id]
            if all(lambda_high):
                clone = main.model_copy(deep=True)
                clone.id = f"{main.id}_lambda_high" if main.id != "ex_main" else "ex_lambda_high"
                clone.inputs["lightcone"] = list(lambda_high)
                extrapolation_jobs.append(clone)
                budget_inputs.setdefault("lambda_extrapolation", []).append(clone.id)
            if all(mu_low):
                clone = main.model_copy(deep=True)
                clone.id = f"{main.id}_mu_low" if main.id != "ex_main" else "ex_mu_low"
                clone.inputs["lightcone"] = list(mu_low)
                extrapolation_jobs.append(clone)
                budget_inputs["lamet_scale"] = [clone.id]
            if all(mu_high):
                clone = main.model_copy(deep=True)
                clone.id = f"{main.id}_mu_high" if main.id != "ex_main" else "ex_mu_high"
                clone.inputs["lightcone"] = list(mu_high)
                extrapolation_jobs.append(clone)
                budget_inputs.setdefault("lamet_scale", []).append(clone.id)
            other_ids: list[str] = []
            a_sym = list(job.params.get("allow_order_a_sym", a_sym_default) or [])
            p_sym = list(job.params.get("allow_order_1overp_sym", p_sym_default) or [])
            ap_sym = list(job.params.get("allow_order_ap_sym", ap_sym_default) or [])
            if a_sym:
                clone = main.model_copy(deep=True)
                clone.id = f"{main.id}_a_sym"
                clone.params["allow_order_a"] = a_sym
                extrapolation_jobs.append(clone)
                other_ids.append(clone.id)
            if p_sym:
                clone = main.model_copy(deep=True)
                clone.id = f"{main.id}_p_sym" if main.id != "ex_main" else "ex_other"
                clone.params["allow_order_1overp"] = p_sym
                extrapolation_jobs.append(clone)
                other_ids.append(clone.id)
            if ap_sym:
                clone = main.model_copy(deep=True)
                clone.id = f"{main.id}_ap_sym"
                clone.params["allow_order_ap"] = ap_sym
                extrapolation_jobs.append(clone)
                other_ids.append(clone.id)
            if any(key in budget_inputs for key in ("zs", "lambda_extrapolation", "lamet_scale")) or other_ids:
                budget = StageJob(
                    id="ex_budget" if main.id == "ex_main" else f"{main.id}_budget",
                    inputs={**budget_inputs, **({"other_extrapolations": other_ids} if other_ids else {})},
                    params={"operation": "systematics_budget"},
                )
                extrapolation_jobs.append(budget)
        extrapolation.jobs = extrapolation_jobs
    resolve_manifest_artifact_metadata(manifest)
    return manifest
