"""Manifest models and validation helpers for lamet-agent workflows."""

from __future__ import annotations

from itertools import product
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lamet_agent.errors import ManifestValidationError
from lamet_agent.utils import resolve_manifest_relative_path

SUPPORTED_GOALS = {
    "parton_distribution_function",
    "distribution_amplitude",
    "custom",
}
SUPPORTED_CORRELATOR_KINDS = {
    "two_point",
    "three_point",
    "four_point",
    "custom",
}
SUPPORTED_DATA_FORMATS = {"csv", "npz", "txt"}
SUPPORTED_PLOT_FORMATS = {"png", "svg", "pdf"}
SUPPORTED_EXPORT_FORMATS = {"csv", "npz", "json"}
SUPPORTED_PURPOSES = {"smoke", "physics"}
SUPPORTED_GAUGES = {"cg", "gi"}
SUPPORTED_HADRONS = {"pion", "proton"}
SUPPORTED_ANALYSIS_CHANNELS = {"qpdf", "qda"}
_SETUP_REQUIRED_KEYS = (
    "lattice_action",
    "n_f",
    "lattice_spacing_fm",
    "spatial_extent",
    "temporal_extent",
    "pion_mass_valence_gev",
    "pion_mass_sea_gev",
)


def _format_nested(value: Any, mapping: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _format_nested(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_nested(item, mapping) for item in value]
    return _format_with_mapping(value, mapping)


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{context} must be an object.")
    return dict(value)


def _require_sequence_of_length(value: Any, *, context: str, length: int) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        raise ManifestValidationError(f"{context} must be a list of length {length}.")
    return list(value)


def _validate_correlator_metadata(kind: str, metadata: dict[str, Any]) -> dict[str, Any]:
    setup_id = metadata.get("setup_id")
    if not isinstance(setup_id, str) or not setup_id.strip():
        raise ManifestValidationError("Correlator metadata must define a non-empty string 'setup_id'.")

    momentum = _require_sequence_of_length(metadata.get("momentum"), context="Correlator metadata 'momentum'", length=3)
    smearing = metadata.get("smearing")
    if not isinstance(smearing, str) or not smearing.strip():
        raise ManifestValidationError("Correlator metadata must define a non-empty string 'smearing'.")

    normalized = dict(metadata)
    normalized["setup_id"] = setup_id
    normalized["momentum"] = [int(component) for component in momentum]
    normalized["smearing"] = smearing

    if kind == "three_point":
        displacement = _require_mapping(metadata.get("displacement"), context="Three-point metadata 'displacement'")
        operator = _require_mapping(metadata.get("operator"), context="Three-point metadata 'operator'")
        for key in ("b", "z"):
            if key not in displacement:
                raise ManifestValidationError(f"Three-point metadata.displacement must define '{key}'.")
        for key in ("gamma", "flavor"):
            value = operator.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ManifestValidationError(f"Three-point metadata.operator must define a non-empty string '{key}'.")
        normalized["displacement"] = {
            "b": int(displacement["b"]),
            "z": int(displacement["z"]),
        }
        normalized["operator"] = {
            "gamma": str(operator["gamma"]),
            "flavor": str(operator["flavor"]),
        }
    return normalized


def _validate_manifest_metadata(metadata: dict[str, Any], goal: str) -> dict[str, Any]:
    purpose = str(metadata.get("purpose", "")).lower()
    if purpose not in SUPPORTED_PURPOSES:
        raise ManifestValidationError(
            f"Manifest metadata.purpose must be one of {sorted(SUPPORTED_PURPOSES)}, got {purpose!r}."
        )

    analysis = _require_mapping(metadata.get("analysis"), context="Manifest metadata.analysis")
    gauge = str(analysis.get("gauge", "")).lower()
    hadron = str(analysis.get("hadron", "")).lower()
    channel = str(analysis.get("channel", "")).lower()
    if gauge not in SUPPORTED_GAUGES:
        raise ManifestValidationError(
            f"Manifest metadata.analysis.gauge must be one of {sorted(SUPPORTED_GAUGES)}, got {gauge!r}."
        )
    if hadron not in SUPPORTED_HADRONS:
        raise ManifestValidationError(
            f"Manifest metadata.analysis.hadron must be one of {sorted(SUPPORTED_HADRONS)}, got {hadron!r}."
        )
    if channel not in SUPPORTED_ANALYSIS_CHANNELS:
        raise ManifestValidationError(
            f"Manifest metadata.analysis.channel must be one of {sorted(SUPPORTED_ANALYSIS_CHANNELS)}, got {channel!r}."
        )
    if goal == "parton_distribution_function" and channel != "qpdf":
        raise ManifestValidationError("goal='parton_distribution_function' requires metadata.analysis.channel='qpdf'.")
    if goal == "distribution_amplitude" and channel != "qda":
        raise ManifestValidationError("goal='distribution_amplitude' requires metadata.analysis.channel='qda'.")

    conventions = metadata.get("conventions")
    if not isinstance(conventions, str) or not conventions.strip():
        raise ManifestValidationError("Manifest metadata must define a non-empty string 'conventions'.")

    setups = _require_mapping(metadata.get("setups"), context="Manifest metadata.setups")
    if not setups:
        raise ManifestValidationError("Manifest metadata.setups must define at least one setup.")
    normalized_setups: dict[str, dict[str, Any]] = {}
    for setup_id, raw_setup in setups.items():
        if not isinstance(setup_id, str) or not setup_id.strip():
            raise ManifestValidationError("Manifest metadata.setups keys must be non-empty strings.")
        setup = _require_mapping(raw_setup, context=f"Manifest metadata.setups[{setup_id!r}]")
        missing = [key for key in _SETUP_REQUIRED_KEYS if key not in setup]
        if missing:
            raise ManifestValidationError(
                f"Manifest metadata.setups[{setup_id!r}] is missing required key(s): {missing}."
            )
        normalized_setups[setup_id] = {
            "lattice_action": str(setup["lattice_action"]),
            "n_f": int(setup["n_f"]),
            "lattice_spacing_fm": float(setup["lattice_spacing_fm"]),
            "spatial_extent": int(setup["spatial_extent"]),
            "temporal_extent": int(setup["temporal_extent"]),
            "pion_mass_valence_gev": float(setup["pion_mass_valence_gev"]),
            "pion_mass_sea_gev": float(setup["pion_mass_sea_gev"]),
        }

    normalized = dict(metadata)
    normalized["purpose"] = purpose
    normalized["analysis"] = {
        "gauge": gauge,
        "hadron": hadron,
        "channel": channel,
    }
    normalized["conventions"] = conventions
    normalized["setups"] = normalized_setups
    return normalized


def _normalize_expansion_values(name: str, raw_values: Any) -> list[Any]:
    if isinstance(raw_values, list):
        return raw_values
    if isinstance(raw_values, dict):
        if "values" in raw_values:
            values = raw_values["values"]
            if not isinstance(values, list):
                raise ManifestValidationError(f"Correlator expand.{name}['values'] must be a list.")
            return values
        if "start" in raw_values and "stop" in raw_values:
            start = int(raw_values["start"])
            stop = int(raw_values["stop"])
            step = int(raw_values.get("step", 1))
            if step == 0:
                raise ManifestValidationError(f"Correlator expand.{name}.step must be non-zero.")
            inclusive = bool(raw_values.get("inclusive", True))
            stop_value = stop + (1 if inclusive and step > 0 else -1 if inclusive else 0)
            return list(range(start, stop_value, step))
    raise ManifestValidationError(
        f"Correlator expand.{name} must be a list or a range-like object with start/stop[/step]."
    )


def _format_with_mapping(value: Any, mapping: dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**mapping)
        except KeyError:
            return value
    return value


def _expand_correlator_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    expand = entry.get("expand")
    if not expand:
        return [entry]
    if not isinstance(expand, dict) or not expand:
        raise ManifestValidationError("Correlator 'expand' must be a non-empty object.")

    keys = list(expand)
    value_lists = [_normalize_expansion_values(name, expand[name]) for name in keys]
    expanded_entries: list[dict[str, Any]] = []
    for values in product(*value_lists):
        mapping = {key: value for key, value in zip(keys, values, strict=True)}
        expanded = {key: value for key, value in entry.items() if key != "expand"}
        for field in ("path", "label"):
            if field in expanded:
                expanded[field] = _format_with_mapping(expanded[field], mapping)
        metadata = _format_nested(dict(expanded.get("metadata", {})), mapping)
        expanded["metadata"] = metadata
        expanded_entries.append(expanded)
    return expanded_entries


@dataclass(slots=True)
class CorrelatorSpec:
    """Describes one correlator input supplied by the user."""

    kind: str
    path: str
    file_format: str
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorrelatorSpec":
        """Validate and construct a correlator specification."""
        kind = data.get("kind")
        path = data.get("path")
        file_format = data.get("file_format")
        if kind not in SUPPORTED_CORRELATOR_KINDS:
            raise ManifestValidationError(
                f"Unsupported correlator kind: {kind!r}. Expected one of {sorted(SUPPORTED_CORRELATOR_KINDS)}."
            )
        if not isinstance(path, str) or not path.strip():
            raise ManifestValidationError("Each correlator must define a non-empty string 'path'.")
        if file_format not in SUPPORTED_DATA_FORMATS:
            raise ManifestValidationError(
                f"Unsupported correlator file_format: {file_format!r}. Expected one of {sorted(SUPPORTED_DATA_FORMATS)}."
            )
        metadata = _validate_correlator_metadata(kind, dict(data.get("metadata", {})))
        return cls(
            kind=kind,
            path=path,
            file_format=file_format,
            label=str(data.get("label", kind)),
            metadata=metadata,
        )


@dataclass(slots=True)
class KernelSpec:
    """Inline hard-kernel definition embedded in the manifest."""

    source: str
    callable_name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KernelSpec":
        """Validate and construct an inline kernel definition."""
        source = data.get("source")
        callable_name = data.get("callable_name")
        if not isinstance(source, str) or not source.strip():
            raise ManifestValidationError("Manifest kernel must define a non-empty string 'source'.")
        if not isinstance(callable_name, str) or not callable_name.strip():
            raise ManifestValidationError("Manifest kernel must define a non-empty string 'callable_name'.")
        return cls(source=source, callable_name=callable_name)


@dataclass(slots=True)
class WorkflowSpec:
    """Optional workflow overrides supplied by the user."""

    stages: list[str] = field(default_factory=list)
    stage_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowSpec":
        """Validate and construct workflow overrides."""
        stages = list(data.get("stages", []))
        if any(not isinstance(stage, str) or not stage.strip() for stage in stages):
            raise ManifestValidationError("Workflow stages must be a list of non-empty stage names.")
        stage_parameters = dict(data.get("stage_parameters", {}))
        return cls(stages=stages, stage_parameters=stage_parameters)


@dataclass(slots=True)
class OutputSpec:
    """Output preferences for a workflow run."""

    directory: str = "outputs"
    plot_formats: list[str] = field(default_factory=lambda: ["pdf"])
    data_formats: list[str] = field(default_factory=lambda: ["csv"])
    keep_intermediates: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputSpec":
        """Validate and construct output settings."""
        directory = str(data.get("directory", "outputs"))
        plot_formats = list(data.get("plot_formats", ["pdf"]))
        data_formats = list(data.get("data_formats", ["csv"]))
        keep_intermediates = bool(data.get("keep_intermediates", True))
        if any(fmt not in SUPPORTED_PLOT_FORMATS for fmt in plot_formats):
            raise ManifestValidationError(
                f"Unsupported plot format in {plot_formats!r}. Expected one of {sorted(SUPPORTED_PLOT_FORMATS)}."
            )
        if any(fmt not in SUPPORTED_EXPORT_FORMATS for fmt in data_formats):
            raise ManifestValidationError(
                f"Unsupported data export format in {data_formats!r}. Expected one of {sorted(SUPPORTED_EXPORT_FORMATS)}."
            )
        return cls(
            directory=directory,
            plot_formats=plot_formats,
            data_formats=data_formats,
            keep_intermediates=keep_intermediates,
        )


@dataclass(slots=True)
class Manifest:
    """Top-level manifest consumed by the rule-based workflow engine."""

    goal: str
    correlators: list[CorrelatorSpec]
    metadata: dict[str, Any]
    kernel: KernelSpec
    workflow: WorkflowSpec = field(default_factory=WorkflowSpec)
    outputs: OutputSpec = field(default_factory=OutputSpec)
    manifest_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], manifest_path: Path | None = None) -> "Manifest":
        """Validate raw manifest data and create the canonical model."""
        goal = data.get("goal")
        if goal not in SUPPORTED_GOALS:
            raise ManifestValidationError(
                f"Unsupported goal: {goal!r}. Expected one of {sorted(SUPPORTED_GOALS)}."
            )
        raw_correlators: list[dict[str, Any]] = []
        for item in data.get("correlators", []):
            raw_correlators.extend(_expand_correlator_entry(dict(item)))
        correlators = [CorrelatorSpec.from_dict(item) for item in raw_correlators]
        if not correlators:
            raise ManifestValidationError("Manifest must define at least one correlator input.")
        metadata = _validate_manifest_metadata(dict(data.get("metadata", {})), goal)
        kernel = KernelSpec.from_dict(data.get("kernel", {}))
        workflow = WorkflowSpec.from_dict(data.get("workflow", {}))
        outputs = OutputSpec.from_dict(data.get("outputs", {}))
        known_setup_ids = set(metadata["setups"])
        for correlator in correlators:
            setup_id = str(correlator.metadata["setup_id"])
            if setup_id not in known_setup_ids:
                raise ManifestValidationError(
                    f"Correlator {correlator.label!r} references unknown metadata.setup_id {setup_id!r}."
                )
        if goal == "custom" and not workflow.stages:
            raise ManifestValidationError("Goal 'custom' requires workflow.stages to be explicitly provided.")
        return cls(
            goal=goal,
            correlators=correlators,
            metadata=metadata,
            kernel=kernel,
            workflow=workflow,
            outputs=outputs,
            manifest_path=manifest_path,
        )

    @property
    def resolved_output_directory(self) -> Path:
        """Return the output root directory resolved from the manifest location."""
        if self.manifest_path is None:
            return Path(self.outputs.directory).resolve()
        return resolve_manifest_relative_path(self.manifest_path, self.outputs.directory)

    @property
    def analysis_metadata(self) -> dict[str, Any]:
        """Return the normalized analysis metadata block."""
        return dict(self.metadata["analysis"])

    def setup_metadata(self, setup_id: str) -> dict[str, Any]:
        """Return normalized lattice-setup metadata for one setup id."""
        try:
            return dict(self.metadata["setups"][setup_id])
        except KeyError as exc:
            raise ManifestValidationError(f"Unknown setup_id {setup_id!r}.") from exc

    def observable_name_for_b(self, b_value: int | float) -> str:
        """Return the effective observable name after accounting for nonzero transverse separation."""
        channel = str(self.analysis_metadata["channel"])
        if channel == "qpdf":
            return "qtmdpdf" if int(b_value) != 0 else "qpdf"
        return "qtmdwf" if int(b_value) != 0 else "qda"


def load_manifest(manifest_path: str | Path) -> Manifest:
    """Load and validate a workflow manifest from disk."""
    path = Path(manifest_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    manifest = Manifest.from_dict(raw, manifest_path=path)
    for correlator in manifest.correlators:
        resolved = resolve_manifest_relative_path(path, correlator.path)
        if not resolved.exists():
            raise ManifestValidationError(
                f"Correlator file does not exist: {resolved}."
            )
    return manifest
