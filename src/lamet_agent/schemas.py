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
        metadata = {key: _format_with_mapping(value, mapping) for key, value in dict(expanded.get("metadata", {})).items()}
        for key, value in mapping.items():
            metadata.setdefault(key, value)
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
        return cls(
            kind=kind,
            path=path,
            file_format=file_format,
            label=str(data.get("label", kind)),
            metadata=dict(data.get("metadata", {})),
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
        metadata = dict(data.get("metadata", {}))
        for required_key in ("ensemble", "conventions"):
            if required_key not in metadata:
                raise ManifestValidationError(
                    f"Manifest metadata must define '{required_key}' so workflow reports stay interpretable."
                )
        kernel = KernelSpec.from_dict(data.get("kernel", {}))
        workflow = WorkflowSpec.from_dict(data.get("workflow", {}))
        outputs = OutputSpec.from_dict(data.get("outputs", {}))
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
