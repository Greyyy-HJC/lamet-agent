"""Interactive manifest planning and correlator HDF5 preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
import re
from typing import Any, Callable, Literal

from pydantic import ValidationError

from lamet_agent.core.banner import BANNER
from lamet_agent.core.llm import request_llm_text
from lamet_agent.core.tools import validate_stage_inputs
from lamet_agent.manifest import AnalysisManifest, validate_manifest_file


IssueSeverity = Literal["error", "warning", "info"]


@dataclass
class PlanIssue:
    """One deterministic issue found while planning a manifest."""

    severity: IssueSeverity
    manifest_path: str
    message: str
    suggested_fix: str | None = None


@dataclass
class H5DatasetSummary:
    """Small HDF5 dataset summary safe to send to an LLM."""

    path: str
    shape: list[int]
    dtype: str
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class H5Inspection:
    """Summary of one inspected HDF5 file."""

    correlator_id: str
    path: str
    exists: bool
    datasets: list[H5DatasetSummary] = field(default_factory=list)
    error: str | None = None


@dataclass
class CorrelatorH5Mapping:
    """Mapping from a source dataset into the standard correlator HDF5 layout."""

    correlator_id: str
    source_file: str
    output_file: str
    datasets: list[dict[str, Any]]
    script_file: str | None = None
    ambiguous: bool = False
    reason: str | None = None


@dataclass
class PlanProposal:
    """A concrete proposal shown by the interactive plan command."""

    report: str
    manifest_edits: list[dict[str, Any]]
    quick_manifest_path: str
    full_manifest_path: str
    data_conversions: list[CorrelatorH5Mapping]
    unresolved_questions: list[str] = field(default_factory=list)


@dataclass
class PlanRunResult:
    """Result of an accepted planning session."""

    quick_manifest_path: str
    full_manifest_path: str
    data_files: list[str]
    issues: list[PlanIssue]
    quick_manifest_changes: list[str] = field(default_factory=list)
    full_manifest_changes: list[str] = field(default_factory=list)


@dataclass
class PlanAgentState:
    """Mutable in-memory state owned by the interactive planning agent."""

    manifest_path: Path
    manifest_text: str
    original_payload: dict[str, Any]
    candidate_payload: dict[str, Any]
    manifest_edits: list[dict[str, Any]] = field(default_factory=list)
    conversions: list[CorrelatorH5Mapping] = field(default_factory=list)
    issues: list[PlanIssue] = field(default_factory=list)
    inspections: list[H5Inspection] = field(default_factory=list)
    quick: dict[str, Any] | None = None
    full: dict[str, Any] | None = None
    quick_path: Path | None = None
    full_path: Path | None = None
    suppressed_full_expansions: set[str] = field(default_factory=set)
    stage_completion_checked: bool = False
    stage_completion_requested: bool = False
    parameter_completion_checked: bool = False
    parameter_completion_requested: bool = False


def _strip_jsonc(text: str) -> str:
    """Remove JSONC comments and trailing commas."""
    out: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and nxt == "/":
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and nxt == "*":
            index += 2
            while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                index += 1
            index += 2
            continue
        out.append(char)
        index += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def load_relaxed_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Load a JSON/JSONC manifest as a plain dict without schema validation."""
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Manifest does not exist: {path}")
    text = manifest_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(_strip_jsonc(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not parseable JSON/JSONC: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Manifest top-level value must be a JSON object.")
    return payload, text


def _manifest_root(manifest_path: Path, payload: dict[str, Any]) -> Path | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    root_value = metadata.get("root_directory")
    if not isinstance(root_value, str) or not root_value.strip():
        return None
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = manifest_path.expanduser().resolve().parent / root
    return root.resolve()


def _artifacts_dir(manifest_path: Path, payload: dict[str, Any]) -> Path:
    root = _manifest_root(manifest_path, payload) or manifest_path.expanduser().resolve().parent
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    raw = metadata.get("artifacts_directory", "artifacts") if isinstance(metadata, dict) else "artifacts"
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _planned_manifest_paths(manifest_path: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    """Return quick/full manifest paths under the manifest artifacts directory."""
    output_dir = _artifacts_dir(manifest_path, payload) / "plan_manifests"
    return output_dir / f"{manifest_path.stem}.quick.json", output_dir / f"{manifest_path.stem}.full.json"


def _resolve_manifest_path(manifest_path: Path, payload: dict[str, Any], value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = _manifest_root(manifest_path, payload)
    if root is None:
        return None
    return (root / path).resolve()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _id_counts(values: list[str]) -> dict[str, int]:
    return {value: values.count(value) for value in set(values)}


def check_manifest_draft(manifest_path: Path, payload: dict[str, Any]) -> list[PlanIssue]:
    """Run deterministic checks that tolerate incomplete manifests."""
    issues: list[PlanIssue] = []
    for block in ("metadata", "inputs", "stages"):
        if block not in payload:
            issues.append(PlanIssue("error", block, f"Missing top-level block `{block}`.", f"Add `{block}`."))

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        stages = {}

    required_metadata = ("run_id", "root_directory", "target_observable", "parton", "resample_mode", "random_seed", "stages")
    for key in required_metadata:
        if key not in metadata:
            issues.append(PlanIssue("error", f"metadata.{key}", f"Missing required metadata field `{key}`."))

    root = _manifest_root(manifest_path, payload)
    if root is None:
        issues.append(PlanIssue("error", "metadata.root_directory", "Cannot resolve metadata.root_directory."))
    elif not root.exists():
        issues.append(
            PlanIssue(
                "error",
                "metadata.root_directory",
                f"Root directory does not exist: {root}",
                "Set metadata.root_directory to an existing directory.",
            )
        )

    stage_order = metadata.get("stages")
    stage_order_list = [item for item in stage_order if isinstance(item, str)] if isinstance(stage_order, list) else []
    if "stages" in metadata and not isinstance(stage_order, list):
        issues.append(PlanIssue("error", "metadata.stages", "`metadata.stages` must be a list."))
    duplicates = sorted(key for key, count in _id_counts(stage_order_list).items() if count > 1)
    if duplicates:
        issues.append(PlanIssue("error", "metadata.stages", f"Duplicate stage ids: {duplicates}."))
    for stage in stage_order_list:
        if stage not in stages:
            issues.append(PlanIssue("error", f"stages.{stage}", f"`metadata.stages` includes `{stage}` but no stage config exists."))

    correlators = inputs.get("correlators", [])
    if not isinstance(correlators, list):
        issues.append(PlanIssue("error", "inputs.correlators", "`inputs.correlators` must be a list."))
        correlators = []
    artifacts = inputs.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
    kernels = inputs.get("kernels", [])
    if not isinstance(kernels, list):
        issues.append(PlanIssue("error", "inputs.kernels", "`inputs.kernels` must be a list."))
        kernels = []

    correlator_ids = [str(item.get("correlator_id")) for item in correlators if isinstance(item, dict) and item.get("correlator_id")]
    artifact_ids = [str(item.get("id")) for item in artifacts if isinstance(item, dict) and item.get("id")]
    job_ids: list[str] = []
    for stage_name, config in stages.items():
        if not isinstance(config, dict):
            issues.append(PlanIssue("error", f"stages.{stage_name}", "Stage config must be an object."))
            continue
        jobs = config.get("jobs", [])
        if not isinstance(jobs, list):
            issues.append(PlanIssue("error", f"stages.{stage_name}.jobs", "Stage jobs must be a list."))
            continue
        for index, job in enumerate(jobs):
            if isinstance(job, dict) and job.get("id"):
                job_ids.append(str(job["id"]))
            else:
                issues.append(PlanIssue("error", f"stages.{stage_name}.jobs[{index}].id", "Stage job is missing `id`."))

    all_ids = correlator_ids + artifact_ids + job_ids
    duplicate_ids = sorted(key for key, count in _id_counts(all_ids).items() if count > 1)
    if duplicate_ids:
        issues.append(PlanIssue("error", "ids", f"Manifest ids must be globally unique: {duplicate_ids}."))

    for index, item in enumerate(correlators):
        if not isinstance(item, dict):
            continue
        data_path = _resolve_manifest_path(manifest_path, payload, item.get("data_path"))
        display_path = f"inputs.correlators[{index}].data_path"
        if data_path is None:
            issues.append(PlanIssue("error", display_path, "Cannot resolve correlator data_path."))
        elif not data_path.exists():
            issues.append(PlanIssue("error", display_path, f"Correlator data file does not exist: {data_path}"))

    for index, item in enumerate(kernels):
        if not isinstance(item, dict):
            continue
        kernel_path = _resolve_manifest_path(manifest_path, payload, item.get("kernel_path"))
        display_path = f"inputs.kernels[{index}].kernel_path"
        if kernel_path is None:
            issues.append(PlanIssue("warning", display_path, "Cannot resolve kernel_path."))
        elif not kernel_path.exists():
            issues.append(PlanIssue("error", display_path, f"Kernel file does not exist: {kernel_path}"))

    known = set(artifact_ids)
    correlator_id_set = set(correlator_ids)
    for stage in stage_order_list:
        config = stages.get(stage)
        if not isinstance(config, dict):
            continue
        for job_index, job in enumerate(config.get("jobs", []) if isinstance(config.get("jobs"), list) else []):
            if not isinstance(job, dict):
                continue
            job_path = f"stages.{stage}.jobs[{job_index}]"
            unknown_correlators = sorted(set(str(item) for item in _as_list(job.get("correlator_ids"))) - correlator_id_set)
            if unknown_correlators:
                issues.append(PlanIssue("error", f"{job_path}.correlator_ids", f"Unknown correlator ids: {unknown_correlators}."))
            inputs_map = job.get("inputs", {})
            if isinstance(inputs_map, dict):
                for role, refs in inputs_map.items():
                    unknown = [str(ref) for ref in _as_list(refs) if str(ref) not in known]
                    if unknown:
                        issues.append(PlanIssue("error", f"{job_path}.inputs.{role}", f"Unavailable upstream ids: {unknown}."))
            if job.get("id"):
                known.add(str(job["id"]))

    renorm = stages.get("renormalization")
    renorm_scheme = None
    if isinstance(renorm, dict) and isinstance(renorm.get("defaults"), dict):
        renorm_scheme = renorm["defaults"].get("scheme")
    if isinstance(renorm_scheme, str):
        for index, kernel in enumerate(kernels):
            if isinstance(kernel, dict) and isinstance(kernel.get("scheme"), str) and kernel["scheme"] != renorm_scheme:
                issues.append(
                    PlanIssue(
                        "warning",
                        f"inputs.kernels[{index}].scheme",
                        f"Kernel scheme `{kernel['scheme']}` differs from renormalization scheme `{renorm_scheme}`.",
                        f"Set inputs.kernels[{index}].scheme to `{renorm_scheme}` unless this is intentional.",
                    )
                )

    try:
        strict = AnalysisManifest.model_validate(copy.deepcopy(payload))
    except (ValidationError, ValueError) as exc:
        issues.append(PlanIssue("info", "manifest", f"Strict manifest validation is not yet clean: {exc}"))
    else:
        for stage in stage_order_list:
            if stage not in strict.stages:
                continue
            for job in strict.stages[stage].jobs:
                try:
                    for message in validate_stage_inputs(stage, strict, job):
                        issues.append(PlanIssue("warning", f"stages.{stage}.jobs.{job.id}", message))
                except Exception as exc:
                    issues.append(PlanIssue("warning", f"stages.{stage}.jobs.{job.id}", f"Stage input check failed: {exc}"))

    for gap in _stage_parameter_gaps(payload):
        issues.append(PlanIssue("warning", str(gap["path"]), str(gap["message"]), str(gap["suggested_fix"])))

    return issues


def _dataset_attrs(obj: Any) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, value in obj.attrs.items():
        if len(attrs) >= 8:
            break
        attrs[str(key)] = str(value)
    return attrs


def inspect_correlator_h5_files(manifest_path: Path, payload: dict[str, Any]) -> list[H5Inspection]:
    """Inspect HDF5/NumPy files referenced by inputs.correlators only."""
    try:
        import h5py
    except ImportError:
        h5py = None

    inspections: list[H5Inspection] = []
    correlators = payload.get("inputs", {}).get("correlators", [])
    if not isinstance(correlators, list):
        return inspections
    for item in correlators:
        if not isinstance(item, dict):
            continue
        resolved = _resolve_manifest_path(manifest_path, payload, item.get("data_path"))
        correlator_id = str(item.get("correlator_id", ""))
        if resolved is None or not resolved.exists():
            inspections.append(H5Inspection(correlator_id=correlator_id, path=str(item.get("data_path", "")), exists=False))
            continue
        datasets: list[H5DatasetSummary] = []
        suffix = resolved.suffix.lower()
        try:
            if suffix in {".h5", ".hdf5"}:
                if h5py is None:
                    raise ValueError("h5py is not installed; install the analysis extra to inspect correlator HDF5 files.")
                with h5py.File(resolved, "r") as h5f:
                    def visit(name: str, obj: Any) -> None:
                        if isinstance(obj, h5py.Dataset):
                            datasets.append(
                                H5DatasetSummary(
                                    path=name,
                                    shape=[int(dim) for dim in obj.shape],
                                    dtype=str(obj.dtype),
                                    attrs=_dataset_attrs(obj),
                                )
                            )

                    h5f.visititems(visit)
            elif suffix == ".npy":
                import numpy as np

                arr = np.load(resolved, mmap_mode="r")
                datasets.append(H5DatasetSummary(path="array", shape=[int(dim) for dim in arr.shape], dtype=str(arr.dtype)))
            elif suffix == ".npz":
                import numpy as np

                with np.load(resolved) as npz:
                    for key in sorted(npz.files):
                        arr = npz[key]
                        datasets.append(H5DatasetSummary(path=str(key), shape=[int(dim) for dim in arr.shape], dtype=str(arr.dtype)))
            else:
                raise ValueError("Unsupported correlator data format; convert to .npy, .npz, or preferably standard .h5.")
        except Exception as exc:
            inspections.append(H5Inspection(correlator_id=correlator_id, path=str(resolved), exists=True, error=str(exc)))
            continue
        inspections.append(H5Inspection(correlator_id=correlator_id, path=str(resolved), exists=True, datasets=datasets))
    return inspections


def _standard_dataset_paths(correlator: dict[str, Any]) -> list[str]:
    kind = correlator.get("kind")
    source_sink = str(correlator.get("source_sink", ""))
    momentum = str(correlator.get("momentum", ""))
    if kind == "2pt":
        return [f"{source_sink}/{correlator.get('src_gamma', '')}/{momentum}"]
    if kind == "3pt":
        gamma = str(correlator.get("current_gamma", ""))
        direction = str(correlator.get("z_direction", ""))
        eta = str(correlator.get("eta", ""))
        bt_values = _as_list(correlator.get("bt"))
        bt = bt_values[0] if bt_values else None
        if bt is None:
            return []
        return [
            f"{source_sink}/{gamma}/{momentum}/b_{direction}/{eta}/bT{bt}/bz{z}"
            for z in _as_list(correlator.get("bz"))
        ]
    return []


def _dataset_names(path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as h5f:
            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    out[name] = [int(dim) for dim in obj.shape]

            h5f.visititems(visit)
    elif suffix == ".npy":
        import numpy as np

        arr = np.load(path, mmap_mode="r")
        out["array"] = [int(dim) for dim in arr.shape]
    elif suffix == ".npz":
        import numpy as np

        with np.load(path) as npz:
            for key in sorted(npz.files):
                out[str(key)] = [int(dim) for dim in npz[key].shape]
    return out


def _choose_source_datasets(correlator: dict[str, Any], source: Path) -> tuple[list[dict[str, Any]], bool, str | None]:
    names = _dataset_names(source)
    targets = _standard_dataset_paths(correlator)
    if not targets:
        return [], True, "Cannot build standard target paths from correlator metadata."
    if all(target in names for target in targets):
        return [], False, None
    available = sorted(names)
    if correlator.get("kind") == "2pt":
        if len(available) != 1:
            return [], True, f"Expected one 2pt dataset or the standard path; found {len(available)} datasets."
        return [{"source": available[0], "target": targets[0], "transpose": False}], True, (
            "Non-standard HDF5 2pt input requires explicit confirmation of time/cfg axes and transpose. "
            f"Available dataset: {_source_summary(names)}. Expected standard target: {targets[0]}."
        )
    if correlator.get("kind") == "3pt":
        bz_values = _as_list(correlator.get("bz"))
        if len(available) != len(targets):
            return [], True, f"Expected {len(targets)} 3pt datasets for bz values {bz_values}; found {len(available)}."
        return [], True, (
            "Non-standard HDF5 3pt input requires explicit source-to-bz mapping and tau/cfg axis confirmation. "
            f"Available datasets: {_source_summary(names)}. Expected standard targets: {targets}."
        )
    return [], True, f"Unsupported correlator kind {correlator.get('kind')!r}."


def _source_summary(names: dict[str, list[int]]) -> str:
    return ", ".join(f"{key}: shape={shape}" for key, shape in sorted(names.items()))


def plan_correlator_h5_conversions(manifest_path: Path, payload: dict[str, Any]) -> list[CorrelatorH5Mapping]:
    """Return required non-ambiguous and ambiguous correlator data conversions."""
    data_dir = _artifacts_dir(manifest_path, payload) / "plan_data"
    conversions: list[CorrelatorH5Mapping] = []
    correlators = payload.get("inputs", {}).get("correlators", [])
    if not isinstance(correlators, list):
        return conversions
    for item in correlators:
        if not isinstance(item, dict):
            continue
        source = _resolve_manifest_path(manifest_path, payload, item.get("data_path"))
        if source is None or not source.exists():
            continue
        suffix = source.suffix.lower()
        correlator_id = str(item.get("correlator_id", source.stem))
        output = data_dir / f"{correlator_id}.h5"
        script = data_dir / f"convert_{correlator_id}.py"
        if suffix not in {".h5", ".hdf5", ".npy", ".npz"}:
            conversions.append(
                CorrelatorH5Mapping(
                    correlator_id=correlator_id,
                    source_file=str(source),
                    output_file=str(output),
                    script_file=str(script),
                    datasets=[],
                    ambiguous=True,
                    reason="Unsupported correlator data format. Convert the data to .npy, .npz, or preferably standard .h5.",
                )
            )
            continue
        try:
            names = _dataset_names(source)
            targets = _standard_dataset_paths(item)
            if suffix in {".h5", ".hdf5"}:
                datasets, ambiguous, reason = _choose_source_datasets(item, source)
            elif suffix == ".npy":
                if len(targets) == 1 and names.get("array"):
                    if item.get("kind") == "2pt":
                        datasets, ambiguous, reason = [{"source": "array", "target": targets[0], "transpose": False}], True, (
                            "NumPy 2pt input requires explicit confirmation of cfg and time axes, momentum selection, and whether transpose is needed. "
                            f"Available array: {_source_summary(names)}. Expected standard target: {targets[0]}."
                        )
                    else:
                        datasets, ambiguous, reason = [{"source": "array", "target": targets[0], "transpose": False}], True, (
                            "NumPy 3pt input requires explicit confirmation of cfg, tau, z/bz ordering, momentum selection, and whether transpose is needed. "
                            f"Available array: {_source_summary(names)}. Expected standard targets: {targets}."
                        )
                else:
                    datasets, ambiguous, reason = [], True, (
                        "NumPy input cannot be mapped without an explicit source/target mapping. "
                        f"Available array: {_source_summary(names)}. Expected standard targets: {targets}."
                    )
            else:
                available = set(names)
                if targets and all(target in available for target in targets):
                    datasets, ambiguous, reason = [{"source": target, "target": target, "transpose": False} for target in targets], False, None
                elif len(targets) == 1 and len(names) == 1:
                    source_name = next(iter(names))
                    datasets, ambiguous, reason = [{"source": source_name, "target": targets[0], "transpose": False}], True, (
                        "NPZ input has one candidate array, but cfg/time or cfg/tau axes and momentum selection must be confirmed explicitly. "
                        f"Available arrays: {_source_summary(names)}. Expected standard target: {targets[0]}."
                    )
                else:
                    datasets, ambiguous, reason = [], True, (
                        "NPZ input requires explicit key-to-target and axis mapping. "
                        f"Available arrays: {_source_summary(names)}. Expected standard targets: {targets}."
                    )
        except Exception as exc:
            datasets, ambiguous, reason = [], True, str(exc)
        if not datasets and not ambiguous:
            continue
        conversions.append(
            CorrelatorH5Mapping(
                correlator_id=correlator_id,
                source_file=str(source),
                output_file=str(output),
                script_file=str(script),
                datasets=datasets,
                ambiguous=ambiguous,
                reason=reason,
            )
        )
    return conversions


def _copy_h5_attrs(source_obj: Any, target_obj: Any) -> None:
    for key, value in source_obj.attrs.items():
        target_obj.attrs[key] = value


def convert_correlator_h5(mapping: CorrelatorH5Mapping) -> None:
    """Write one converted standard correlator HDF5 file."""
    if mapping.ambiguous:
        raise ValueError(f"Cannot convert ambiguous mapping for {mapping.correlator_id}: {mapping.reason}")
    import h5py
    import numpy as np

    output = Path(mapping.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    if mapping.script_file:
        script = Path(mapping.script_file)
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            "from lamet_agent.planning import CorrelatorH5Mapping, convert_correlator_h5\n\n"
            "mapping = CorrelatorH5Mapping(**"
            + repr(_dataclass_json(mapping))
            + ")\n"
            "convert_correlator_h5(mapping)\n",
            encoding="utf-8",
        )
    suffix = Path(mapping.source_file).suffix.lower()
    npy_data = np.load(mapping.source_file, mmap_mode="r") if suffix == ".npy" else None
    npz_data = np.load(mapping.source_file) if suffix == ".npz" else None
    h5_source = h5py.File(mapping.source_file, "r") if suffix in {".h5", ".hdf5"} else None
    try:
        with h5py.File(output, "w") as dst:
            for item in mapping.datasets:
                source_name = str(item.get("source") or "array")
                target_name = str(item["target"])
                if h5_source is not None:
                    data = np.asarray(h5_source[source_name])
                elif npz_data is not None:
                    data = np.asarray(npz_data[source_name])
                else:
                    data = np.asarray(npy_data)
                original_shape = list(data.shape)
                fixed_axes = {int(axis) for axis in (item.get("index") or {})}
                for axis, index in sorted((item.get("index") or {}).items(), key=lambda pair: int(pair[0]), reverse=True):
                    data = np.take(data, int(index), axis=int(axis))
                if item.get("axis_order") is not None:
                    axes = [int(axis) for axis in item["axis_order"]]
                    if sorted(axes) == list(range(1, data.ndim + 1)):
                        axes = [axis - 1 for axis in axes]
                    elif axes and max(axes) >= data.ndim:
                        remaining_axes = [axis for axis in range(len(original_shape)) if axis not in fixed_axes]
                        axes = [remaining_axes.index(axis) for axis in axes if axis in remaining_axes]
                    data = np.transpose(data, axes)
                if item.get("transpose"):
                    data = data.T
                dataset = dst.create_dataset(target_name, data=data)
                if h5_source is not None:
                    _copy_h5_attrs(h5_source[source_name], dataset)
                dataset.attrs["lamet_agent_original_file"] = mapping.source_file
                dataset.attrs["lamet_agent_original_dataset"] = source_name
                dataset.attrs["lamet_agent_original_shape"] = json.dumps(original_shape)
                dataset.attrs["lamet_agent_transposed"] = bool(item.get("transpose"))
                if item.get("axis_order") is not None:
                    dataset.attrs["lamet_agent_axis_order"] = json.dumps(item["axis_order"])
                if item.get("index"):
                    dataset.attrs["lamet_agent_index"] = json.dumps(item["index"])
    finally:
        if h5_source is not None:
            h5_source.close()
        if npz_data is not None:
            npz_data.close()


def _set_kernel_scheme_from_renorm(payload: dict[str, Any]) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    stages = payload.get("stages")
    inputs = payload.get("inputs")
    if not isinstance(stages, dict) or not isinstance(inputs, dict):
        return edits
    renorm = stages.get("renormalization")
    if not isinstance(renorm, dict) or not isinstance(renorm.get("defaults"), dict):
        return edits
    scheme = renorm["defaults"].get("scheme")
    if not isinstance(scheme, str):
        return edits
    kernels = inputs.get("kernels")
    if not isinstance(kernels, list):
        return edits
    for index, kernel in enumerate(kernels):
        if isinstance(kernel, dict) and kernel.get("scheme") != scheme:
            old = kernel.get("scheme")
            kernel["scheme"] = scheme
            edits.append({"path": f"inputs.kernels[{index}].scheme", "old": old, "new": scheme})
    return edits


def _set_path_value(payload: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    parts = path.split(".")
    target = payload
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    old = target.get(parts[-1])
    target[parts[-1]] = value
    return {"path": path, "old": old, "new": value}


def _get_path_value(payload: dict[str, Any], path: str) -> Any:
    target: Any = payload
    for part in path.split("."):
        if not isinstance(target, dict) or part not in target:
            return None
        target = target[part]
    return copy.deepcopy(target)


def _normalise_pt2_windows(value: Any) -> list[dict[str, int]]:
    windows: list[dict[str, int]] = []
    if not isinstance(value, list):
        return windows
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            tmin = int(item["tmin"])
            tmax = int(item["tmax"])
        except (KeyError, TypeError, ValueError):
            continue
        windows.append({"tmin": tmin, "tmax": tmax})
    return windows


def _expand_pt2_windows(value: Any) -> list[dict[str, int]]:
    windows = _normalise_pt2_windows(value)
    if not windows:
        return [{"tmin": 3, "tmax": 12}, {"tmin": 4, "tmax": 12}, {"tmin": 5, "tmax": 12}]
    tmax_values = [item["tmax"] for item in windows]
    default_tmax = max(tmax_values)
    tmins = sorted({item["tmin"] for item in windows})
    candidates = [max(1, min(tmins) - 1), *tmins, max(tmins) + 1, max(tmins) + 2]
    by_key = {(item["tmin"], item["tmax"]): dict(item) for item in windows}
    for tmin in candidates:
        if tmin < default_tmax:
            by_key.setdefault((tmin, default_tmax), {"tmin": tmin, "tmax": default_tmax})
    return [by_key[key] for key in sorted(by_key)]


def _expand_tau_cuts(value: Any) -> list[int]:
    cuts = sorted({int(item) for item in value if isinstance(item, int) or str(item).isdigit()}) if isinstance(value, list) else []
    if not cuts:
        return [2, 3, 4]
    candidates = {cut for cut in cuts if cut > 0}
    if not candidates:
        return [2, 3, 4]
    max_cut = max(candidates)
    candidates.add(max_cut + 1)
    candidates.add(max_cut + 2)
    return sorted(candidates)


def _remove_edit_for_path(edits: list[dict[str, Any]], path: str) -> None:
    edits[:] = [item for item in edits if item.get("path") != path]


def _merge_revision_edits(existing: list[dict[str, Any]], new_edits: list[dict[str, Any]]) -> None:
    for edit in new_edits:
        path = edit.get("path")
        if isinstance(path, str):
            _remove_edit_for_path(existing, path)
        existing.append(edit)


def _update_conversion_paths(payload: dict[str, Any], conversions: list[CorrelatorH5Mapping], manifest_path: Path) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    root = _manifest_root(manifest_path, payload)
    correlators = payload.get("inputs", {}).get("correlators", [])
    if root is None or not isinstance(correlators, list):
        return edits
    by_id = {item.correlator_id: item for item in conversions if not item.ambiguous and item.datasets}
    for index, correlator in enumerate(correlators):
        if not isinstance(correlator, dict):
            continue
        mapping = by_id.get(str(correlator.get("correlator_id")))
        if mapping is None:
            continue
        output = Path(mapping.output_file)
        try:
            new_path = str(output.relative_to(root))
        except ValueError:
            new_path = str(output)
        old = correlator.get("data_path")
        correlator["data_path"] = new_path
        edits.append({"path": f"inputs.correlators[{index}].data_path", "old": old, "new": new_path})
    return edits


def _shrink_list(value: Any) -> Any:
    return value[:1] if isinstance(value, list) and value else value


def _expand_nstate(value: Any) -> Any:
    values = [int(item) for item in _as_list(value) if isinstance(item, int) or str(item).isdigit()]
    if not values:
        return [2, 3]
    expanded = set(values)
    expanded.add(max(values) + 1)
    return sorted(expanded)


def _expand_prior_width(value: Any) -> list[float]:
    values: list[float] = []
    for item in _as_list(value):
        try:
            width = float(item)
        except (TypeError, ValueError):
            continue
        if width > 0:
            values.append(width)
    if not values:
        return [0.5, 1.0, 2.0]
    expanded = set(values)
    expanded.add(min(values) * 0.5)
    expanded.add(max(values) * 2.0)
    return sorted(expanded)


def _expand_correlator_search_params(
    params: dict[str, Any],
    *,
    fill_missing: bool,
    base_path: str,
    suppressed_paths: set[str],
) -> None:
    params["model_average"] = True
    if (fill_missing or "pt2_windows" in params) and f"{base_path}.pt2_windows" not in suppressed_paths:
        params["pt2_windows"] = _expand_pt2_windows(params.get("pt2_windows"))
    if (fill_missing or "pt3_tau_cuts" in params) and f"{base_path}.pt3_tau_cuts" not in suppressed_paths:
        params["pt3_tau_cuts"] = _expand_tau_cuts(params.get("pt3_tau_cuts"))
    if (fill_missing or "nstate" in params) and f"{base_path}.nstate" not in suppressed_paths:
        params["nstate"] = _expand_nstate(params.get("nstate"))
    if (fill_missing or "prior_width" in params) and f"{base_path}.prior_width" not in suppressed_paths:
        params["prior_width"] = _expand_prior_width(params.get("prior_width"))


def _make_quick_variant(payload: dict[str, Any]) -> dict[str, Any]:
    quick = copy.deepcopy(payload)
    metadata = quick.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["resample_mode"] = "jk"
        metadata["sample_error_mode"] = "mean"
    stages = quick.get("stages")
    if not isinstance(stages, dict):
        return quick
    for stage_name, config in stages.items():
        if not isinstance(config, dict):
            continue
        defaults = config.get("defaults")
        if isinstance(defaults, dict):
            defaults["model_average"] = False
            for key in ("pt2_windows", "pt3_tau_cuts", "nstate", "fit_scope", "fit_strategy", "prior_width", "order"):
                if key in defaults:
                    defaults[key] = _shrink_list(defaults[key])
            scheme_scan = defaults.get("scheme_scan")
            if isinstance(scheme_scan, dict):
                for key in ("zmin_values", "zmax_values", "order", "posterior_prior_error_scale"):
                    if key in scheme_scan:
                        scheme_scan[key] = _shrink_list(scheme_scan[key])
                if isinstance(scheme_scan.get("max_schemes"), int):
                    scheme_scan["max_schemes"] = min(int(scheme_scan["max_schemes"]), 8)
        jobs = config.get("jobs")
        if isinstance(jobs, list):
            for job in jobs:
                if not isinstance(job, dict) or not isinstance(job.get("params"), dict):
                    continue
                job["params"]["model_average"] = False
                for key in ("pt2_windows", "pt3_tau_cuts", "nstate", "fit_scope", "fit_strategy", "prior_width", "order"):
                    if key in job["params"]:
                        job["params"][key] = _shrink_list(job["params"][key])
    return quick


def _make_full_variant(payload: dict[str, Any], *, suppressed_paths: set[str] | None = None) -> dict[str, Any]:
    suppressed_paths = suppressed_paths or set()
    full = copy.deepcopy(payload)
    metadata = full.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["sample_error_mode"] = "covariance"
    stages = full.get("stages")
    if not isinstance(stages, dict):
        return full
    correlator = stages.get("correlator_analysis")
    if isinstance(correlator, dict):
        defaults = correlator.setdefault("defaults", {})
        if isinstance(defaults, dict):
            _expand_correlator_search_params(
                defaults,
                fill_missing=True,
                base_path="stages.correlator_analysis.defaults",
                suppressed_paths=suppressed_paths,
            )
        jobs = correlator.get("jobs")
        if isinstance(jobs, list):
            for index, job in enumerate(jobs):
                if isinstance(job, dict) and isinstance(job.get("params"), dict):
                    _expand_correlator_search_params(
                        job["params"],
                        fill_missing=False,
                        base_path=f"stages.correlator_analysis.jobs[{index}].params",
                        suppressed_paths=suppressed_paths,
                    )
    return full


def build_repaired_manifests(
    manifest_path: Path,
    payload: dict[str, Any],
    conversions: list[CorrelatorH5Mapping],
    *,
    suppressed_full_expansions: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Return quick/full manifests plus deterministic edits."""
    base = copy.deepcopy(payload)
    edits = _set_kernel_scheme_from_renorm(base)
    edits.extend(_update_conversion_paths(base, conversions, manifest_path))
    quick = _make_quick_variant(base)
    full = _make_full_variant(base, suppressed_paths=suppressed_full_expansions)
    return quick, full, edits


def _dataclass_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_dataclass_json(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: _dataclass_json(getattr(value, key)) for key in value.__dataclass_fields__}
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match is None:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _json_pointer_parts(path: str) -> list[str]:
    if not isinstance(path, str) or not path:
        raise ValueError(f"JSON Patch path must be a non-empty string: {path!r}")
    if path.startswith("/"):
        parts = [part.replace("~1", "/").replace("~0", "~") for part in path.split("/")[1:]]
    else:
        parts = path.split(".")
    if not parts or parts[0] not in {"metadata", "inputs", "stages"}:
        raise ValueError("JSON Patch may only modify /metadata, /inputs, or /stages.")
    return parts


def _json_pointer_display(parts: list[str]) -> str:
    return ".".join(parts)


def _resolve_patch_parent(payload: dict[str, Any], parts: list[str]) -> tuple[Any, str]:
    target: Any = payload
    for part in parts[:-1]:
        if isinstance(target, dict):
            if part not in target:
                raise ValueError(f"JSON Patch parent does not exist: {_json_pointer_display(parts[:-1])}")
            target = target[part]
        elif isinstance(target, list):
            if not part.isdigit():
                raise ValueError(f"JSON Patch list index must be an integer: {part!r}")
            index = int(part)
            if index < 0 or index >= len(target):
                raise ValueError(f"JSON Patch list index out of range: {index}")
            target = target[index]
        else:
            raise ValueError(f"JSON Patch parent is not a container: {_json_pointer_display(parts[:-1])}")
    return target, parts[-1]


def _apply_one_json_patch(payload: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    op = patch.get("op")
    path = patch.get("path")
    if op not in {"add", "replace", "remove"}:
        raise ValueError(f"Unsupported JSON Patch op: {op!r}")
    parts = _json_pointer_parts(str(path))
    parent, key = _resolve_patch_parent(payload, parts)
    display = _json_pointer_display(parts)
    if isinstance(parent, dict):
        exists = key in parent
        if op == "add":
            old = copy.deepcopy(parent.get(key))
            parent[key] = copy.deepcopy(patch.get("value"))
            return {"path": display, "old": old, "new": copy.deepcopy(parent[key]), "note": patch.get("note", "LLM plan edit")}
        if op == "replace":
            if not exists:
                raise ValueError(f"Cannot replace missing JSON object key: {display}")
            old = copy.deepcopy(parent[key])
            parent[key] = copy.deepcopy(patch.get("value"))
            return {"path": display, "old": old, "new": copy.deepcopy(parent[key]), "note": patch.get("note", "LLM plan edit")}
        if not exists:
            raise ValueError(f"Cannot remove missing JSON object key: {display}")
        old = copy.deepcopy(parent[key])
        del parent[key]
        return {"path": display, "old": old, "new": None, "note": patch.get("note", "LLM plan edit")}

    if isinstance(parent, list):
        if key == "-":
            if op != "add":
                raise ValueError("JSON Patch '-' list target is only valid for add.")
            old = None
            parent.append(copy.deepcopy(patch.get("value")))
            return {"path": display, "old": old, "new": copy.deepcopy(parent[-1]), "note": patch.get("note", "LLM plan edit")}
        if not key.isdigit():
            raise ValueError(f"JSON Patch list index must be an integer: {key!r}")
        index = int(key)
        if op == "add":
            if index < 0 or index > len(parent):
                raise ValueError(f"JSON Patch list add index out of range: {index}")
            parent.insert(index, copy.deepcopy(patch.get("value")))
            return {"path": display, "old": None, "new": copy.deepcopy(parent[index]), "note": patch.get("note", "LLM plan edit")}
        if index < 0 or index >= len(parent):
            raise ValueError(f"JSON Patch list index out of range: {index}")
        if op == "replace":
            old = copy.deepcopy(parent[index])
            parent[index] = copy.deepcopy(patch.get("value"))
            return {"path": display, "old": old, "new": copy.deepcopy(parent[index]), "note": patch.get("note", "LLM plan edit")}
        old = copy.deepcopy(parent[index])
        del parent[index]
        return {"path": display, "old": old, "new": None, "note": patch.get("note", "LLM plan edit")}

    raise ValueError(f"JSON Patch target parent is not a container: {display}")


def apply_manifest_json_patches(payload: dict[str, Any], patches: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply a guarded JSON Patch subset to a manifest copy."""
    candidate = copy.deepcopy(payload)
    edits: list[dict[str, Any]] = []
    for patch in patches:
        if not isinstance(patch, dict):
            raise ValueError("Each JSON Patch item must be an object.")
        edits.append(_apply_one_json_patch(candidate, patch))
    return candidate, edits


def _strict_manifest_issues(payload: dict[str, Any]) -> list[PlanIssue]:
    issues: list[PlanIssue] = []
    try:
        strict = AnalysisManifest.model_validate(copy.deepcopy(payload))
    except (ValidationError, ValueError) as exc:
        return [PlanIssue("error", "manifest", f"Strict manifest validation failed: {exc}")]
    for stage in strict.metadata.stages:
        if stage not in strict.stages:
            continue
        for job in strict.stages[stage].jobs:
            try:
                for message in validate_stage_inputs(stage, strict, job):
                    issues.append(PlanIssue("error", f"stages.{stage}.jobs.{job.id}", message))
            except Exception as exc:
                issues.append(PlanIssue("error", f"stages.{stage}.jobs.{job.id}", f"Stage input check failed: {exc}"))
    return issues


def _stage_parameter_gaps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = payload.get("metadata", {})
    stages = payload.get("stages", {})
    inputs = payload.get("inputs", {})
    order = metadata.get("stages", []) if isinstance(metadata, dict) else []
    stage_order = [stage for stage in order if isinstance(stage, str)] if isinstance(order, list) else []
    kernels = inputs.get("kernels", []) if isinstance(inputs, dict) else []
    matching_kernel_ids = [item.get("kernel_id") for item in kernels if isinstance(item, dict) and item.get("stage") == "matching" and item.get("kernel_id")]
    gaps: list[dict[str, Any]] = []
    if not isinstance(stages, dict):
        return gaps
    for stage in stage_order:
        config = stages.get(stage)
        if not isinstance(config, dict):
            continue
        defaults = config.get("defaults", {})
        defaults = defaults if isinstance(defaults, dict) else {}
        jobs = config.get("jobs", [])
        if not isinstance(jobs, list):
            continue
        for index, job in enumerate(jobs):
            if not isinstance(job, dict):
                continue
            job_id = str(job.get("id", index))
            params = {**defaults, **(job.get("params") if isinstance(job.get("params"), dict) else {})}
            roles = set(job.get("inputs", {}).keys()) if isinstance(job.get("inputs"), dict) else set()
            def add_gap(parameter: str, path: str, message: str, suggested_fix: str) -> None:
                gaps.append({"stage": stage, "job_id": job_id, "parameter": parameter, "path": path, "message": message, "suggested_fix": suggested_fix, "question_id": f"stage_params.{stage}.{job_id}"})
            if stage == "renormalization":
                if roles != {"target", "denominator"}:
                    add_gap("inputs", f"stages.{stage}.jobs[{index}].inputs", "renormalization requires target and denominator input roles.", 'Example: {"target": "ca_pz", "denominator": "ca_p0"}.')
                if params.get("scheme") != "hybrid_ratio":
                    add_gap("scheme", f"stages.{stage}.defaults.scheme", "renormalization requires scheme='hybrid_ratio'.", 'Set scheme to "hybrid_ratio".')
                scheme_parameters = params.get("scheme_parameters")
                if not isinstance(scheme_parameters, dict) or "zs_fm" not in scheme_parameters:
                    add_gap("scheme_parameters.zs_fm", f"stages.{stage}.defaults.scheme_parameters.zs_fm", "hybrid_ratio requires scheme_parameters.zs_fm.", 'Example: {"scheme_parameters": {"zs_fm": 0.2}}.')
            elif stage == "fourier_transform":
                if roles != {"input"}:
                    add_gap("inputs", f"stages.{stage}.jobs[{index}].inputs", "fourier_transform requires exactly one input role named input.", 'Example: {"input": "rn_pz"}.')
                for key, example in (("order", 'Choose "LA", "NLA", or ["LA", "NLA"].'), ("coord_unit", 'Choose "lattice", "fm", "gev_inv", or "lambda".'), ("y_grid", 'Example: {"start": -2.0, "stop": 2.0, "num": 100}.'), ("pz_gev", "Example: 2.15.")):
                    if key not in params:
                        add_gap(key, f"stages.{stage}.defaults.{key}", f"fourier_transform job {job_id!r} is missing parameter {key}.", example)
                if "sector" not in params and "part" not in params:
                    add_gap("sector", f"stages.{stage}.defaults.sector", f"fourier_transform job {job_id!r} is missing sector or part.", 'For PDF choose one of "valence", "total", "full", "sea"; alternatively set part to "re", "im", or "both".')
            elif stage == "perturbative_matching":
                if roles != {"quasi"}:
                    add_gap("inputs", f"stages.{stage}.jobs[{index}].inputs", "perturbative_matching requires exactly one input role named quasi.", 'Example: {"quasi": "ft_pz"}.')
                if "kernel_id" not in params and len(matching_kernel_ids) != 1:
                    add_gap("kernel_id", f"stages.{stage}.defaults.kernel_id", f"perturbative_matching job {job_id!r} is missing kernel_id.", "Use one declared inputs.kernels[].kernel_id.")
                for key, example in (("pz_gev", "Example: 2.15."), ("mu", "Example: 2.0."), ("component", 'Choose "re" or "im".')):
                    if key not in params:
                        add_gap(key, f"stages.{stage}.defaults.{key}", f"perturbative_matching job {job_id!r} is missing parameter {key}.", example)
            elif stage == "extrapolation":
                if "momenta" not in roles:
                    add_gap("inputs.momenta", f"stages.{stage}.jobs[{index}].inputs", "extrapolation requires a momenta input role, but the stage is currently a placeholder.", 'Example: {"momenta": ["mt_pz1", "mt_pz2"]}.')
    return gaps


def validate_candidate_payload(manifest_path: Path, payload: dict[str, Any]) -> tuple[bool, list[PlanIssue]]:
    """Validate a candidate manifest payload before it can become writable state."""
    issues = check_manifest_draft(manifest_path, payload)
    issues.extend(_strict_manifest_issues(payload))
    blocking = [issue for issue in issues if issue.severity == "error"]
    return not blocking, issues


PLAN_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["call_tool", "request_user_input", "propose_plan", "finish"]},
        "reason": {"type": "string"},
        "tool_name": {"type": "string"},
        "args": {"type": "object"},
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}


PLAN_TOOL_CATALOG = {
    "load_manifest": "Return the current in-memory manifest candidate and planned output paths.",
    "check_manifest_draft": "Run deterministic manifest checks that tolerate incomplete drafts.",
    "list_stage_parameter_gaps": "Return structured missing parameters and missing input roles for configured stages, including allowed options or example values.",
    "inspect_correlator_h5_files": "Summarize HDF5, NPY, and NPZ datasets/arrays referenced by inputs.correlators.",
    "plan_correlator_h5_conversions": "Detect source HDF5/NPY/NPZ files that need conversion to the standard correlator HDF5 layout.",
    "apply_correlator_conversion_mapping": "Apply one user-confirmed correlator conversion. Args must be {correlator_id, datasets:[{source,target,axis_order?,index?,transpose?}, ...]}; never pass source/target at top level.",
    "validate_candidate_manifest": "Run strict schema, DAG, and stage-local validation on the current candidate.",
    "apply_manifest_patch_to_candidate": "Apply guarded JSON Patch edits to the in-memory candidate after validation.",
    "build_quick_full_candidates": "Build quick/full manifest candidates and validate their strict schema.",
}


def _planning_system_prompt() -> str:
    return (
        "You are the planning controller for a Python LaMET workflow agent. "
        "You control the plan loop by choosing one action at a time. "
        "Use planning tools to inspect state, validate assumptions, apply candidate manifest patches, "
        "ask the user for missing intent, and then propose a plan. "
        "Your first action must be call_tool load_manifest, and you must call check_manifest_draft before asking user input. "
        "Never claim a manifest edit was applied until a tool observation confirms it. "
        "Do not write files; final writes happen only after the user accepts. "
        "Return exactly one JSON object matching this schema: "
        + json.dumps(PLAN_ACTION_SCHEMA)
        + "\nAvailable planning tools:\n"
        + "\n".join(f"- {name}: {description}" for name, description in PLAN_TOOL_CATALOG.items())
        + "\nJSON Patch rules: edits may only target /metadata, /inputs, or /stages; use op add, replace, or remove. "
        "For request_user_input, args.prompt must be a concrete user-facing question and args.question_id must identify the decision. "
        "Ask exactly one question per request_user_input action; never combine stage choices, metadata values, parameter values, and data-axis mappings in one prompt. "
        "For ordinary manifest scalar fields, ask for exactly one manifest field at a time and set question_id to the exact dotted manifest path, for example metadata.random_seed, inputs.correlators.0.momentum, inputs.correlators.0.src_gamma, or inputs.correlators.1.current_gamma. "
        "Do not ask for several ordinary manifest fields in one answer; after the user answers one field, let the automatic patch observation update the candidate before asking the next field. "
        "When multiple items are missing, ask and resolve them one at a time, starting with deterministic manifest fields such as metadata.random_seed before broader workflow choices. "
        "Prefer Yes/No or multiple-choice questions only when the answer is genuinely binary or enumerable; use free-form questions when the user may need to name a subset, axis meaning, or concrete parameter values. "
        "Keep request_user_input prompts concise: state the file shape, uncertain axes or indices, and exact answer format only. "
        "If metadata.stages is not the full canonical flow, ask whether the user wants to add extra downstream stages. "
        "Use question_id 'stage.add_remaining'. This question may be free-form when the user may want only a subset, for example: 'only add renormalization and fourier_transform'. "
        "Add only stages whose inputs can be wired unambiguously; otherwise ask another concise question. "
        "If a configured stage has missing parameters or missing required input roles, explain which stage/job is incomplete and ask a Yes/No question before patching. "
        "Use list_stage_parameter_gaps for structured missing-parameter details when available. "
        "For that question, use question_id 'stage_params.<stage>.<job_id>' when possible. "
        "If the user says yes, ask for a concrete value when it is not inferable; list allowed options for enum-like fields and give one valid example for list/dict fields. "
        "If the value is inferable from existing manifest examples or upstream metadata, patch it only after the Yes answer and state the exact manifest path/value. "
        "For missing required fields, prefer request_user_input unless the user's instruction or examples clearly establish the value. "
        "For stage additions, preserve existing ids and wire jobs through existing upstream job ids. "
        "For correlator data conversion, inspect HDF5/NPY/NPZ inputs and never guess ambiguous axes or source keys. "
        "Use multiple-choice questions for simple axis/index choices, but use free-form questions for high-dimensional mappings where the user must describe source, target, cfg/time or cfg/tau axes, z/bz ordering, momentum selection, optional axis_order, optional index selections, and transpose. "
        "When the user gives an unambiguous mapping, call apply_correlator_conversion_mapping with args.correlator_id and args.datasets as a non-empty list of dataset mappings. "
        "Each dataset mapping must include source and target, with optional axis_order, index, and transpose. "
        "axis_order is zero-based and may name either the remaining array axes after index selection, such as [0,1], or the original source axes, such as [3,4] after fixing axes 0,1,2. "
        "Do not use one-based axis_order such as [1,2] for a two-dimensional post-index dataset. "
        "Do not pass source, target, axis_order, index, or transpose directly in top-level args. "
        "For multi-bz 3pt data, include one datasets item per standard bz target in a single tool call when practical. "
        "Do not create a custom conversion section in the manifest and do not encode conversion mappings as JSON Patch manifest edits. "
        "Only use JSON Patch for ordinary manifest fields such as metadata, inputs, and stages. "
        "If a correlator data file is not .npy, .npz, .h5, or .hdf5, tell the user the format is unsupported and strongly recommend converting to standard .h5."
    )


def _initial_planning_user_prompt(manifest_path: Path, manifest_text: str) -> str:
    return json.dumps(
        {
            "task": "Prepare this LaMET analysis manifest for execution.",
            "manifest_path": str(manifest_path),
            "manifest_text": manifest_text,
            "stage_ids": ["correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"],
            "stage_completion_policy": (
                "The canonical full flow is correlator_analysis -> renormalization -> fourier_transform -> perturbative_matching -> extrapolation -> review. "
                "If the manifest contains only a prefix or subset, ask whether to add extra stages before proposing a plan; allow a free-form subset such as only renormalization and fourier_transform."
            ),
            "stage_parameter_guidance": {
                "correlator_analysis": {
                    "common_defaults": {
                        "pt2_windows": [{"tmin": 2, "tmax": 12}, {"tmin": 3, "tmax": 12}],
                        "pt3_tau_cuts": [2, 3],
                        "nstate": [2],
                        "fit_scope": ["ratio"],
                        "fit_strategy": ["joint"],
                        "fitting_form": "Breit",
                    },
                    "options": {
                        "fit_scope": ["ratio", "FH", "ratio+FH"],
                        "fit_strategy": ["joint", "chained"],
                        "fitting_form": ["Breit", "NonBreit"],
                        "component": ["re", "im", "both"],
                    },
                },
                "renormalization": {
                    "required": {"scheme": "hybrid_ratio", "scheme_parameters": {"zs_fm": 0.2}},
                    "optional": {"normalization": True, "scheme_parameters": {"m0_gev": 0.0, "delta_m_gev": 0.0}},
                },
                "fourier_transform": {
                    "required": {"order": ["LA", "NLA"], "coord_unit": "lattice", "sector": "valence", "y_grid": {"start": -2.0, "stop": 2.0, "num": 100}, "pz_gev": 2.15},
                    "options": {"order": ["LA", "NLA"], "sector_pdf": ["valence", "total", "full", "sea"], "part": ["re", "im", "both"]},
                },
                "perturbative_matching": {
                    "required": {"kernel_id": "declared inputs.kernels kernel_id", "pz_gev": 2.15, "mu": 2.0, "component": "re"},
                    "options": {"component": ["re", "im"]},
                },
                "extrapolation": {"status": "placeholder; requires momenta input role but is not implemented yet"},
                "review": {"required": "none"},
            },
            "common_stage_contracts": {
                "renormalization": {
                    "inputs": {"target": "upstream bare matrix-element job", "denominator": "zero-momentum/reference bare matrix-element job"},
                    "defaults": {"scheme": "hybrid_ratio", "scheme_parameters": {"zs_fm": "required"}},
                },
                "fourier_transform": {"inputs": {"input": "renormalized matrix-element job or artifact"}},
                "perturbative_matching": {"inputs": {"quasi": "Fourier transform job or artifact"}},
            },
            "correlator_conversion_contract": {
                "standard_2pt_h5": "source_sink/src_gamma/momentum with dataset shape (Lt, n_cfg)",
                "standard_3pt_h5": "source_sink/current_gamma/momentum/b_<z_direction>/<eta>/bT<bt>/bz<bz> with dataset shape (tsep+1, n_cfg)",
                "npy_source": "single array; source may be 'array'; user must identify cfg/time or cfg/tau axes and any selected momentum/z indices",
                "npz_source": "source must be an NPZ key; user must map each key to one standard target",
                "apply_conversion_tool_args": {
                    "correlator_id": "correlator_id from inputs.correlators",
                    "datasets": [
                        {
                            "source": "HDF5 dataset path, NPZ key, or 'array' for NPY",
                            "target": "one standard target path",
                            "axis_order": "optional zero-based list producing standard (time_or_tau, cfg) order; may use remaining axes after index selection or original source axes",
                            "index": "optional object mapping source axis number to selected index before axis_order/transpose",
                            "transpose": "optional final transpose boolean",
                        }
                    ],
                },
                "mapping_item": {
                    "source": "HDF5 dataset path, NPZ key, or 'array' for NPY",
                    "target": "one standard target path",
                    "axis_order": "optional zero-based list producing standard (time_or_tau, cfg) order; may use remaining axes after index selection or original source axes",
                    "index": "optional object mapping source axis number to selected index before transpose",
                    "transpose": "optional final transpose boolean",
                },
            },
        },
        indent=2,
    )


class _PlanAgentSession:
    def __init__(
        self,
        *,
        backend: str,
        manifest_path: Path,
        manifest_text: str,
        api_key: str | None,
        provider: str | None,
        model_name: str | None,
        base_url: str | None,
    ) -> None:
        self.backend = backend
        self.api_key = api_key
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": _planning_system_prompt()},
            {"role": "user", "content": _initial_planning_user_prompt(manifest_path, manifest_text)},
        ]
        self.mock_phase = "load"
        self.last_revision: str | None = None

    def observe(self, observation: dict[str, Any]) -> None:
        self.messages.append({"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False, indent=2)})
        if observation.get("event") == "user_revision":
            self.last_revision = str(observation.get("text", ""))
            self.mock_phase = "mock_revision"
        elif observation.get("event") == "user_answer":
            if observation.get("question_id") == "stage.add_remaining":
                self.mock_phase = "build"
            elif str(observation.get("question_id", "")).startswith("stage_params."):
                value = str(observation.get("value", "")).strip().lower()
                self.mock_phase = "blocked" if value in {"no", "n", "false", "0"} else "build"
            else:
                self.mock_phase = "mock_answer"
        elif observation.get("event") == "question_skipped":
            self.mock_phase = "conversions"
        elif "not the full canonical stage flow" in str(observation.get("error", "")):
            self.mock_phase = "stage_completion"
        elif "still have missing parameters or input roles" in str(observation.get("error", "")):
            self.mock_phase = "blocked"
        elif "missing parameters or input roles" in str(observation.get("error", "")):
            self.mock_phase = "parameter_completion"

    def decide(self) -> dict[str, Any]:
        if self.backend == "mock":
            return self._mock_decide()
        text = request_llm_text(
            backend=self.backend,
            messages=self.messages,
            api_key=self.api_key,
            provider=self.provider,
            model_name=self.model_name,
            base_url=self.base_url,
        )
        action = _parse_json_object(text)
        self.messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        return action

    def _mock_decide(self) -> dict[str, Any]:
        phase = self.mock_phase
        if phase == "load":
            self.mock_phase = "check"
            return {"action": "call_tool", "tool_name": "load_manifest", "args": {}, "reason": "Inspect the draft manifest."}
        if phase == "check":
            self.mock_phase = "maybe_seed"
            return {"action": "call_tool", "tool_name": "check_manifest_draft", "args": {}, "reason": "Find deterministic manifest issues."}
        if phase == "maybe_seed":
            self.mock_phase = "conversions"
            return {
                "action": "request_user_input",
                "reason": "metadata.random_seed is required when absent.",
                "args": {
                    "question_id": "metadata.random_seed",
                    "prompt": "metadata.random_seed is required. Which seed should be used?",
                    "choices": [
                        {"label": "1", "value": 1984, "description": "Use 1984, matching the repository examples."},
                        {"label": "2", "value": 20260707, "description": "Use a date-based seed for this planning run."},
                        {"label": "3", "value": "__custom_int__", "description": "Enter a custom positive integer seed."},
                    ],
                    "custom_hint": "Enter random_seed as an integer: ",
                    "skip_if_present": "metadata.random_seed",
                },
            }
        if phase == "mock_answer":
            self.mock_phase = "conversions"
            value = self._latest_user_answer()
            return {
                "action": "call_tool",
                "tool_name": "apply_manifest_patch_to_candidate",
                "args": {"patches": [{"op": "add", "path": "/metadata/random_seed", "value": int(value)}]},
                "reason": "Apply the user-selected random seed.",
            }
        if phase == "conversions":
            self.mock_phase = "inspect"
            return {"action": "call_tool", "tool_name": "plan_correlator_h5_conversions", "args": {}, "reason": "Plan any correlator data conversions."}
        if phase == "inspect":
            self.mock_phase = "build"
            return {"action": "call_tool", "tool_name": "inspect_correlator_h5_files", "args": {}, "reason": "Inspect correlator data inputs."}
        if phase == "build":
            self.mock_phase = "propose"
            return {"action": "call_tool", "tool_name": "build_quick_full_candidates", "args": {}, "reason": "Build quick and full manifest candidates."}
        if phase == "stage_completion":
            return {
                "action": "request_user_input",
                "reason": "The manifest is not the full canonical stage flow.",
                "args": {
                    "question_id": "stage.add_remaining",
                    "prompt": "This manifest is not a full canonical flow. Add extra downstream stages?",
                    "choices": [
                        {"label": "1", "value": "yes", "description": "Yes, add downstream stages when inputs can be wired unambiguously."},
                        {"label": "2", "value": "no", "description": "No, keep the manifest as a partial workflow."},
                    ],
                },
            }
        if phase == "parameter_completion":
            return {
                "action": "request_user_input",
                "reason": "A configured stage is missing required parameters or input roles.",
                "args": {
                    "question_id": "stage_params.missing",
                    "prompt": "A configured stage is missing required parameters. Add them before building manifests?",
                    "choices": [
                        {"label": "1", "value": "yes", "description": "Yes, add the missing parameters using explicit values or examples."},
                        {"label": "2", "value": "no", "description": "No, keep the manifest unchanged."},
                    ],
                },
            }
        if phase == "mock_revision":
            self.mock_phase = "build"
            note = self.last_revision or ""
            text = note.lower()
            suppressions = []
            if ("tau" in text or "pt3_tau_cuts" in text) and (
                "改回" in note or "恢复" in note or "撤回" in note or "undo" in text or "revert" in text
            ):
                suppressions.append("stages.correlator_analysis.defaults.pt3_tau_cuts")
            return {
                "action": "call_tool",
                "tool_name": "apply_manifest_patch_to_candidate",
                "args": {
                    "patches": "__mock_revision__",
                    "revision": note,
                    "suppress_full_expansions": suppressions,
                },
                "reason": "Apply the user's revision as candidate manifest patches.",
            }
        if phase == "blocked":
            self.mock_phase = "done"
            return {
                "action": "finish",
                "reason": "Configured stages still have missing parameters or input roles. No manifest files were written.",
                "args": {"error": True},
            }
        self.mock_phase = "done"
        return {"action": "propose_plan", "reason": "Present the latest validated candidate.", "args": {"summary": "Mock planning summary."}}

    def _latest_user_answer(self) -> Any:
        for message in reversed(self.messages):
            try:
                observation = json.loads(message["content"]).get("observation", {})
            except Exception:
                continue
            if observation.get("event") == "user_answer":
                return observation.get("value")
        return 1984


def _get_dotted_path(payload: dict[str, Any], path: str) -> Any:
    target: Any = payload
    for part in path.split("."):
        if not isinstance(target, dict) or part not in target:
            return None
        target = target[part]
    return target


def _ask_plan_agent_question(args: dict[str, Any], input_func: Callable[[str], str], output_func: Callable[[str], None]) -> Any:
    output_func("")
    output_func(str(args["prompt"]))
    choices = args.get("choices")
    question_id = str(args.get("question_id") or "")
    if isinstance(choices, list) and choices:
        for index, choice in enumerate(choices, start=1):
            if isinstance(choice, dict):
                output_func(f"  {index}. {choice.get('description', choice.get('label', ''))}")
            else:
                output_func(f"  {index}. {choice}")
        output_func("  q. Quit without writing files.")
        while True:
            raw = input_func("Select an option: ").strip()
            if raw.lower() in {"q", "quit"}:
                return "quit"
            selected: dict[str, Any] | None = None
            for index, choice in enumerate(choices, start=1):
                if isinstance(choice, dict):
                    labels = {str(index), str(choice.get("label")), str(choice.get("value"))}
                    if raw.lower() in {item.lower() for item in labels}:
                        selected = choice
                        break
                elif raw.lower() in {str(index), str(choice).lower()}:
                    selected = {"value": choice}
                    break
            if selected is None:
                if question_id == "stage.add_remaining" and raw:
                    return raw
                output_func("Please choose one of the listed options.")
                continue
            value = selected.get("value")
            if value == "__custom_int__":
                while True:
                    custom = input_func(str(args.get("custom_hint") or "Enter value: ")).strip()
                    try:
                        parsed = int(custom)
                    except ValueError:
                        output_func("Please enter an integer.")
                        continue
                    if parsed <= 0:
                        output_func("Please enter a positive integer.")
                        continue
                    return parsed
            return value
    return input_func("Answer: ").strip()


def _valid_plan_agent_question(args: dict[str, Any]) -> bool:
    prompt = args.get("prompt")
    question_id = args.get("question_id")
    return isinstance(prompt, str) and bool(prompt.strip()) and isinstance(question_id, str) and bool(question_id.strip())


def _json_pointer_from_question_id(question_id: str) -> str | None:
    if question_id == "random_seed":
        question_id = "metadata.random_seed"
    question_id = re.sub(r"\[(\d+)\]", r".\1", question_id)
    parts = question_id.split(".")
    if not parts or parts[0] not in {"metadata", "inputs", "stages"}:
        return None
    escaped = [part.replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)


def _manifest_question_id_from_user_input_action(args: dict[str, Any], reason: str) -> str | None:
    raw = args.get("question_id")
    if isinstance(raw, str) and raw.strip():
        question_id = raw.strip()
        if _json_pointer_from_question_id(question_id) is not None:
            return "metadata.random_seed" if question_id == "random_seed" else question_id
    prompt = str(args.get("prompt") or "")
    text = f"{prompt}\n{reason}".lower()
    if "random_seed" in text or "random seed" in text:
        return "metadata.random_seed"
    if "bs_samples" in text or "bootstrap samples" in text:
        return "metadata.bs_samples"
    if "bin_size" in text or "bin size" in text:
        return "metadata.bin_size"
    return None


def _coerce_user_answer_for_manifest_path(question_id: str, value: Any) -> Any:
    integer_fields = {
        "metadata.random_seed",
        "metadata.bs_samples",
        "metadata.bin_size",
    }
    if question_id in integer_fields:
        return int(value)
    if question_id.endswith(".tsep"):
        return int(value)
    if question_id.endswith(".a_fm") or question_id.endswith(".pz_gev") or question_id.endswith(".pz_out_gev"):
        return float(value)
    if question_id.endswith(".bt") or question_id.endswith(".bz"):
        if isinstance(value, list):
            return [int(item) for item in value]
        text = str(value).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in re.split(r"[,，\s]+", text) if part.strip()]
        if not isinstance(parsed, list):
            parsed = [parsed]
        return [int(item) for item in parsed]
    if question_id.startswith("stages.") and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if question_id.endswith(".kernel_parameters") or question_id.endswith(".scheme_parameters"):
        if isinstance(value, dict):
            return value
        text = str(value).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"zs_fm": float(text)}
        return parsed if isinstance(parsed, dict) else {"zs_fm": float(parsed)}
    return value


def _apply_user_answer_to_candidate(state: PlanAgentState, question_id: str, value: Any) -> dict[str, Any]:
    """Apply direct answers to manifest-path questions through the same patch guardrails."""
    if question_id == "stage.add_remaining":
        state.stage_completion_checked = True
        text = str(value).strip().lower()
        negative = text in {"no", "n", "false", "0"} or "keep" in text and "partial" in text or "不" in text and ("加" in text or "添加" in text)
        state.stage_completion_requested = not negative and text in {"yes", "y", "true", "1"}
        return {"event": "user_answer_not_applied", "question_id": question_id, "value": value, "reason": "stage completion preference recorded for the planning agent."}
    if question_id.startswith("stage_params."):
        state.parameter_completion_checked = True
        state.parameter_completion_requested = str(value).strip().lower() in {"yes", "y", "true", "1"}
        return {"event": "user_answer_not_applied", "question_id": question_id, "value": value, "reason": "stage parameter completion preference recorded for the planning agent."}
    pointer = _json_pointer_from_question_id(question_id)
    if pointer is None:
        return {"event": "user_answer_not_applied", "question_id": question_id, "reason": "question_id is not a manifest path."}
    try:
        coerced = _coerce_user_answer_for_manifest_path(question_id, value)
    except (TypeError, ValueError):
        return {
            "event": "user_answer_not_applied",
            "question_id": question_id,
            "reason": f"Answer {value!r} could not be converted to the required manifest value type.",
        }
    op = "replace" if _get_dotted_path(state.candidate_payload, question_id) is not None else "add"
    observation = _run_planning_tool(
        state,
        "apply_manifest_patch_to_candidate",
        {
            "patches": [
                {
                    "op": op,
                    "path": pointer,
                    "value": coerced,
                    "note": "Applied user answer from planner question.",
                }
            ]
        },
    )
    observation["event"] = "user_answer_applied"
    observation["question_id"] = question_id
    observation["value"] = coerced
    return observation


def _mock_revision_patches(state: PlanAgentState, note: str) -> list[dict[str, Any]]:
    """Return deterministic mock patches so tests can exercise the agent patch path."""
    text = note.lower()
    payload = state.candidate_payload
    if "renormalization" in text or "重整" in note:
        stages = payload.get("stages", {})
        metadata = payload.get("metadata", {})
        order = list(metadata.get("stages", [])) if isinstance(metadata, dict) and isinstance(metadata.get("stages"), list) else []
        jobs = stages.get("correlator_analysis", {}).get("jobs", []) if isinstance(stages, dict) else []
        denominator = None
        targets: list[str] = []
        for job in jobs if isinstance(jobs, list) else []:
            if not isinstance(job, dict) or not isinstance(job.get("id"), str):
                continue
            job_id = job["id"]
            if "p0" in job_id:
                denominator = job_id
            elif re.search(r"p[1-9]", job_id):
                targets.append(job_id)
        denominator = denominator or (jobs[0]["id"] if isinstance(jobs, list) and jobs and isinstance(jobs[0], dict) else "ca")
        targets = targets or [job["id"] for job in jobs[1:] if isinstance(job, dict) and isinstance(job.get("id"), str)]
        renorm_jobs = [
            {"id": target.replace("ca_", "rn_", 1) if target.startswith("ca_") else f"rn_{target}", "inputs": {"target": target, "denominator": denominator}}
            for target in targets
        ]
        if "renormalization" not in order:
            index = order.index("correlator_analysis") + 1 if "correlator_analysis" in order else len(order)
            order.insert(index, "renormalization")
        return [
            {"op": "replace", "path": "/metadata/stages", "value": order},
            {
                "op": "add",
                "path": "/stages/renormalization",
                "value": {
                    "defaults": {
                        "normalization": False,
                        "scheme": "hybrid_ratio",
                        "scheme_parameters": {"zs_fm": 0.1722, "m0_gev": 0.0, "delta_m_gev": 0.0},
                    },
                    "jobs": renorm_jobs,
                },
            },
        ]
    if ("fit window" in text or "window" in text or "窗口" in note) and ("search" in text or "scan" in text or "多" in note or "加" in note):
        defaults = payload.get("stages", {}).get("correlator_analysis", {}).get("defaults", {})
        return [
            {
                "op": "replace",
                "path": "/stages/correlator_analysis/defaults/pt2_windows",
                "value": _expand_pt2_windows(defaults.get("pt2_windows")),
                "note": "LLM expanded the fit-window search.",
            },
            {
                "op": "replace",
                "path": "/stages/correlator_analysis/defaults/pt3_tau_cuts",
                "value": _expand_tau_cuts(defaults.get("pt3_tau_cuts")),
                "note": "LLM expanded the fit-window search.",
            },
        ]
    if ("tau" in text or "pt3_tau_cuts" in text) and ("改回" in note or "恢复" in note or "撤回" in note or "undo" in text or "revert" in text):
        original = _get_path_value(state.original_payload, "stages.correlator_analysis.defaults.pt3_tau_cuts")
        return [
            {
                "op": "replace",
                "path": "/stages/correlator_analysis/defaults/pt3_tau_cuts",
                "value": original,
                "note": "LLM reverted the tau-cut search.",
            }
        ]
    return []


def _run_planning_tool(state: PlanAgentState, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "load_manifest":
        quick_path, full_path = _planned_manifest_paths(state.manifest_path, state.candidate_payload)
        state.quick_path = quick_path
        state.full_path = full_path
        canonical_stages = ["correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"]
        metadata = state.candidate_payload.get("metadata", {})
        configured_stages = metadata.get("stages", []) if isinstance(metadata, dict) else []
        configured_stage_list = [stage for stage in configured_stages if isinstance(stage, str)] if isinstance(configured_stages, list) else []
        stage_parameter_gaps = _stage_parameter_gaps(state.candidate_payload)
        return {
            "tool_name": tool_name,
            "manifest": state.candidate_payload,
            "quick_manifest_path": str(quick_path),
            "full_manifest_path": str(full_path),
            "canonical_stage_flow": canonical_stages,
            "configured_stages": configured_stage_list,
            "missing_canonical_stages": [stage for stage in canonical_stages if stage not in configured_stage_list],
            "stage_completion_question_required": configured_stage_list != canonical_stages,
            "stage_parameter_gaps": stage_parameter_gaps,
            "stage_parameter_question_required": bool(stage_parameter_gaps),
        }
    if tool_name == "check_manifest_draft":
        state.issues = check_manifest_draft(state.manifest_path, state.candidate_payload)
        return {"tool_name": tool_name, "issues": _dataclass_json(state.issues)}
    if tool_name == "list_stage_parameter_gaps":
        return {"tool_name": tool_name, "stage_parameter_gaps": _stage_parameter_gaps(state.candidate_payload)}
    if tool_name == "inspect_correlator_h5_files":
        state.inspections = inspect_correlator_h5_files(state.manifest_path, state.candidate_payload)
        return {"tool_name": tool_name, "h5_inspections": _dataclass_json(state.inspections)}
    if tool_name == "plan_correlator_h5_conversions":
        state.conversions = plan_correlator_h5_conversions(state.manifest_path, state.candidate_payload)
        return {"tool_name": tool_name, "planned_data_conversions": _dataclass_json(state.conversions)}
    if tool_name == "apply_correlator_conversion_mapping":
        correlator_id = str(args.get("correlator_id") or "")
        datasets = args.get("datasets")
        if not correlator_id or not isinstance(datasets, list) or not datasets:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "Expected args format: {'correlator_id': '...', 'datasets': [{'source': 'array', 'target': '...', 'axis_order': [..], 'index': {'0': 0}, 'transpose': false}, ...]}. Do not pass source/target at top level.",
            }
        correlators = state.candidate_payload.get("inputs", {}).get("correlators", [])
        correlator = next((item for item in correlators if isinstance(item, dict) and str(item.get("correlator_id")) == correlator_id), None)
        mapping = next((item for item in state.conversions if item.correlator_id == correlator_id), None)
        if correlator is None or mapping is None:
            return {"tool_name": tool_name, "ok": False, "error": f"Unknown correlator conversion {correlator_id!r}."}
        source = Path(mapping.source_file)
        names = _dataset_names(source)
        targets = set(_standard_dataset_paths(correlator))
        cleaned = []
        seen_targets: set[str] = set()
        for item in datasets:
            if not isinstance(item, dict):
                return {"tool_name": tool_name, "ok": False, "error": "Each dataset mapping must be an object."}
            source_name = str(item.get("source") or "array")
            target = str(item.get("target") or "")
            if target not in targets:
                return {"tool_name": tool_name, "ok": False, "error": f"Target {target!r} is not one of the standard targets {sorted(targets)}."}
            if source_name not in names:
                return {"tool_name": tool_name, "ok": False, "error": f"Source {source_name!r} is not in {sorted(names)}."}
            if target in seen_targets:
                return {"tool_name": tool_name, "ok": False, "error": f"Target {target!r} is mapped more than once."}
            seen_targets.add(target)
            shape = list(names[source_name])
            original_shape = list(shape)
            fixed_axes = {int(axis) for axis in (item.get("index") or {})}
            for axis, index in sorted((item.get("index") or {}).items(), key=lambda pair: int(pair[0]), reverse=True):
                axis_i = int(axis)
                index_i = int(index)
                if axis_i < 0 or axis_i >= len(shape):
                    return {"tool_name": tool_name, "ok": False, "error": f"Index axis {axis_i} is out of bounds for source {source_name!r} shape {original_shape}."}
                if index_i < 0 or index_i >= shape[axis_i]:
                    return {"tool_name": tool_name, "ok": False, "error": f"Index {index_i} is out of bounds for axis {axis_i} of source {source_name!r} shape {original_shape}."}
                shape.pop(axis_i)
            if item.get("axis_order") is not None:
                axes = [int(axis) for axis in item["axis_order"]]
                if len(set(axes)) != len(axes):
                    return {"tool_name": tool_name, "ok": False, "error": f"axis_order {axes} has duplicate axes."}
                if sorted(axes) == list(range(1, len(shape) + 1)):
                    axes = [axis - 1 for axis in axes]
                elif axes and max(axes) >= len(shape):
                    remaining_axes = [axis for axis in range(len(original_shape)) if axis not in fixed_axes]
                    if any(axis not in remaining_axes for axis in axes):
                        return {"tool_name": tool_name, "ok": False, "error": f"axis_order {axes} is not compatible with index-fixed axes {sorted(fixed_axes)} and source shape {original_shape}."}
                    axes = [remaining_axes.index(axis) for axis in axes]
                if sorted(axes) != list(range(len(shape))):
                    return {"tool_name": tool_name, "ok": False, "error": f"axis_order {axes} is not a permutation of remaining axes for shape {shape}."}
                shape = [shape[axis] for axis in axes]
            if item.get("transpose"):
                shape = list(reversed(shape))
            if len(shape) != 2:
                return {"tool_name": tool_name, "ok": False, "error": f"Mapped dataset {target!r} has final shape {shape}; expected a 2D standard correlator dataset."}
            if correlator.get("kind") == "3pt" and isinstance(correlator.get("tsep"), int) and shape[0] != int(correlator["tsep"]) + 1:
                return {"tool_name": tool_name, "ok": False, "error": f"Mapped 3pt dataset {target!r} has tau length {shape[0]}; expected tsep+1={int(correlator['tsep']) + 1}."}
            out = {"source": source_name, "target": target, "transpose": bool(item.get("transpose", False))}
            if item.get("axis_order") is not None:
                out["axis_order"] = [int(axis) for axis in item["axis_order"]]
            if item.get("index") is not None:
                out["index"] = {str(axis): int(index) for axis, index in item["index"].items()}
            cleaned.append(out)
        if seen_targets != targets:
            missing = sorted(targets - seen_targets)
            extra = sorted(seen_targets - targets)
            return {"tool_name": tool_name, "ok": False, "error": f"Dataset mappings must cover every standard target exactly once. Missing={missing}; extra={extra}."}
        mapping.datasets = cleaned
        mapping.ambiguous = False
        mapping.reason = None
        state.quick = None
        state.full = None
        return {"tool_name": tool_name, "ok": True, "conversion": _dataclass_json(mapping)}
    if tool_name == "validate_candidate_manifest":
        ok, issues = validate_candidate_payload(state.manifest_path, state.candidate_payload)
        state.issues = issues
        return {"tool_name": tool_name, "ok": ok, "issues": _dataclass_json(issues)}
    if tool_name == "apply_manifest_patch_to_candidate":
        patches = args.get("patches", [])
        if patches == "__mock_revision__":
            patches = _mock_revision_patches(state, str(args.get("revision") or ""))
        if not isinstance(patches, list):
            return {"tool_name": tool_name, "ok": False, "error": "patches must be a list of JSON Patch objects."}
        try:
            candidate, edits = apply_manifest_json_patches(state.candidate_payload, patches)
        except ValueError as exc:
            return {"tool_name": tool_name, "ok": False, "error": str(exc)}
        ok, issues = validate_candidate_payload(state.manifest_path, candidate)
        correlators = candidate.get("inputs", {}).get("correlators", [])
        incomplete_correlator = isinstance(correlators, list) and any(isinstance(item, dict) and not _standard_dataset_paths(item) for item in correlators)
        incomplete_stage_params = bool(_stage_parameter_gaps(candidate))
        incomplete_kernel = any(
            issue.severity == "error" and ("kernel_id" in issue.message or "kernel_parameters" in issue.message or "zs_fm" in issue.message)
            for issue in issues
        )
        if not ok and not incomplete_correlator and not incomplete_stage_params and not incomplete_kernel and any(issue.severity == "error" and "Field required" not in issue.message for issue in issues):
            return {"tool_name": tool_name, "ok": False, "issues": _dataclass_json(issues), "edits": edits}
        state.candidate_payload = candidate
        state.manifest_edits.extend(edits)
        suppressions = args.get("suppress_full_expansions")
        if isinstance(suppressions, list):
            state.suppressed_full_expansions.update(str(item) for item in suppressions if isinstance(item, str))
        state.issues = issues
        state.quick = None
        state.full = None
        return {"tool_name": tool_name, "ok": True, "candidate_complete": ok, "edits": edits, "issues": _dataclass_json(issues)}
    if tool_name == "build_quick_full_candidates":
        canonical_stages = ["correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"]
        metadata = state.candidate_payload.get("metadata", {})
        configured_stages = metadata.get("stages", []) if isinstance(metadata, dict) else []
        configured_stage_list = [stage for stage in configured_stages if isinstance(stage, str)] if isinstance(configured_stages, list) else []
        original_metadata = state.original_payload.get("metadata", {})
        original_stages = original_metadata.get("stages", []) if isinstance(original_metadata, dict) else []
        original_stage_list = [stage for stage in original_stages if isinstance(stage, str)] if isinstance(original_stages, list) else []
        if configured_stage_list != canonical_stages and not state.stage_completion_checked:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "This manifest is not the full canonical stage flow. Ask the user first with question_id='stage.add_remaining' whether to add extra downstream stages; allow a free-form subset such as renormalization and fourier_transform.",
                "canonical_stage_flow": canonical_stages,
                "configured_stages": configured_stage_list,
                "missing_canonical_stages": [stage for stage in canonical_stages if stage not in configured_stage_list],
            }
        if state.stage_completion_requested and configured_stage_list == original_stage_list:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "The user answered yes to adding stages, but metadata.stages has not changed. Patch the requested stages first or ask a follow-up question.",
                "configured_stages": configured_stage_list,
                "missing_canonical_stages": [stage for stage in canonical_stages if stage not in configured_stage_list],
            }
        parameter_gaps = _stage_parameter_gaps(state.candidate_payload)
        if parameter_gaps and not state.parameter_completion_checked:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "Configured stages have missing parameters or input roles. Explain the gaps and ask a Yes/No question first with question_id like 'stage_params.<stage>.<job_id>'.",
                "stage_parameter_gaps": parameter_gaps,
            }
        if parameter_gaps:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "Configured stages still have missing parameters or input roles. Patch the missing manifest paths first, ask for concrete values, or quit without writing files.",
                "stage_parameter_gaps": parameter_gaps,
            }
        ambiguous = [item for item in state.conversions if item.ambiguous]
        if ambiguous:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "Ambiguous correlator conversions must be resolved before building quick/full manifests.",
                "ambiguous_conversions": _dataclass_json(ambiguous),
            }
        quick, full, edits = build_repaired_manifests(
            state.manifest_path,
            state.candidate_payload,
            state.conversions,
            suppressed_full_expansions=state.suppressed_full_expansions,
        )
        quick_issues = _strict_manifest_issues(quick)
        full_issues = _strict_manifest_issues(full)
        if quick_issues or full_issues:
            return {
                "tool_name": tool_name,
                "ok": False,
                "quick_issues": _dataclass_json(quick_issues),
                "full_issues": _dataclass_json(full_issues),
            }
        state.quick = quick
        state.full = full
        state.quick_path, state.full_path = _planned_manifest_paths(state.manifest_path, state.candidate_payload)
        for edit in edits:
            _merge_revision_edits(state.manifest_edits, [edit])
        return {
            "tool_name": tool_name,
            "ok": True,
            "deterministic_manifest_edits": edits,
            "quick_manifest_path": str(state.quick_path),
            "full_manifest_path": str(state.full_path),
        }
    return {"tool_name": tool_name, "ok": False, "error": f"Unknown planning tool: {tool_name}"}


def _missing_parameters(issues: list[PlanIssue]) -> list[str]:
    return [
        f"{issue.manifest_path}: {issue.message}"
        for issue in issues
        if issue.manifest_path.startswith("metadata.") and "Missing required" in issue.message
    ]


def _inconsistent_settings(issues: list[PlanIssue]) -> list[str]:
    return [
        f"{issue.manifest_path}: {issue.message}"
        for issue in issues
        if (
            "differs from" in issue.message
            or "Duplicate" in issue.message
            or "Unavailable upstream" in issue.message
            or "Unknown correlator" in issue.message
            or "does not exist" in issue.message
            or "Strict manifest validation" not in issue.message and issue.severity == "warning"
        )
    ]


def _render_bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- none"]


def _short_repr(value: Any, *, limit: int = 220) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _manifest_change_lines(before: Any, after: Any, *, prefix: str = "") -> list[str]:
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        lines: list[str] = []
        for key in sorted(set(before) | set(after)):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in before:
                lines.append(f"{child_prefix}: <missing> -> {_short_repr(after[key])}")
            elif key not in after:
                lines.append(f"{child_prefix}: {_short_repr(before[key])} -> <removed>")
            else:
                lines.extend(_manifest_change_lines(before[key], after[key], prefix=child_prefix))
        return lines
    return [f"{prefix}: {_short_repr(before)} -> {_short_repr(after)}"]


def _render_proposal(proposal: PlanProposal, issues: list[PlanIssue]) -> str:
    summary = proposal.report.strip()
    if "\n" in summary:
        summary = summary.splitlines()[0].strip()
    lines = [summary, "", "Missing parameters:"]
    lines.extend(_render_bullets(_missing_parameters(issues)))
    lines.extend(["", "Inconsistent settings:"])
    lines.extend(_render_bullets(_inconsistent_settings(issues)))
    lines.extend(["", "Suggested modifications:"])
    if proposal.manifest_edits:
        rendered = []
        for item in proposal.manifest_edits:
            note = f" ({item['note']})" if "note" in item else ""
            rendered.append(f"{item['path']}: {item.get('old')!r} -> {item.get('new')!r}{note}")
        lines.extend(_render_bullets(rendered))
    else:
        lines.append("- none")
    lines.extend(["", "Data conversions:"])
    conversions = [item for item in proposal.data_conversions if not item.ambiguous and item.datasets]
    if conversions:
        lines.extend(f"- {item.correlator_id}: {item.source_file} -> {item.output_file}" for item in conversions)
    else:
        lines.append("- none")
    lines.extend(["", f"Quick manifest: {proposal.quick_manifest_path}", f"Full manifest: {proposal.full_manifest_path}"])
    return "\n".join(lines)


def _render_written_summary(result: PlanRunResult) -> str:
    lines = [
        f"Wrote quick manifest: {result.quick_manifest_path}",
        f"Wrote full manifest: {result.full_manifest_path}",
    ]
    for path in result.data_files:
        lines.append(f"Wrote converted data: {path}")
    lines.extend(["", "Quick manifest changes:"])
    lines.extend(_render_bullets(result.quick_manifest_changes))
    lines.extend(["", "Full manifest changes:"])
    lines.extend(_render_bullets(result.full_manifest_changes))
    if result.issues:
        lines.extend(["", "Validation issues:"])
        lines.extend(f"- {issue.severity}: {issue.message}" for issue in result.issues)
    return "\n".join(lines)


def write_planned_outputs(
    source_payload: dict[str, Any],
    quick: dict[str, Any],
    full: dict[str, Any],
    conversions: list[CorrelatorH5Mapping],
    quick_path: Path,
    full_path: Path,
) -> PlanRunResult:
    """Apply conversions, write manifests, and run strict validation."""
    for conversion in conversions:
        if not conversion.ambiguous and conversion.datasets:
            convert_correlator_h5(conversion)
    quick_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    quick_path.write_text(json.dumps(quick, indent=2) + "\n", encoding="utf-8")
    full_path.write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    issues: list[PlanIssue] = []
    for label, path in (("quick", quick_path), ("full", full_path)):
        try:
            validate_manifest_file(path)
        except Exception as exc:
            issues.append(PlanIssue("error", str(path), f"Generated {label} manifest failed strict validation: {exc}"))
    return PlanRunResult(
        quick_manifest_path=str(quick_path),
        full_manifest_path=str(full_path),
        data_files=[item.output_file for item in conversions if not item.ambiguous and item.datasets],
        issues=issues,
        quick_manifest_changes=_manifest_change_lines(source_payload, quick),
        full_manifest_changes=_manifest_change_lines(source_payload, full),
    )


def run_interactive_plan(
    manifest_path: Path,
    *,
    backend: str,
    api_key: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> PlanRunResult | None:
    """Run the terminal planning loop under LLM/tool control."""
    payload, manifest_text = load_relaxed_manifest(manifest_path)
    state = PlanAgentState(
        manifest_path=manifest_path,
        manifest_text=manifest_text,
        original_payload=copy.deepcopy(payload),
        candidate_payload=copy.deepcopy(payload),
    )
    session = _PlanAgentSession(
        backend=backend,
        manifest_path=manifest_path,
        manifest_text=manifest_text,
        api_key=api_key,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
    )
    output_func(BANNER)

    for _ in range(60):
        action = session.decide()
        action_type = action.get("action")
        reason = str(action.get("reason", ""))
        args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}

        if action_type == "call_tool":
            tool_name = str(action.get("tool_name") or "")
            if not tool_name:
                session.observe({"event": "invalid_action", "action": action, "error": "call_tool requires tool_name."})
                continue
            observation = _run_planning_tool(state, tool_name, args)
            session.observe(observation)
            continue

        if action_type == "request_user_input":
            if not _valid_plan_agent_question(args):
                session.observe(
                    {
                        "event": "user_input_rejected",
                        "error": "request_user_input requires args.question_id and a concrete args.prompt. Do not ask the terminal until you can state the exact question.",
                        "action": action,
                    }
                )
                continue
            skip_path = args.get("skip_if_present")
            if isinstance(skip_path, str) and _get_dotted_path(state.candidate_payload, skip_path) is not None:
                session.observe({"event": "question_skipped", "reason": f"{skip_path} is already present."})
                continue
            answer = _ask_plan_agent_question(args, input_func, output_func)
            if answer == "quit":
                output_func("Plan cancelled; no files were written.")
                return None
            question_id = _manifest_question_id_from_user_input_action(args, reason) or str(args.get("question_id"))
            normalized_question_id = re.sub(r"\[(\d+)\]", r".\1", question_id)
            if normalized_question_id.endswith(".kernel_parameters.zs_fm") or normalized_question_id.endswith(".scheme_parameters.zs_fm"):
                question_id = normalized_question_id.rsplit(".", 1)[0]
            if _json_pointer_from_question_id(question_id) is None:
                text = f"{args.get('prompt', '')}\n{reason}"
                match = re.search(
                    r"correlator\s+['\"]([^'\"]+)['\"].*?['\"](momentum|src_gamma|sink_gamma|current_gamma|z_direction|eta|bt|bz|tsep|a_fm|pz_gev|pz_out_gev)['\"]",
                    text,
                    flags=re.I | re.S,
                )
                correlators = state.candidate_payload.get("inputs", {}).get("correlators", [])
                if match and isinstance(correlators, list):
                    for index, item in enumerate(correlators):
                        if isinstance(item, dict) and str(item.get("correlator_id")) == match.group(1):
                            question_id = f"inputs.correlators.{index}.{match.group(2)}"
                            break
                if _json_pointer_from_question_id(question_id) is None:
                    text_lower = text.lower()
                    for gap in _stage_parameter_gaps(state.candidate_payload):
                        stage = str(gap.get("stage", ""))
                        job_id = str(gap.get("job_id", ""))
                        parameter = str(gap.get("parameter", ""))
                        if stage.lower() in text_lower and job_id.lower() in text_lower and parameter.lower() in text_lower:
                            question_id = str(gap.get("path"))
                            break
                if _json_pointer_from_question_id(question_id) is None:
                    kernels = state.candidate_payload.get("inputs", {}).get("kernels", [])
                    text_lower = text.lower()
                    if isinstance(kernels, list) and len(kernels) == 1:
                        if "zs_fm" in text_lower:
                            question_id = "inputs.kernels.0.kernel_parameters"
                        elif "kernel_id" in text_lower:
                            question_id = "inputs.kernels.0.kernel_id"
            session.observe({"event": "user_answer", "question_id": question_id, "value": answer})
            applied = _apply_user_answer_to_candidate(state, question_id, answer)
            session.observe(applied)
            continue

        if action_type == "propose_plan":
            if state.quick is None or state.full is None or state.quick_path is None or state.full_path is None:
                session.observe(
                    {
                        "event": "proposal_rejected",
                        "error": "No validated quick/full manifest candidates are available. Call build_quick_full_candidates first.",
                    }
                )
                continue
            proposal = PlanProposal(
                report=str(args.get("summary") or reason or "Planning proposal is ready."),
                manifest_edits=state.manifest_edits,
                quick_manifest_path=str(state.quick_path),
                full_manifest_path=str(state.full_path),
                data_conversions=state.conversions,
            )
            output_func(_render_proposal(proposal, state.issues))
            answer = input_func("Accept these modifications and write files? [a]ccept/[r]evise/[q]uit: ").strip().lower()
            if answer in {"a", "accept", "y", "yes"}:
                result = write_planned_outputs(
                    state.original_payload,
                    state.quick,
                    state.full,
                    state.conversions,
                    state.quick_path,
                    state.full_path,
                )
                output_func(_render_written_summary(result))
                return result
            if answer in {"r", "revise"}:
                note = input_func("Revision instruction: ").strip()
                if note:
                    if "stage" in note.lower() or "阶段" in note:
                        state.stage_completion_checked = True
                    session.observe({"event": "user_revision", "text": note})
                output_func("")
                output_func("")
                continue
            if answer in {"q", "quit", "n", "no"}:
                output_func("Plan cancelled; no files were written.")
                return None
            output_func("Please enter accept, revise, or quit.")
            session.observe({"event": "proposal_response_invalid", "value": answer})
            continue

        if action_type == "finish":
            if args.get("error"):
                raise ValueError(str(reason or "Plan finished with a blocking error."))
            output_func(str(reason or "Plan finished without writing files."))
            return None

        session.observe({"event": "invalid_action", "action": action, "error": "Unknown planning action."})

    raise ValueError("Planning agent exceeded the maximum number of action steps.")
