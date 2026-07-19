"""Core helpers for interactive planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import ValidationError

from lamet_agent.core.tools import validate_stage_inputs
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.manifest_params import merge_stage_params


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
    attrs: dict[str, str] = field(default_factory=dict)
    datasets: list[H5DatasetSummary] = field(default_factory=list)
    error: str | None = None


@dataclass
class CorrelatorH5Mapping:
    """Mapping from a source dataset into the standard correlator HDF5 layout."""

    correlator_id: str
    source_file: str
    output_file: str
    datasets: list[dict[str, Any]]
    attrs: dict[str, Any] = field(default_factory=dict)
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
        kernel_parameters = item.get("kernel_parameters")
        if isinstance(kernel_parameters, dict) and "zs_fm" in kernel_parameters:
            issues.append(
                PlanIssue(
                    "error",
                    f"inputs.kernels[{index}].kernel_parameters.zs_fm",
                    "Matching zs_fm must be a flat perturbative_matching stage parameter.",
                    "Move it to stages.perturbative_matching.defaults.zs_fm or the relevant jobs[].params.zs_fm.",
                )
            )
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
        nested = renorm["defaults"].get("scheme_parameters")
        if isinstance(nested, dict) and "zs_fm" in nested:
            issues.append(
                PlanIssue(
                    "error",
                    "stages.renormalization.defaults.scheme_parameters.zs_fm",
                    "Renormalization zs_fm must be a flat stage parameter.",
                    "Move it to stages.renormalization.defaults.zs_fm.",
                )
            )
        jobs = renorm.get("jobs", [])
        for index, job in enumerate(jobs if isinstance(jobs, list) else []):
            params = job.get("params") if isinstance(job, dict) else None
            nested = params.get("scheme_parameters") if isinstance(params, dict) else None
            if isinstance(nested, dict) and "zs_fm" in nested:
                issues.append(
                    PlanIssue(
                        "error",
                        f"stages.renormalization.jobs[{index}].params.scheme_parameters.zs_fm",
                        "Renormalization zs_fm must be a flat job parameter.",
                        f"Move it to stages.renormalization.jobs[{index}].params.zs_fm.",
                    )
                )
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

    strict_payload = copy.deepcopy(payload)
    for kernel in strict_payload.get("inputs", {}).get("kernels", []) if isinstance(strict_payload.get("inputs"), dict) else []:
        if isinstance(kernel, dict) and kernel.get("stage") == "matching":
            kernel["stage"] = "perturbative_matching"
    try:
        strict = AnalysisManifest.model_validate(strict_payload)
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


def _set_kernel_stage_aliases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    inputs = payload.get("inputs")
    kernels = inputs.get("kernels") if isinstance(inputs, dict) else None
    if not isinstance(kernels, list):
        return edits
    for index, kernel in enumerate(kernels):
        if isinstance(kernel, dict) and kernel.get("stage") == "matching":
            kernel["stage"] = "perturbative_matching"
            edits.append({"path": f"inputs.kernels[{index}].stage", "old": "matching", "new": "perturbative_matching"})
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
            if stage_name == "correlator_analysis":
                defaults["model_average"] = False
            for key in ("pt2_windows", "pt3_tau_cuts", "nstate", "fit_scope", "fit_strategy", "prior_width", "order"):
                if key in defaults:
                    defaults[key] = _shrink_list(defaults[key])
            scheme_scan = defaults.get("scheme_scan")
            if isinstance(scheme_scan, dict):
                scheme_scan["model_average"] = False
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
                if stage_name == "correlator_analysis":
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
    edits = _set_kernel_stage_aliases(base)
    edits.extend(_set_kernel_scheme_from_renorm(base))
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
    matching_kernels = [
        item
        for item in kernels
        if isinstance(item, dict) and item.get("stage") in {"matching", "perturbative_matching"} and item.get("kernel_id")
    ]
    matching_kernel_ids = [item.get("kernel_id") for item in matching_kernels]
    renorm_kernels = [
        item
        for item in kernels
        if isinstance(item, dict) and item.get("stage") == "renormalization" and item.get("kernel_id")
    ]
    renorm_kernel_ids = [item.get("kernel_id") for item in renorm_kernels]
    correlators = inputs.get("correlators", []) if isinstance(inputs, dict) else []
    artifacts = {
        str(item.get("id")): item
        for item in (inputs.get("artifacts", []) if isinstance(inputs, dict) else [])
        if isinstance(item, dict) and item.get("id")
    }
    jobs_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    if isinstance(stages, dict):
        for stage_id, stage_config in stages.items():
            if not isinstance(stage_config, dict):
                continue
            for candidate in stage_config.get("jobs", []):
                if isinstance(candidate, dict) and candidate.get("id"):
                    jobs_by_id[str(candidate["id"])] = (str(stage_id), candidate)

    def has_discrete_kinematics(stage_id: str, candidate: dict[str, Any], seen: set[str]) -> bool:
        if stage_id == "correlator_analysis":
            config = stages.get(stage_id, {}) if isinstance(stages, dict) else {}
            stage_defaults = config.get("defaults", {}) if isinstance(config, dict) else {}
            params = {
                **(stage_defaults if isinstance(stage_defaults, dict) else {}),
                **(candidate.get("params") if isinstance(candidate.get("params"), dict) else {}),
            }
            momentum = (
                params.get("final_momentum")
                if str(params.get("fitting_form", "Breit")) == "NonBreit"
                else params.get("momentum")
            )
            ids = set(candidate.get("correlator_ids", []))
            return any(
                isinstance(item, dict)
                and item.get("correlator_id") in ids
                and item.get("correlator_type") == "2pt"
                and momentum in _as_list(item.get("momentum"))
                and item.get("volume") is not None
                and item.get("lattice_spacing_fm") is not None
                for item in correlators
            )
        for value in (candidate.get("inputs") or {}).values():
            for reference in _as_list(value):
                reference = str(reference)
                artifact = artifacts.get(reference)
                if artifact is not None and all(
                    artifact.get(key) is not None for key in ("momentum", "volume", "lattice_spacing_fm")
                ):
                    return True
                upstream = jobs_by_id.get(reference)
                if upstream is not None and reference not in seen:
                    if has_discrete_kinematics(upstream[0], upstream[1], seen | {reference}):
                        return True
        return False

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
            params = merge_stage_params(
                defaults,
                job.get("params") if isinstance(job.get("params"), dict) else {},
            )
            derived_momentum_available = has_discrete_kinematics(stage, job, {job_id})
            roles = set(job.get("inputs", {}).keys()) if isinstance(job.get("inputs"), dict) else set()
            def add_gap(parameter: str, path: str, message: str, suggested_fix: str) -> None:
                gaps.append({"stage": stage, "job_id": job_id, "parameter": parameter, "path": path, "message": message, "suggested_fix": suggested_fix, "question_id": f"stage_params.{stage}.{job_id}"})
            if stage == "renormalization":
                scheme = params.get("scheme")
                if scheme in {"ratio", "hybrid_ratio"}:
                    if roles != {"target", "denominator"}:
                        add_gap(
                            "inputs",
                            f"stages.{stage}.jobs[{index}].inputs",
                            f"{scheme} requires target and denominator input roles.",
                            'Example: {"target": "ca_pz", "denominator": "ca_p0"}.',
                        )
                    if scheme == "hybrid_ratio" and "zs_fm" not in params:
                        add_gap(
                            "zs_fm",
                            f"stages.{stage}.defaults.zs_fm",
                            "hybrid_ratio requires flat parameter zs_fm.",
                            'Example: {"zs_fm": 0.2}.',
                        )
                elif scheme == "hybrid_self_renormalization":
                    scheme_parameters = params.get("scheme_parameters", {})
                    if not isinstance(scheme_parameters, dict):
                        scheme_parameters = {}
                    if "LambdaQCD_gev" not in scheme_parameters:
                        add_gap(
                            "LambdaQCD_gev",
                            f"stages.{stage}.jobs[{index}].params.scheme_parameters.LambdaQCD_gev",
                            "hybrid_self_renormalization requires an explicit LambdaQCD_gev value.",
                            'Example: {"scheme_parameters": {"LambdaQCD_gev": 0.1}}.',
                        )
                    if roles == {"reference"}:
                        if "d" not in scheme_parameters:
                            add_gap(
                                "d",
                                f"stages.{stage}.jobs[{index}].params.scheme_parameters.d",
                                "hybrid_self_renormalization fit jobs require fixed parameter d.",
                                'Example: {"scheme_parameters": {"d": -0.08183}}.',
                            )
                    elif roles != {"target", "zR"}:
                        add_gap(
                            "inputs",
                            f"stages.{stage}.jobs[{index}].inputs",
                            "hybrid_self_renormalization requires either reference or target plus zR input roles.",
                            'Use {"reference": "bare_ref"} for a fit job or '
                            '{"target": "bare", "zR": "rn_fit"} for an apply job.',
                        )
                    if not renorm_kernel_ids:
                        add_gap(
                            "kernel_id",
                            "inputs.kernels",
                            "hybrid_self_renormalization requires a declared renormalization kernel.",
                            'Declare ZMSbar_pdf or ZMSbar_da with stage "renormalization".',
                        )
                    elif len(renorm_kernel_ids) > 1 and "kernel_id" not in params:
                        add_gap(
                            "kernel_id",
                            f"stages.{stage}.defaults.kernel_id",
                            f"hybrid_self_renormalization job {job_id!r} must select a renormalization kernel.",
                            "Use one declared inputs.kernels[].kernel_id.",
                        )
                else:
                    legacy_message = (
                        " scheme 'self_renormalization' was renamed to 'hybrid_self_renormalization'."
                        if scheme == "self_renormalization"
                        else ""
                    )
                    add_gap(
                        "scheme",
                        f"stages.{stage}.defaults.scheme",
                        "renormalization requires a supported scheme." + legacy_message,
                        'Choose "ratio", "hybrid_ratio", or "hybrid_self_renormalization".',
                    )
            elif stage == "fourier_transform":
                if roles != {"input"}:
                    add_gap("inputs", f"stages.{stage}.jobs[{index}].inputs", "fourier_transform requires exactly one input role named input.", 'Example: {"input": "rn_pz"}.')
                for key, example in (("order", 'Choose "LA", "NLA", or ["LA", "NLA"].'), ("coord_unit", 'Choose "lattice", "fm", "gev_inv", or "lambda".'), ("y_grid", 'Example: {"start": -2.0, "stop": 2.0, "num": 100}.'), ("momentum_gev", "Declare momentum, volume, and lattice_spacing_fm on the upstream correlator or partial-run artifact.")):
                    if key not in params and not (key == "momentum_gev" and derived_momentum_available):
                        add_gap(key, f"stages.{stage}.defaults.{key}", f"fourier_transform job {job_id!r} is missing parameter {key}.", example)
                if "sector" not in params and "part" not in params:
                    add_gap("sector", f"stages.{stage}.defaults.sector", f"fourier_transform job {job_id!r} is missing sector or part.", 'For PDF choose one of "valence", "total", "full", "sea"; alternatively set part to "re", "im", or "both".')
            elif stage == "perturbative_matching":
                if roles != {"quasi"}:
                    add_gap("inputs", f"stages.{stage}.jobs[{index}].inputs", "perturbative_matching requires exactly one input role named quasi.", 'Example: {"quasi": "ft_pz"}.')
                if "kernel_id" not in params and len(matching_kernel_ids) != 1:
                    add_gap("kernel_id", f"stages.{stage}.defaults.kernel_id", f"perturbative_matching job {job_id!r} is missing kernel_id.", "Use one declared inputs.kernels[].kernel_id.")
                for key, example in (("momentum_gev", "Declare momentum, volume, and lattice_spacing_fm on the upstream correlator or partial-run artifact."), ("mu", "Example: 2.0."), ("component", 'Choose "re" or "im".')):
                    if key not in params and not (key == "momentum_gev" and derived_momentum_available):
                        add_gap(key, f"stages.{stage}.defaults.{key}", f"perturbative_matching job {job_id!r} is missing parameter {key}.", example)
                selected_kernel_id = params.get("kernel_id") or (matching_kernel_ids[0] if len(matching_kernel_ids) == 1 else None)
                selected_kernel = next(
                    (item for item in matching_kernels if item.get("kernel_id") == selected_kernel_id),
                    None,
                )
                if (
                    isinstance(selected_kernel, dict)
                    and "hybrid" in str(selected_kernel.get("kernel_id", "")).lower()
                    and "zs_fm" not in params
                ):
                    add_gap("zs_fm", f"stages.{stage}.defaults.zs_fm", f"hybrid matching job {job_id!r} is missing flat parameter zs_fm.", 'Example: {"zs_fm": 0.2}.')
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
