"""Core helpers for interactive planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import ValidationError

from lamet_agent.core.tools import validate_stage_diagnostics
from lamet_agent.manifest import AnalysisManifest, physical_momentum_gev
from lamet_agent.manifest_params import (
    StageValidationContext,
    get_stage_parameter_contract,
    merge_stage_params,
    resolve_stage_params,
)


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
    operation: str = "copy"


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
    stage_required_checked: set[str] = field(default_factory=set)
    stage_optional_checked: set[str] = field(default_factory=set)
    parameter_completion_checked: bool = False
    parameter_completion_requested: bool = False
    path_repair_project_root: Path | None = None


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
    if manifest_path.suffix.lower() == ".txt":
        return draft_manifest_from_text(manifest_path, text), text
    try:
        payload = json.loads(_strip_jsonc(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not parseable JSON/JSONC: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Manifest top-level value must be a JSON object.")
    return payload, text


def draft_manifest_from_text(path: Path, text: str) -> dict[str, Any]:
    """Build a sparse plan-only manifest draft from a free-form text request."""
    raw_paths = [
        token.strip("'\"`,;:()[]{}")
        for token in re.findall(r"(?:~|/|\.{1,2}/)?[^\s'\"`]+?\.(?:h5|hdf5|npy|npz|nc)", text, flags=re.I)
    ]
    lowered = text.lower()
    root = path.parent.resolve()
    run_id = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_") or "planned_analysis"
    target_match = re.search(r"target[_\s-]*observable\s*[:=]?\s*(pdf|da|gpd)\b", text, flags=re.I)
    target_observable = target_match.group(1).lower() if target_match else "gpd" if re.search(r"\bgpd\b", lowered) else "da" if re.search(r"\bda\b", lowered) else "pdf"
    parton = "gluon" if "gluon" in lowered else "quark"
    correlators: list[dict[str, Any]] = []
    current_sources: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    paths: list[str] = []
    for raw_path in raw_paths:
        raw_lower = raw_path.lower()
        if "{mom}" in raw_lower or "{tsep}" in raw_lower:
            resolved = Path(raw_path).expanduser()
            if not resolved.is_absolute():
                resolved = (path.parent / resolved).resolve()
            pattern = resolved.name.replace("{mom}", "*").replace("{tsep}", "*")
            paths.extend(str(item) for item in sorted(resolved.parent.glob(pattern)))
        else:
            paths.append(raw_path)
    seen_data_paths: set[str] = set()
    ensemble_match = re.search(r"ensemble\s*[:=]?\s*([A-Za-z0-9_.-]+)", text, flags=re.I)
    ensemble = ensemble_match.group(1) if ensemble_match else None
    hadron = (
        "kaon"
        if "kaon" in lowered
        else "jpsi"
        if "jpsi" in lowered
        else "nucleon"
        if "nucleon" in lowered or "proton" in lowered
        else "pion"
        if "pion" in lowered
        else None
    )
    gfix_match = re.search(r"\b(CG|GI)\b", text, flags=re.I)
    gfix = gfix_match.group(1).upper() if gfix_match else "CG" if "coulomb" in lowered or "coulomb-gauge" in lowered else "GI" if "gauge invariant" in lowered else None
    spacing_match = re.search(r"\blattice_spacing_fm\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.I) or re.search(r"\blattice\s+spacing\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*(?:fm)?", text, flags=re.I) or re.search(r"(?:^|[,\s])a\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(?:fm)?", text, flags=re.I)
    lattice_spacing_fm = float(spacing_match.group(1)) if spacing_match else None
    volume_match = re.search(r"\bS[1-9]\d*T[1-9]\d*\b", text)
    volume = volume_match.group(0) if volume_match else None
    explicit_bz = list(dict.fromkeys(int(item) for item in re.findall(r"\bbz\s*=?\s*(-?\d+)", lowered)))
    explicit_tsep = list(dict.fromkeys(int(item) for item in re.findall(r"\btsep\s*=?\s*(\d+)", lowered)))
    explicit_bT = list(dict.fromkeys(int(item) for item in re.findall(r"\bbt\s*=?\s*(-?\d+)", lowered)))
    seed_match = re.search(r"random[_\s-]*seed\s*[:=]?\s*(\d+)", text, flags=re.I)
    resample_match = re.search(r"resample[_\s-]*mode\s*[:=]?\s*(jk|jackknife|bs|bootstrap)\b", text, flags=re.I)
    sample_error_match = re.search(r"sample[_\s-]*error[_\s-]*mode\s*[:=]?\s*(mean|median|covariance)\b", text, flags=re.I)
    source_operator_match = re.search(r"source[_\s-]*operator\s*[:=]?\s*([A-Za-z0-9_+-]+)", text, flags=re.I)
    sink_operator_match = re.search(r"sink[_\s-]*operator\s*[:=]?\s*([A-Za-z0-9_+-]+)", text, flags=re.I)
    current_operator_match = re.search(r"current[_\s-]*operator(?:\s+for\s+3pt)?\s*[:=]?\s*([A-Za-z0-9_+-]+)", text, flags=re.I)
    polarization_match = re.search(
        r"polarization\s*[:=]?\s*(unpolarized|helicity|transversity)\b",
        text,
        flags=re.I,
    )
    source_operator = source_operator_match.group(1) if source_operator_match else None
    sink_operator = sink_operator_match.group(1) if sink_operator_match else None
    current_operator = current_operator_match.group(1) if current_operator_match else None
    component_match = re.search(r"\bcomponent\s*[:=]?\s*(re|im|both)\b", text, flags=re.I)
    fit_scope_match = re.search(r"\bfit_scope\s*[:=]?\s*(3pt_ratio\+FH|3pt_ratio|qda_ratio|FH)\b", text, flags=re.I)
    fitting_form_match = re.search(r"\bfitting_form\s*[:=]?\s*(Breit|NonBreit)\b", text, flags=re.I)
    ft_order_match = re.search(r"\b(?:fourier\s+)?order\s*[:=]?\s*(LA|NLA)\b", text, flags=re.I)
    ft_sector_match = re.search(r"\bsector\s*[:=]?\s*(sea|valence|singlet|full)\b", text, flags=re.I)
    ft_part_match = re.search(r"\bpart\s*[:=]?\s*(re|im|both)\b", text, flags=re.I)
    y_grid_match = re.search(r"\by_grid\s*[:=]?\s*(\{[^{}]*\})", text, flags=re.I)
    scheme_scan_match = re.search(r"\bscheme_scan\s*[:=]?\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text, flags=re.I)
    quasi_y_match = re.search(r"\bquasi_y_ls\s*[:=]?\s*(\{[^{}]*\})", text, flags=re.I)
    lc_x_match = re.search(r"\blc_x_ls\s*[:=]?\s*(\{[^{}]*\})", text, flags=re.I)
    kernel_id_match = re.search(r"\bkernel_id\s*[:=]?\s*([A-Za-z0-9_+-]+)", text, flags=re.I)
    literature_match = re.search(r"\bliterature\s*[:=]?\s*(true|false)\b", text, flags=re.I)
    literature_max_papers_match = re.search(r"\bliterature_max_papers\s*[:=]?\s*(\d+)", text, flags=re.I)
    y_grid = json.loads(y_grid_match.group(1)) if y_grid_match else None
    scheme_scan = json.loads(scheme_scan_match.group(1)) if scheme_scan_match else None
    quasi_y_ls = json.loads(quasi_y_match.group(1)) if quasi_y_match else None
    lc_x_ls = json.loads(lc_x_match.group(1)) if lc_x_match else None
    initial_match = re.search(r"initial[_\s-]*momentum\s*[:=]?\s*(PX-?\d+PY-?\d+PZ-?\d+)", text, flags=re.I)
    final_match = re.search(r"final[_\s-]*momentum\s*[:=]?\s*(PX-?\d+PY-?\d+PZ-?\d+)", text, flags=re.I)
    momentum_match = re.search(r"(?<!initial[_\s-])(?<!final[_\s-])\bmomentum\s*[:=]?\s*(PX-?\d+PY-?\d+PZ-?\d+)", text, flags=re.I)
    explicit_momentum = momentum_match.group(1).upper() if momentum_match else None
    nonbreit = target_observable == "gpd" or "non-forward" in lowered or "nonbreit" in lowered
    for index, raw_path in enumerate(paths):
        token = raw_path.lower()
        is_current = "current" in token or "insertion" in token
        kind = "3pt" if "3pt" in token and not is_current else "2pt"
        label = "current" if is_current else kind
        resolved = Path(raw_path).expanduser()
        if not resolved.is_absolute():
            resolved = (path.parent / resolved).resolve()
        try:
            data_path = str(resolved.relative_to(root))
        except ValueError:
            data_path = str(resolved)
        if data_path in seen_data_paths:
            continue
        seen_data_paths.add(data_path)
        if resolved.suffix.lower() == ".nc":
            artifact_stage = "perturbative_matching" if resolved.stem.startswith("mt_") or "matching" in token else "fourier_transform" if resolved.stem.startswith("ft_") or "fourier" in token else "renormalization" if resolved.stem.startswith("rn_") or "renorm" in token else "correlator_analysis"
            artifact = {"id": resolved.stem, "stage": artifact_stage, "path": data_path}
            context_match = re.search(rf"{re.escape(resolved.name)}[^.\n]*(?:\.[^\n]*)?", text, flags=re.I)
            context = context_match.group(0) if context_match else text
            artifact_momentum = re.search(r"PX-?\d+PY-?\d+PZ-?\d+", context, flags=re.I)
            artifact_volume = re.search(r"\bS[1-9]\d*T[1-9]\d*\b", context)
            artifact_spacing = re.search(r"lattice[_\s-]*spacing(?:[_\s-]*fm)?\s*([0-9]*\.?[0-9]+)", context, flags=re.I)
            artifact_bz_direction = re.search(r"bz_direction\s*([A-Za-z]+)", context, flags=re.I)
            artifact_coord_unit = re.search(r"coord_unit\s*(lattice|fm|gev_inv|lambda)", context, flags=re.I)
            if artifact_momentum:
                artifact["momentum"] = artifact_momentum.group(0).upper()
            if artifact_volume:
                artifact["volume"] = artifact_volume.group(0)
            if artifact_spacing:
                artifact["lattice_spacing_fm"] = float(artifact_spacing.group(1))
            if "pion" in context.lower():
                artifact["hadron"] = "pion"
            if "kaon" in context.lower():
                artifact["hadron"] = "kaon"
            if "jpsi" in context.lower():
                artifact["hadron"] = "jpsi"
            if " cg" in context.lower() or "gfix cg" in context.lower():
                artifact["gfix"] = "CG"
            if " gi" in context.lower() or "gfix gi" in context.lower():
                artifact["gfix"] = "GI"
            if artifact_bz_direction:
                artifact["bz_direction"] = artifact_bz_direction.group(1).upper()
            if artifact_coord_unit:
                artifact["coord_unit"] = artifact_coord_unit.group(1).lower()
            artifacts.append(artifact)
            continue
        if is_current:
            operator_match = re.search(r"(?:current|insertion)[_-]?([A-Za-z0-9]+)?", resolved.stem, flags=re.I)
            current_sources.append(
                {
                    "source_id": f"current_{index}",
                    "data_path": data_path,
                    "current_operator": operator_match.group(1) if operator_match and operator_match.group(1) else current_operator,
                }
            )
            continue
        parsed = re.search(r"_p(?P<mom>[^_]+)_(?P<kind>[23]pt)(?:_ts(?P<tsep>\d+))?$", resolved.stem)
        canonical_momentum = re.search(r"PX-?\d+PY-?\d+PZ-?\d+", resolved.stem, flags=re.I)
        mom = parsed.group("mom") if parsed else None
        momentum = (
            canonical_momentum.group(0).upper()
            if canonical_momentum
            else mom
            if isinstance(mom, str) and mom.upper().startswith("PX")
            else f"PX{mom}PY0PZ0"
            if mom is not None
            else explicit_momentum
        )
        is_nonlocal_2pt = kind == "2pt" and ("nonlocal" in token or ("qda" in lowered and "nonlocal" in lowered and "local" not in token.replace("nonlocal", "")))
        if kind == "3pt":
            companion = resolved.with_name(resolved.stem.split("_3pt_ts", 1)[0] + "_2pt" + resolved.suffix)
            if companion.exists():
                try:
                    companion_data_path = str(companion.relative_to(root))
                except ValueError:
                    companion_data_path = str(companion)
                if companion_data_path not in seen_data_paths:
                    seen_data_paths.add(companion_data_path)
                    companion_correlator = {
                            "correlator_id": companion.stem,
                            "correlator_type": "2pt",
                            "data_path": companion_data_path,
                            "ensemble": ensemble,
                            "hadron": hadron,
                            "gfix": gfix,
                            "source_operator": source_operator,
                            "sink_operator": sink_operator,
                            "volume": volume,
                            "momentum": [momentum],
                            "lattice_spacing_fm": lattice_spacing_fm,
                            "plan_generated": True,
                    }
                    if momentum is None:
                        companion_correlator.pop("momentum")
                    correlators.append({key: value for key, value in companion_correlator.items() if value is not None})
        correlator = {
            "correlator_id": resolved.stem if resolved.exists() else f"{label}_{index}",
            "correlator_type": kind,
            "data_path": data_path,
            "ensemble": ensemble,
            "hadron": hadron,
            "gfix": gfix,
            "source_operator": source_operator,
            "sink_operator": sink_operator,
            "volume": volume,
            "momentum": [momentum],
            "lattice_spacing_fm": lattice_spacing_fm,
            "plan_generated": True,
        }
        if kind == "3pt":
            if current_operator is not None:
                correlator["current_operator"] = current_operator
            if polarization_match:
                correlator["polarization"] = polarization_match.group(1).lower()
            stem_tsep = re.search(r"(?:^|_)ts(?:ep)?(?P<tsep>\d+)(?:_|$)", resolved.stem, flags=re.I)
            if parsed and parsed.group("tsep"):
                correlator["tsep"] = [int(parsed.group("tsep"))]
            elif stem_tsep:
                correlator["tsep"] = [int(stem_tsep.group("tsep"))]
            elif explicit_tsep:
                correlator["tsep"] = explicit_tsep
            if explicit_bT:
                correlator["bT"] = explicit_bT
            if resolved.exists() and resolved.suffix.lower() == ".npy":
                import numpy as np

                shape = np.load(resolved, mmap_mode="r").shape
                correlator["bz"] = list(range(int(shape[0]))) if len(shape) >= 3 else explicit_bz or None
            else:
                if explicit_bz:
                    correlator["bz"] = explicit_bz
            direction_match = re.search(r"bz_direction\s*[:=]?\s*(X|Y|Z|XY|XZ|YZ|XYZ)\b", text, flags=re.I)
            if direction_match:
                correlator["bz_direction"] = direction_match.group(1).upper()
            elif "_x_" in token:
                correlator["bz_direction"] = "X"
        elif is_nonlocal_2pt:
            if explicit_bT:
                correlator["bT"] = explicit_bT
            if resolved.exists() and resolved.suffix.lower() == ".npy":
                import numpy as np

                shape = np.load(resolved, mmap_mode="r").shape
                correlator["bz"] = list(range(int(shape[0]))) if len(shape) >= 3 else explicit_bz or None
            else:
                if explicit_bz:
                    correlator["bz"] = explicit_bz
            direction_match = re.search(r"bz_direction\s*[:=]?\s*(X|Y|Z|XY|XZ|YZ|XYZ)\b", text, flags=re.I)
            if direction_match:
                correlator["bz_direction"] = direction_match.group(1).upper()
        if momentum is None:
            correlator.pop("momentum", None)
        correlator = {key: value for key, value in correlator.items() if value is not None}
        correlators.append(correlator)
    two_points = [item for item in correlators if item.get("correlator_type") == "2pt" and item.get("bz") is None]
    for current_source in current_sources:
        for two_point in two_points:
            suffix = ""
            if len(current_sources) > 1 or len(two_points) > 1:
                momentum_label = str(_as_list(two_point.get("momentum"))[0]).replace("PX", "px").replace("PY", "py").replace("PZ", "pz")
                suffix = f"_{current_source['current_operator']}_{momentum_label}"
            composed = {
                "correlator_id": f"planned_3pt_from_current{suffix}",
                "correlator_type": "3pt",
                "data_path": f"artifacts/plan_data/{run_id}_planned_3pt{suffix}.h5",
                "current_operator": current_source.get("current_operator"),
                "plan_sources": {"two_point": two_point["data_path"], "current": current_source["data_path"]},
            }
            for key in ("ensemble", "hadron", "gfix", "source_operator", "sink_operator", "momentum", "volume", "lattice_spacing_fm"):
                if key in two_point:
                    composed[key] = two_point[key]
            if polarization_match:
                composed["polarization"] = polarization_match.group(1).lower()
            if explicit_tsep:
                composed["tsep"] = explicit_tsep
            if explicit_bT:
                composed["bT"] = explicit_bT
            if explicit_bz:
                composed["bz"] = explicit_bz
            direction_match = re.search(r"bz_direction\s*[:=]?\s*(X|Y|Z|XY|XZ|YZ|XYZ)\b", text, flags=re.I)
            if direction_match:
                composed["bz_direction"] = direction_match.group(1).upper()
            correlators.append({key: value for key, value in composed.items() if value is not None})
    stages = ["correlator_analysis"] if correlators else []
    for stage in ("renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"):
        if (stage in lowered or ("matching" in lowered and stage == "perturbative_matching") or ("fourier" in lowered and stage == "fourier_transform")) and stage not in stages:
            stages.append(stage)
    if artifacts and not stages:
        if any(item["stage"] == "renormalization" for item in artifacts):
            stages.extend(["fourier_transform", "perturbative_matching"])
        elif any(item["stage"] == "fourier_transform" for item in artifacts):
            stages.append("perturbative_matching")
        elif any(item["stage"] == "perturbative_matching" for item in artifacts):
            stages.append("extrapolation")
        if "review" in lowered:
            stages.append("review")
    payload: dict[str, Any] = {
        "metadata": {
            "run_id": run_id,
            "root_directory": str(root),
            "target_observable": target_observable,
            "parton": parton,
            "stages": stages,
        },
        "inputs": {"correlators": correlators, "artifacts": artifacts, "kernels": []},
        "stages": {},
    }
    if seed_match:
        payload["metadata"]["random_seed"] = int(seed_match.group(1))
    if resample_match:
        mode = resample_match.group(1).lower()
        payload["metadata"]["resample_mode"] = "jk" if mode in {"jk", "jackknife"} else "bs"
    if sample_error_match:
        payload["metadata"]["sample_error_mode"] = sample_error_match.group(1).lower()
    if correlators:
        jobs_by_momentum: dict[str, list[str]] = {}
        unassigned_ids: list[str] = []
        for item in correlators:
            momenta = _as_list(item.get("momentum"))
            if not momenta:
                unassigned_ids.append(str(item["correlator_id"]))
                continue
            momentum = str(momenta[0])
            jobs_by_momentum.setdefault(momentum, []).append(str(item["correlator_id"]))
        has_qda = any(item.get("correlator_type") == "2pt" and item.get("bz") is not None for item in correlators)
        ca_defaults: dict[str, Any] = {}
        if fit_scope_match:
            ca_defaults["fit_scope"] = [fit_scope_match.group(1)]
        if fitting_form_match:
            ca_defaults["fitting_form"] = fitting_form_match.group(1)
        if component_match:
            ca_defaults["component"] = component_match.group(1).lower()
        initial_momentum = initial_match.group(1).upper() if initial_match else None
        final_momentum = final_match.group(1).upper() if final_match else None
        if nonbreit and initial_momentum and final_momentum:
            payload["stages"]["correlator_analysis"] = {
                "defaults": ca_defaults,
                "jobs": [{"id": "ca_nonforward", "correlator_ids": [item["correlator_id"] for item in correlators], "params": {"initial_momentum": initial_momentum, "final_momentum": final_momentum}}],
            }
        else:
            ca_jobs = []
            for index, (momentum, ids) in enumerate(sorted(jobs_by_momentum.items())):
                match = re.fullmatch(r"PX(-?\d+)PY(-?\d+)PZ(-?\d+)", momentum)
                components = [int(value) for value in match.groups()] if match else []
                nonzero = next((abs(value) for value in reversed(components) if value), index)
                ca_jobs.append({"id": f"ca_p{nonzero}", "correlator_ids": ids, "params": {"momentum": momentum}})
            if unassigned_ids:
                ca_jobs.append({"id": "ca_unassigned", "correlator_ids": unassigned_ids, "params": {}})
            payload["stages"]["correlator_analysis"] = {
                "defaults": ca_defaults,
                "jobs": ca_jobs,
            }
    if "renormalization" in stages and "correlator_analysis" in payload["stages"]:
        ca_jobs = payload["stages"]["correlator_analysis"]["jobs"]
        denominator = next((job["id"] for job in ca_jobs if job["params"].get("momentum") == "PX0PY0PZ0"), None)
        nonzero_jobs = [job for job in ca_jobs if denominator is None or job["id"] != denominator]
        scheme = "hybrid" if "hybrid" in lowered else "ratio" if "ratio" in lowered else None
        strategy = "self_renormalization" if "self_renormalization" in lowered else "external_denominator" if "external_denominator" in lowered else None
        rn_defaults: dict[str, Any] = {}
        if scheme is not None:
            rn_defaults["scheme"] = scheme
        if strategy is not None:
            rn_defaults["strategy"] = strategy
        zs_match = re.search(r"zs_fm(?:\s+for\s+[A-Za-z0-9_+-]+)?\s*[:=]?\s*([0-9]*\.?[0-9]+)", lowered)
        if zs_match:
            rn_defaults["zs_fm"] = float(zs_match.group(1))
        rn_jobs = []
        for job in nonzero_jobs:
            inputs = {"target": job["id"]}
            if denominator is not None:
                inputs["denominator"] = denominator
            rn_jobs.append({"id": job["id"].replace("ca_", "rn_"), "inputs": inputs})
        payload["stages"]["renormalization"] = {"defaults": rn_defaults, "jobs": rn_jobs}
    if "fourier_transform" in stages:
        rn_jobs = payload["stages"].get("renormalization", {}).get("jobs", [])
        source_jobs = rn_jobs or [{"id": item["id"]} for item in artifacts if item["stage"] == "renormalization"]
        ft_defaults: dict[str, Any] = {}
        if not rn_jobs and gfix is not None:
            ft_defaults["gfix"] = gfix
        ft_grid = quasi_y_ls if quasi_y_ls is not None else y_grid
        if ft_grid is not None:
            ft_defaults["quasi_y_ls"] = ft_grid
        if ft_order_match:
            ft_defaults["order"] = ft_order_match.group(1).upper()
        if ft_sector_match:
            ft_defaults["sector"] = "full" if target_observable == "da" or parton == "gluon" else ft_sector_match.group(1).lower()
        if component_match:
            ft_defaults["component"] = component_match.group(1).lower()
        elif ft_part_match:
            ft_defaults["component"] = ft_part_match.group(1).lower()
        if not correlators and hadron != "hadron":
            ft_defaults["hadron"] = hadron
        if polarization_match and not rn_jobs:
            ft_defaults["polarization"] = polarization_match.group(1).lower()
        if scheme_scan is not None:
            ft_defaults["scheme_scan"] = scheme_scan
        payload["stages"]["fourier_transform"] = {
            "defaults": ft_defaults,
            "jobs": [{"id": job["id"].replace("rn_", "ft_"), "inputs": {"input": job["id"]}} for job in source_jobs],
        }
    if "perturbative_matching" in stages:
        ft_jobs = payload["stages"].get("fourier_transform", {}).get("jobs", []) or [{"id": item["id"]} for item in artifacts if item["stage"] == "fourier_transform"]
        kernel_id = kernel_id_match.group(1) if kernel_id_match else None
        scheme = payload["stages"].get("renormalization", {}).get("defaults", {}).get("scheme")
        if scheme is None:
            scheme = "hybrid" if re.search(r"\bhybrid\b", lowered) else "ratio" if re.search(r"\bratio\b", lowered) else None
        if scheme is None and kernel_id is not None:
            scheme = "hybrid" if "hybrid" in kernel_id.lower() else "ratio" if "ratio" in kernel_id.lower() else None
        if kernel_id is not None:
            payload["inputs"]["kernels"] = [{"stage": "perturbative_matching", "kernel_id": kernel_id, "kernel_path": str(Path(__file__).resolve().parents[1] / "kernels.py")}]
        mt_defaults: dict[str, Any] = {}
        if kernel_id is not None:
            mt_defaults["kernel_id"] = kernel_id
        if scheme is not None:
            mt_defaults["scheme"] = scheme
        if lc_x_ls is not None:
            mt_defaults["lc_x_ls"] = lc_x_ls
        zs_match = re.search(r"zs_fm(?:\s+if[^:]*|\s+for\s+[A-Za-z0-9_+-]+)?\s*[:=]?\s*([0-9]*\.?[0-9]+)", lowered)
        if zs_match:
            mt_defaults["zs_fm"] = float(zs_match.group(1))
        payload["stages"]["perturbative_matching"] = {
            "defaults": mt_defaults,
            "jobs": [{"id": job["id"].replace("ft_", "mt_"), "inputs": {"quasi": job["id"]}} for job in ft_jobs],
        }
    if "extrapolation" in stages:
        mt_jobs = payload["stages"].get("perturbative_matching", {}).get("jobs", []) or [{"id": item["id"]} for item in artifacts if item["stage"] == "perturbative_matching"]
        payload["stages"]["extrapolation"] = {"defaults": {}, "jobs": [{"id": "extrapolate_all", "inputs": {"lightcone": [job["id"] for job in mt_jobs]}}]}
    if "review" in stages:
        review_defaults: dict[str, Any] = {}
        if literature_match or literature_max_papers_match:
            review_defaults["literature"] = literature_match is None or literature_match.group(1).lower() == "true"
        if literature_max_papers_match:
            review_defaults["literature_max_papers"] = int(literature_max_papers_match.group(1))
        payload["stages"]["review"] = {"defaults": review_defaults, "jobs": [{"id": "review"}]}
    payload["metadata"]["stages"] = [
        stage
        for stage in stages
        if stage in payload["stages"] and payload["stages"][stage].get("jobs")
    ]
    payload["stages"] = {stage: payload["stages"][stage] for stage in payload["metadata"]["stages"]}
    return payload


def normalize_planning_constraints(payload: dict[str, Any]) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    correlators = payload.get("inputs", {}).get("correlators", []) if isinstance(payload.get("inputs"), dict) else []
    for index, correlator in enumerate(correlators if isinstance(correlators, list) else []):
        if not isinstance(correlator, dict):
            continue
        for key in ("bT", "bz", "tsep"):
            value = correlator.get(key)
            if not isinstance(value, list):
                continue
            deduped = list(dict.fromkeys(value))
            if deduped != value:
                correlator[key] = deduped
                edits.append({"path": f"inputs.correlators[{index}].{key}", "old": value, "new": deduped, "note": f"Deduplicated correlator {key} values."})
    stages = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
    for stage in stages:
        try:
            normalizer = get_stage_parameter_contract(stage).normalize_draft
        except ValueError:
            continue
        if normalizer is not None:
            edits.extend(normalizer(payload))
    return edits


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
    if isinstance(stage_order, list):
        for stage in stages:
            if stage not in stage_order_list:
                issues.append(
                    PlanIssue(
                        "error",
                        f"stages.{stage}",
                        f"`stages.{stage}` is configured but not listed in `metadata.stages`.",
                        f"Add `{stage}` to `metadata.stages` to run it, or remove `stages.{stage}`.",
                    )
                )

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
        if isinstance(item.get("plan_sources"), dict):
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
    matching = stages.get("perturbative_matching")
    matching_scheme = (
        matching.get("defaults", {}).get("scheme")
        if isinstance(matching, dict) and isinstance(matching.get("defaults"), dict)
        else None
    )
    if (
        isinstance(renorm_scheme, str)
        and isinstance(matching_scheme, str)
        and matching_scheme != renorm_scheme
    ):
        issues.append(
            PlanIssue(
                "warning",
                "stages.perturbative_matching.defaults.scheme",
                f"Matching scheme `{matching_scheme}` differs from renormalization scheme `{renorm_scheme}`.",
                f"Set the matching scheme to `{renorm_scheme}` unless this is intentional.",
            )
        )

    strict_payload = copy.deepcopy(payload)
    for correlator in strict_payload.get("inputs", {}).get("correlators", []) if isinstance(strict_payload.get("inputs"), dict) else []:
        if isinstance(correlator, dict):
            correlator.pop("plan_generated", None)
            correlator.pop("plan_sources", None)
    for kernel in strict_payload.get("inputs", {}).get("kernels", []) if isinstance(strict_payload.get("inputs"), dict) else []:
        if isinstance(kernel, dict) and kernel.get("stage") == "matching":
            kernel["stage"] = "perturbative_matching"
    try:
        AnalysisManifest.model_validate(strict_payload)
    except Exception as exc:
        issues.append(PlanIssue("info", "manifest", f"Strict manifest validation is not yet clean: {exc}"))

    for gap in _stage_parameter_gaps(payload, manifest_path):
        message = str(gap["message"])
        if gap.get("physics"):
            message += f" Physics: {gap['physics']}"
        issues.append(PlanIssue("warning", str(gap["path"]), message, str(gap["suggested_fix"])))

    return issues


def _set_kernel_scheme_from_renorm(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Move legacy kernel scheme fields to stage defaults and split renorm composites."""
    edits: list[dict[str, Any]] = []
    stages = payload.get("stages")
    inputs = payload.get("inputs")
    if not isinstance(stages, dict) or not isinstance(inputs, dict):
        return edits
    renorm = stages.get("renormalization")
    if isinstance(renorm, dict) and isinstance(renorm.get("defaults"), dict):
        defaults = renorm["defaults"]
        legacy = defaults.get("scheme")
        mapping = {
            "hybrid_ratio": ("hybrid", "ratio"),
            "hybrid_self_renormalization": ("ratio", "self_renormalization"),
            "self_renormalization": ("ratio", "self_renormalization"),
        }
        if legacy in mapping:
            scheme, strategy = mapping[str(legacy)]
            defaults["scheme"] = scheme
            defaults.setdefault("strategy", strategy)
            edits.append(
                {
                    "path": "stages.renormalization.defaults",
                    "old": {"scheme": legacy},
                    "new": {"scheme": scheme, "strategy": defaults["strategy"]},
                }
            )
        elif legacy in {"ratio", "hybrid", "msbar"} and "strategy" not in defaults:
            defaults["strategy"] = "external_denominator"
            edits.append(
                {
                    "path": "stages.renormalization.defaults.strategy",
                    "old": None,
                    "new": "external_denominator",
                }
            )
    kernels = inputs.get("kernels")
    if not isinstance(kernels, list):
        return edits
    for index, kernel in enumerate(kernels):
        if not isinstance(kernel, dict):
            continue
        old = kernel.pop("scheme", None)
        if old is not None:
            edits.append({"path": f"inputs.kernels[{index}].scheme", "old": old, "new": None})
        if kernel.get("stage") != "perturbative_matching":
            continue
        kernel_id = str(kernel.get("kernel_id", ""))
        encoded = next(
            (value for value in ("ratio", "hybrid", "msbar") if value in kernel_id.split("_")),
            None,
        )
        matching = stages.get("perturbative_matching")
        if encoded is not None and isinstance(matching, dict):
            defaults = matching.setdefault("defaults", {})
            if isinstance(defaults, dict) and "scheme" not in defaults:
                defaults["scheme"] = encoded
                edits.append(
                    {
                        "path": "stages.perturbative_matching.defaults.scheme",
                        "old": None,
                        "new": encoded,
                    }
                )
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


def _expand_pt3_windows(value: Any) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                tau_cut = int(item["tau_cut"])
            except (KeyError, TypeError, ValueError):
                continue
            window: dict[str, Any] = {"tau_cut": tau_cut}
            tsep_ls = item.get("tsep_ls")
            if isinstance(tsep_ls, list):
                try:
                    window["tsep_ls"] = [int(t) for t in tsep_ls]
                except (TypeError, ValueError):
                    continue
            windows.append(window)
    if not windows:
        return [{"tau_cut": cut} for cut in _expand_tau_cuts(None)]
    grouped: dict[tuple[int, ...] | None, list[int]] = {}
    for window in windows:
        tsep_key = tuple(window["tsep_ls"]) if "tsep_ls" in window else None
        grouped.setdefault(tsep_key, []).append(int(window["tau_cut"]))
    expanded: list[dict[str, Any]] = []
    for tsep_key, cuts in grouped.items():
        for cut in _expand_tau_cuts(cuts):
            item: dict[str, Any] = {"tau_cut": cut}
            if tsep_key is not None:
                item["tsep_ls"] = list(tsep_key)
            expanded.append(item)
    return expanded


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
        correlator.pop("plan_sources", None)
        correlator.pop("plan_generated", None)
        edits.append({"path": f"inputs.correlators[{index}].data_path", "old": old, "new": new_path})
    return edits


def _shrink_list(value: Any) -> Any:
    return value[:1] if isinstance(value, list) and value else value


def _make_quick_variant(payload: dict[str, Any]) -> dict[str, Any]:
    quick = copy.deepcopy(payload)
    stages = quick.get("stages")
    if not isinstance(stages, dict):
        return quick
    for stage_name, config in stages.items():
        if not isinstance(config, dict):
            continue
        defaults = config.get("defaults")
        if isinstance(defaults, dict):
            for key in ("pt2_windows", "pt3_windows", "nstate", "fit_scope", "fit_strategy", "prior_width", "order"):
                if key in defaults:
                    defaults[key] = _shrink_list(defaults[key])
            scheme_scan = defaults.get("scheme_scan")
            if isinstance(scheme_scan, dict):
                for key in ("zmin_fm", "zmax_fm", "order", "posterior_prior_error_scale"):
                    if key in scheme_scan:
                        scheme_scan[key] = _shrink_list(scheme_scan[key])
                if isinstance(scheme_scan.get("max_schemes"), int):
                    scheme_scan["max_schemes"] = min(int(scheme_scan["max_schemes"]), 8)
        jobs = config.get("jobs")
        if isinstance(jobs, list):
            for job in jobs:
                if not isinstance(job, dict) or not isinstance(job.get("params"), dict):
                    continue
                for key in ("pt2_windows", "pt3_windows", "nstate", "fit_scope", "fit_strategy", "prior_width", "order"):
                    if key in job["params"]:
                        job["params"][key] = _shrink_list(job["params"][key])
    return quick


def _make_full_variant(payload: dict[str, Any], *, suppressed_paths: set[str] | None = None) -> dict[str, Any]:
    full = copy.deepcopy(payload)
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
    for correlator in base.get("inputs", {}).get("correlators", []):
        if isinstance(correlator, dict):
            correlator.pop("plan_generated", None)
            correlator.pop("plan_sources", None)
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
    if any(part in {"plan_sources", "plan_generated"} for part in parts):
        raise ValueError("JSON Patch may not modify plan-only conversion fields.")
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


def _strict_manifest_issues(payload: dict[str, Any], manifest_path: Path | None = None) -> list[PlanIssue]:
    issues: list[PlanIssue] = []
    strict_payload = copy.deepcopy(payload)
    for correlator in strict_payload.get("inputs", {}).get("correlators", []) if isinstance(strict_payload.get("inputs"), dict) else []:
        if isinstance(correlator, dict):
            correlator.pop("plan_generated", None)
            correlator.pop("plan_sources", None)
    try:
        strict = AnalysisManifest.model_validate(strict_payload)
        if manifest_path is not None:
            for artifact in strict.inputs.artifacts:
                resolved = _resolve_manifest_path(manifest_path, payload, artifact.path)
                if resolved is not None:
                    artifact.path = resolved.as_posix()
    except Exception as exc:
        return [PlanIssue("error", "manifest", f"Strict manifest validation failed: {exc}")]
    for stage in strict.metadata.stages:
        if stage not in strict.stages:
            continue
        for job in strict.stages[stage].jobs:
            try:
                for diagnostic in validate_stage_diagnostics(stage, strict, job):
                    issues.append(
                        PlanIssue(
                            "error",
                            diagnostic.path,
                            diagnostic.detailed_message(),
                            diagnostic.suggested_fix or None,
                        )
                    )
            except Exception as exc:
                issues.append(PlanIssue("error", f"stages.{stage}.jobs.{job.id}", f"Stage input check failed: {exc}"))
    return issues


def _stage_parameter_gaps(payload: dict[str, Any], manifest_path: Path | None = None) -> list[dict[str, Any]]:
    metadata = payload.get("metadata", {})
    stages = payload.get("stages", {})
    inputs = payload.get("inputs", {})
    order = metadata.get("stages", []) if isinstance(metadata, dict) else []
    stage_order = [stage for stage in order if isinstance(stage, str)] if isinstance(order, list) else []
    kernels = inputs.get("kernels", []) if isinstance(inputs, dict) else []
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

    def derived_metadata(stage_id: str, candidate: dict[str, Any], seen: set[str]) -> dict[str, Any]:
        if stage_id == "correlator_analysis":
            config = stages.get(stage_id, {}) if isinstance(stages, dict) else {}
            stage_defaults = config.get("defaults", {}) if isinstance(config, dict) else {}
            params = resolve_stage_params(
                stage_id,
                stage_defaults if isinstance(stage_defaults, dict) else {},
                candidate.get("params") if isinstance(candidate.get("params"), dict) else {},
            )
            momentum = (
                params.get("final_momentum")
                if str(params.get("fitting_form")) == "NonBreit"
                else params.get("momentum")
            )
            ids = set(candidate.get("correlator_ids", []))
            two_point = next(
                (
                    item
                    for item in correlators
                    if isinstance(item, dict)
                    and item.get("correlator_id") in ids
                    and item.get("correlator_type") == "2pt"
                    and momentum in _as_list(item.get("momentum"))
                ),
                {},
            )
            three_point = next(
                (
                    item
                    for item in correlators
                    if isinstance(item, dict)
                    and item.get("correlator_id") in ids
                    and item.get("correlator_type") == "3pt"
                    and momentum in _as_list(item.get("momentum"))
                ),
                {},
            )
            selected_momentum = {"momentum": momentum} if two_point and momentum is not None else {}
            nonbreit_momenta = (
                {
                    "initial_momentum": params.get("initial_momentum"),
                    "final_momentum": params.get("final_momentum"),
                }
                if str(params.get("fitting_form")) == "NonBreit"
                else {}
            )
            return {
                **two_point,
                "_gfix_source": "correlator",
                **selected_momentum,
                **nonbreit_momenta,
                **{
                    key: three_point[key]
                    for key in ("hadron", "current_operator", "polarization")
                    if three_point.get(key) is not None
                },
            }
        resolved: dict[str, Any] = {}
        candidate_inputs = candidate.get("inputs") or {}
        ordered_roles = ("input", "quasi", "target", "reference", "denominator", "zR")
        for value in (
            [candidate_inputs[role] for role in ordered_roles if role in candidate_inputs]
            + [value for role, value in candidate_inputs.items() if role not in ordered_roles]
        ):
            for reference in _as_list(value):
                reference = str(reference)
                artifact = artifacts.get(reference)
                if artifact is not None:
                    artifact_metadata = dict(artifact)
                    artifact_metadata["_gfix_source"] = "artifact"
                    if manifest_path is not None:
                        path = _resolve_manifest_path(manifest_path, payload, artifact.get("path"))
                        if path is not None and path.suffix.lower() == ".nc" and path.is_file():
                            from lamet_agent.core.data import read_netcdf_attrs

                            try:
                                attrs = read_netcdf_attrs(path)
                            except Exception:
                                attrs = {}
                            artifact_metadata = {**artifact_metadata, **attrs}
                    for key, item in artifact_metadata.items():
                        if item is not None:
                            resolved.setdefault(key, item)
                upstream = jobs_by_id.get(reference)
                if upstream is not None and reference not in seen:
                    for key, item in derived_metadata(upstream[0], upstream[1], seen | {reference}).items():
                        if item is not None:
                            resolved.setdefault(key, item)
        return resolved

    gaps: list[dict[str, Any]] = []
    seen_gaps: set[tuple[str, str, str]] = set()
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
        try:
            contract = get_stage_parameter_contract(stage)
        except ValueError:
            continue
        for index, job in enumerate(jobs):
            if not isinstance(job, dict):
                continue
            job_id = str(job.get("id", index))
            authored_params = merge_stage_params(
                defaults,
                job.get("params") if isinstance(job.get("params"), dict) else {},
            )
            upstream_metadata = derived_metadata(stage, job, {job_id})
            gfix_source = upstream_metadata.pop("_gfix_source", None)
            inherited_gfix = upstream_metadata.get("gfix")
            resolved_params = resolve_stage_params(
                stage, defaults, job.get("params") if isinstance(job.get("params"), dict) else {}
            )
            effective_params = {**upstream_metadata, **resolved_params}
            if stage == "fourier_transform" and gfix_source == "artifact" and "gfix" not in authored_params:
                effective_params.pop("gfix", None)
            if all(
                upstream_metadata.get(key) is not None
                for key in ("momentum", "volume", "lattice_spacing_fm")
            ):
                try:
                    effective_params["momentum_gev"] = physical_momentum_gev(
                        str(upstream_metadata["momentum"]),
                        str(upstream_metadata["volume"]),
                        float(upstream_metadata["lattice_spacing_fm"]),
                    )
                except (TypeError, ValueError):
                    pass
            selected_correlators = [
                item
                for item in correlators
                if isinstance(item, dict)
                and item.get("correlator_id") in set(job.get("correlator_ids", []))
            ]
            partner_kinematics = {}
            if stage == "fourier_transform" and isinstance(job.get("inputs"), dict):
                partner_reference = job["inputs"].get("hermitian_partner")
                if isinstance(partner_reference, str):
                    partner_kinematics = derived_metadata(
                        stage,
                        {"inputs": {"input": partner_reference}},
                        {job_id},
                    )
            context = StageValidationContext(
                stage=stage,
                job_id=job_id,
                job_path=f"stages.{stage}.jobs.{job_id}",
                params=effective_params,
                inputs=dict(job.get("inputs", {})) if isinstance(job.get("inputs"), dict) else {},
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
                resources={
                    "kernels": list(kernels) if isinstance(kernels, list) else [],
                    "selected_correlators": selected_correlators,
                    "gfix_source": gfix_source,
                    "inherited_gfix": inherited_gfix,
                    "partner_kinematics": partner_kinematics,
                },
                authored_params=authored_params,
            )
            for diagnostic in contract.evaluate(context):
                parameter = diagnostic.parameters[0] if diagnostic.parameters else "params"
                display_parameter = (
                    "momentum_gev"
                    if parameter in {"derived.momentum_gev", "momentum_gev"}
                    else "inputs"
                    if parameter == "inputs"
                    else parameter
                )
                if parameter == "inputs.kernels":
                    path = "inputs.kernels"
                elif parameter == "correlator_ids":
                    path = f"stages.{stage}.jobs[{index}].correlator_ids"
                elif parameter.startswith("inputs"):
                    path = f"stages.{stage}.jobs[{index}].inputs"
                elif parameter.startswith("derived.") or parameter == "momentum_gev":
                    path = f"stages.{stage}.jobs[{index}].inputs"
                elif parameter in contract.job_parameters:
                    path = f"stages.{stage}.jobs[{index}].params.{parameter}"
                else:
                    path = f"stages.{stage}.defaults.{parameter}"
                gap_key = (stage, path, diagnostic.code)
                if gap_key in seen_gaps:
                    continue
                seen_gaps.add(gap_key)
                gaps.append(
                    {
                        "stage": stage,
                        "job_id": job_id,
                        "parameter": display_parameter,
                        "path": path,
                        "code": diagnostic.code,
                        "message": diagnostic.message,
                        "physics": diagnostic.physics,
                        "suggested_fix": diagnostic.suggested_fix,
                        "question_id": f"stage_params.{stage}.{'shared' if '.defaults.' in path else job_id}.{display_parameter}",
                    }
                )
    return gaps


def validate_candidate_payload(manifest_path: Path, payload: dict[str, Any]) -> tuple[bool, list[PlanIssue]]:
    """Validate a candidate manifest payload before it can become writable state."""
    issues = check_manifest_draft(manifest_path, payload)
    issues.extend(_strict_manifest_issues(payload, manifest_path))
    blocking = [issue for issue in issues if issue.severity == "error"]
    return not blocking, issues
