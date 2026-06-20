"""Stage tool-registry resolution and call preparation for the agent loop."""

from __future__ import annotations

import inspect
import logging
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from lamet_agent.manifest import AnalysisManifest, ArtifactInput, StageJob

from .stages import resolve_stage_package

_PLOT_TOOLS = frozenset({"tune_ground_state", "tune_bare_matrix", "fit_bare_matrix_grid", "plot_matched_pdf"})
_RENORM_ARTIFACT_TOOLS = frozenset({"apply_ratio_scheme_renormalization", "plot_renormalized_matrix_element"})
_FOURIER_ARTIFACT_TOOLS = frozenset(
    {
        "run_fourier_transform",
        "plot_fourier_result",
        "plot_fourier_extension_quality_result",
        "report_fourier_result",
    }
)
_FOURIER_LOAD_KEYS = frozenset({"input_format", "h5_group", "coord_key", "re_key", "im_key", "resample_mode"})
_FOURIER_RUN_KEYS = frozenset(
    {
        "k_grid",
        "scheme_scan",
        "method",
        "order",
        "observable",
        "coord_unit",
        "pz_gev",
        "pz_prime_gev",
        "a_fm",
        "im_flip_for_ft",
        "Lambda0",
        "posterior_prior_error_scale",
        "fit_error_mode",
        "part",
        "output_scale",
        "save_path",
        "plot_fourier",
        "plot_extension",
        "report",
    }
)
_MATCHING_KERNEL_KEYS = frozenset({"kernel_id", "pz_gev", "mu", "zs_fm"})


def setup_logger(
    log_file: str | Path,
    console_output: bool = False,
    mode: str = "w",
    logger_name: str = "my_logger",
) -> logging.Logger:
    """Create and configure a file logger with optional console output."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(path, mode=mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def log_nonlinear_fit_quality(
    fit: Any,
    *,
    kind: str = "fit",
    label: str | None = None,
    logger: logging.Logger | None = None,
    q_min: float = 0.05,
) -> str:
    """Log a compact Good/Bad quality line for an lsqfit nonlinear fit."""
    use_logger = logger or logging.getLogger("my_logger")
    name = f"{kind} {label}" if label else kind
    q_value = float(getattr(fit, "Q", float("nan")))
    chi2 = float(getattr(fit, "chi2", float("nan")))
    dof = int(getattr(fit, "dof", 0) or 0)
    loggbf = float(getattr(fit, "logGBF", float("nan")))
    chi2_dof = chi2 / dof if dof else float("nan")
    status = "Good" if q_value >= float(q_min) else "Bad"
    message = (
        "%s %s: Q=%.6g chi2/dof=%.6g chi2=%.6g dof=%s logGBF=%.6g",
        status,
        name,
        q_value,
        chi2_dof,
        chi2,
        dof,
        loggbf,
    )
    if status == "Bad":
        use_logger.warning(*message)
    else:
        use_logger.info(*message)
    return status

def resolve_plot_save_path(
    raw: str | None,
    *,
    artifacts_dir: Path,
    default_stem: str = "fit_on_data",
    root_directory: Path | None = None,
) -> str:
    """Resolve output stems.

    Defaults go under ``artifacts_dir``. Explicit relative paths are resolved
    against ``root_directory`` when the manifest declares one; otherwise they
    preserve the historical behavior of writing under ``artifacts_dir``.
    """
    if raw:
        if root_directory is None:
            stem = Path(raw).name
            for suffix in (".png", ".pdf", ".svg"):
                if stem.lower().endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            if not stem:
                stem = default_stem
            return str(artifacts_dir / stem)

        stem_path = Path(raw).expanduser()
        stem_text = str(stem_path)
        for suffix in (".png", ".pdf", ".svg"):
            if stem_text.lower().endswith(suffix):
                stem_text = stem_text[: -len(suffix)]
                break
        stem_path = Path(stem_text)
        if str(stem_path) in {"", "."}:
            stem_path = Path(default_stem)
        if stem_path.is_absolute():
            return str(stem_path)
        if root_directory is not None:
            return str((root_directory / stem_path).resolve())
    else:
        stem = default_stem
    return str(artifacts_dir / stem)


def _manifest_root(manifest: AnalysisManifest) -> Path | None:
    root = manifest.root_directory
    return Path(root).expanduser().resolve() if root is not None else None


def _run_scoped_plot_stem(manifest: AnalysisManifest, stem: str) -> str:
    """Prefix default plot stems with the run id so adjacent runs do not collide."""
    run_id = Path(str(manifest.run_id)).name or "run"
    return f"{run_id}_{stem}"


def resolve_stage_tools(stage: str) -> dict[str, Callable[..., dict[str, Any]]]:
    """Return the ``STAGE_TOOLS`` registry for a stage, or an empty dict."""
    package_name = resolve_stage_package(stage)
    if not package_name:
        return {}
    module = import_module(f"lamet_agent.stages.{package_name}.functions")
    return getattr(module, "STAGE_TOOLS", {})


def validate_stage_inputs(stage: str, manifest: Any, job: StageJob) -> list[str]:
    """Return a stage's input issues via its ``validate_stage_inputs`` helper."""
    package_name = resolve_stage_package(stage)
    if not package_name:
        return []
    module = import_module(f"lamet_agent.stages.{package_name}.skills")
    validator = getattr(module, "validate_stage_inputs", None)
    return validator(manifest, job) if callable(validator) else []


def _resolve_one_data_path(value: str, manifest: AnalysisManifest) -> str:
    if Path(value).is_absolute():
        return value
    return str((manifest.root_directory / value).resolve())


def _resolve_path_container(value: Any, manifest: AnalysisManifest) -> Any:
    if isinstance(value, str):
        return _resolve_one_data_path(value, manifest)
    if isinstance(value, list):
        return [_resolve_path_container(item, manifest) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_path_container(item, manifest) for key, item in value.items()}
    return value


def _declared_artifact_path(manifest: AnalysisManifest, job: StageJob, role: str) -> str | None:
    """Return the resolved path for a job input role backed by inputs.artifacts."""
    ref = job.inputs.get(role)
    if not isinstance(ref, str):
        return None
    for artifact in manifest.inputs.artifacts:
        if artifact.id == ref:
            return artifact.path
    return None


def resolve_tool_args(args: dict[str, Any], manifest: AnalysisManifest) -> dict[str, Any]:
    """Resolve manifest-relative file paths in tool arguments."""
    if manifest.root_directory is None and (manifest.manifest_dir is None or manifest.project_root is None):
        return args
    resolved = dict(args)
    for key in ("path", "pt2_path", "pt3_paths", "netcdf_path", "target_netcdf_path", "denominator_netcdf_path"):
        if key in resolved:
            resolved[key] = _resolve_path_container(resolved[key], manifest)
    return resolved


def filter_tool_kwargs(tool: Any, args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop LLM-supplied keys that are not in the tool signature."""
    sig = inspect.signature(tool)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return args, {}
    allowed = {
        name
        for name, p in sig.parameters.items()
        if name != "store"
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    filtered = {key: value for key, value in args.items() if key in allowed}
    dropped = {key: value for key, value in args.items() if key not in allowed}
    return filtered, dropped


def prepare_tool_args(
    tool_name: str,
    args: dict[str, Any],
    *,
    manifest: AnalysisManifest,
    stage: str,
    job: StageJob,
    effective_params: dict[str, Any],
    artifacts_dir: Path,
    store: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve paths and force plot output under ``artifacts_dir``."""
    artifacts_dir = Path(artifacts_dir)
    store = store or {}
    resolved = resolve_tool_args(args, manifest)

    if stage == "correlator_analysis":
        selected = [item for item in manifest.correlators if item.correlator_id in job.correlator_ids]
        pt2 = next((item for item in selected if item.kind == "2pt"), None)
        pt3 = sorted((item for item in selected if item.kind == "3pt"), key=lambda item: item.tsep or 0)
        defaults = dict(effective_params)
        if "component" in defaults:
            defaults["part"] = defaults.pop("component")
        defaults["resample_mode"] = manifest.metadata.resample_mode
        if pt2 is not None:
            defaults.update(
                {
                    "pt2_path": pt2.data_path,
                    "source_sink": pt2.source_sink,
                    "momentum": pt2.momentum,
                    "gamma": pt2.src_gamma,
                    "pt2_gamma": pt2.src_gamma,
                    "ensemble": pt2.ensemble,
                    "tag": job.id,
                    "hadron": pt2.hadron,
                    "gfix": pt2.gfix,
                }
            )
        if pt3:
            first = pt3[0]
            defaults.update(
                {
                    "pt3_paths": {str(item.tsep): item.data_path for item in pt3},
                    "tsep_ls": [item.tsep for item in pt3],
                    "z_values": first.bz,
                    "direction": first.z_direction,
                    "pt3_gamma": first.current_gamma,
                    "b_dir": f"b_{first.z_direction}",
                    "eta": first.eta,
                    "bt": f"bT{first.bt[0]}",
                    "b_label": f"b{first.bt[0]}",
                }
            )
        if tool_name == "tune_bare_matrix":
            if "nstate" in defaults:
                defaults["nstate_values"] = defaults.pop("nstate")
            if "fit_strategy" in defaults:
                defaults["fit_strategies"] = defaults.pop("fit_strategy")
        elif tool_name == "fit_bare_matrix_grid":
            for key in ("nstate", "fit_strategy"):
                if isinstance(defaults.get(key), list):
                    defaults.pop(key)
            defaults["save_path"] = str(artifacts_dir / job.id)
            defaults["job_id"] = job.id
            defaults["a_fm"] = pt2.a_fm if pt2 is not None else None
            defaults["pz_gev"] = pt2.pz_gev if pt2 is not None else None
        for key, value in defaults.items():
            if key not in resolved or resolved[key] is None:
                resolved[key] = value
        if tool_name == "fit_bare_matrix_grid" and "model_average" in defaults:
            resolved["model_average"] = defaults["model_average"]

    if stage == "renormalization":
        if tool_name == "apply_ratio_scheme_renormalization":
            for key, value in effective_params.items():
                if key not in resolved or resolved[key] is None:
                    resolved[key] = value
            resolved.update(
                {
                    "target": "target",
                    "denominator": "denominator",
                    "save_path": str(artifacts_dir / job.id),
                    "job_id": job.id,
                }
            )
        elif tool_name == "plot_renormalized_matrix_element":
            resolved.update({"data": "output", "save_path": str(artifacts_dir / job.id)})
    if stage == "fourier_transform":
        fourier = dict(effective_params)
        if "component" in fourier and "part" not in fourier:
            fourier["part"] = fourier.pop("component")
        source = store.get("input")
        source_metadata = source.model_dump() if isinstance(source, ArtifactInput) else getattr(source, "attrs", {})
        for key in ("a_fm", "pz_gev", "hadron", "gfix"):
            if key not in fourier and key in source_metadata:
                fourier[key] = source_metadata[key]
        if "method" not in fourier and str(fourier.get("gfix", "")).upper() in {"CG", "GI"}:
            fourier["method"] = str(fourier["gfix"]).upper()
        if "observable" not in fourier:
            target = manifest.metadata.target_observable
            parton = manifest.metadata.parton
            hadron = str(fourier.get("hadron", "")).lower()
            if target == "pdf" and hadron == "pion":
                fourier["observable"] = f"pion_{parton}_quasi_pdf"
            elif target == "da" and hadron == "pion":
                fourier["observable"] = "meson_quasi_da"
        if tool_name == "load_renormalized_matrix_element_samples":
            resolved.update({key: fourier[key] for key in _FOURIER_LOAD_KEYS if key in fourier})
            if isinstance(source, ArtifactInput):
                resolved["path"] = source.path
            elif "path" not in resolved:
                artifact_path = _declared_artifact_path(manifest, job, "input")
                if artifact_path is not None:
                    resolved["path"] = artifact_path
            if "resample_mode" not in resolved:
                resolved["resample_mode"] = manifest.metadata.resample_mode
        elif tool_name == "run_fourier_transform":
            resolved.update({key: fourier[key] for key in _FOURIER_RUN_KEYS if key in fourier})
            resolved["save_path"] = str(artifacts_dir / job.id)
            resolved.setdefault("plot_fourier", {"save_path": f"{job.id}.pdf"})
            resolved.setdefault("plot_extension", {"save_path": f"{job.id}_extension.pdf"})
            resolved.setdefault("report", {"save_path": f"{job.id}_report.md"})
        if tool_name in _FOURIER_ARTIFACT_TOOLS:
            resolved["artifacts_dir"] = str(artifacts_dir)

    if stage == "perturbative_matching":
        from lamet_agent.stages.matching.functions import resolve_kernel_id

        matching = dict(effective_params)
        declared_id = str(matching.get("kernel_id", ""))
        declaration = next((item for item in manifest.kernels if item.kernel_id == declared_id), None)
        if declaration is not None:
            parameters = dict(declaration.kernel_parameters)
            parameters.update(matching)
            matching = parameters
            matching["kernel_id"] = resolve_kernel_id(declared_id, declaration.scheme)
        if tool_name == "load_quasi_pdf":
            resolved["component"] = matching.get("component", "re")
            quasi = store.get("quasi")
            if isinstance(quasi, ArtifactInput):
                resolved["path"] = quasi.path
            elif "path" not in resolved:
                artifact_path = _declared_artifact_path(manifest, job, "quasi")
                if artifact_path is not None:
                    resolved["path"] = artifact_path
        elif tool_name == "build_matching_kernel":
            resolved.update({key: matching[key] for key in _MATCHING_KERNEL_KEYS if key in matching})
        elif tool_name == "apply_matching":
            resolved.update({"save_path": str(artifacts_dir / job.id), "artifacts_dir": str(artifacts_dir)})
        elif tool_name == "plot_matched_pdf":
            resolved.update({"save_path": str(artifacts_dir / job.id), "artifacts_dir": str(artifacts_dir)})
        elif tool_name == "report_matching_result":
            resolved.update({key: matching[key] for key in ("kernel_id", "pz_gev", "mu", "zs_fm", "component") if key in matching})
            resolved.update({"save_path": f"{job.id}_report.md", "artifacts_dir": str(artifacts_dir)})
    if tool_name in _RENORM_ARTIFACT_TOOLS:
        raw_save = resolved.get("save_path")
        if isinstance(raw_save, str) or raw_save is None:
            stem = "renormalized_matrix_element"
            default_stem = _run_scoped_plot_stem(manifest, stem)
            resolved["save_path"] = resolve_plot_save_path(
                raw_save if isinstance(raw_save, str) else None,
                artifacts_dir=artifacts_dir,
                default_stem=default_stem,
                root_directory=_manifest_root(manifest),
            )
        resolved["artifacts_dir"] = str(artifacts_dir)
    if tool_name in _PLOT_TOOLS:
        raw_save = resolved.get("save_path")
        if raw_save is None and tool_name == "fit_bare_matrix_grid":
            grid = manifest.metadata.get("correlator_grid", {})
            if isinstance(grid, dict) and isinstance(grid.get("save_path"), str):
                raw_save = grid["save_path"]
        if isinstance(raw_save, str) or raw_save is None:
            if tool_name == "fit_bare_matrix_grid":
                stem = "bare_matrix_elements"
            elif tool_name == "plot_matched_pdf":
                stem = "matched_pdf"
            else:
                stem = "fit_on_data"
            default_stem = _run_scoped_plot_stem(manifest, stem)
            resolved["save_path"] = resolve_plot_save_path(
                raw_save if isinstance(raw_save, str) else None,
                artifacts_dir=artifacts_dir,
                default_stem=default_stem,
                root_directory=_manifest_root(manifest),
            )
        resolved["artifacts_dir"] = str(artifacts_dir)
    return resolved
