"""Stage tool-registry resolution and call preparation for the agent loop."""

from __future__ import annotations

import inspect
import logging
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from lamet_agent.manifest import AnalysisManifest, resolve_data_path

from .stages import resolve_stage_package

_PLOT_TOOLS = frozenset({"tune_ground_state", "tune_bare_matrix", "fit_bare_matrix_grid"})
_RENORM_ARTIFACT_TOOLS = frozenset({"apply_ratio_scheme_renormalization", "plot_renormalized_matrix_element"})
_RENORM_APPLY_KEYS = frozenset({"target", "denominator", "zs", "delta_m", "m0", "z0", "save_path"})
_RENORM_PLOT_KEYS = frozenset({"data", "title"})
_FOURIER_LOAD_KEYS = frozenset({"input_format", "h5_group", "coord_key", "re_key", "im_key"})
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
        "posterior_prior_error_scale",
        "fit_error_mode",
    }
)
_CORRELATOR_PT2_TOOLS = frozenset({"inspect_correlator_scale", "tune_ground_state"})
_CORRELATOR_GRID_TOOLS = frozenset({"tune_bare_matrix", "fit_bare_matrix_grid"})
_CORRELATOR_PT2_KEYS = frozenset({"pt2_path", "pt2_windows", "nstate", "svdcut", "resample_mode", "n_boot", "seed"})
_CORRELATOR_GRID_KEYS = frozenset(
    {
        "pt2_path",
        "pt3_paths",
        "tsep_ls",
        "z_values",
        "ensemble",
        "tag",
        "variant",
        "momentum",
        "direction",
        "source_sink",
        "pt2_gamma",
        "pt3_gamma",
        "b_dir",
        "eta",
        "bt",
        "b_label",
        "pt2_windows",
        "pt3_windows",
        "pt3_tau_cuts",
        "fit_strategy",
        "nstate",
        "seed",
        "svdcut",
        "part",
        "q_min",
        "output_subdir",
        "resample_mode",
        "n_boot",
        "posterior_prior_error_scale",
    }
)


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


def set_my_logger(
    log_file: str | Path,
    console_output: bool = False,
    mode: str = "w",
) -> logging.Logger:
    """Compatibility wrapper for LaMETLat-style logger setup."""
    return setup_logger(log_file, console_output=console_output, mode=mode)


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
) -> str:
    """Map any plot save_path to a stem under ``artifacts_dir``."""
    if raw:
        stem = Path(raw).name
        for suffix in (".png", ".pdf", ".svg"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if not stem:
            stem = default_stem
    else:
        stem = default_stem
    return str(artifacts_dir / stem)


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


def validate_stage_inputs(stage: str, manifest: Any) -> list[str]:
    """Return a stage's input issues via its ``validate_stage_inputs`` helper."""
    package_name = resolve_stage_package(stage)
    if not package_name:
        return []
    module = import_module(f"lamet_agent.stages.{package_name}.skills")
    validator = getattr(module, "validate_stage_inputs", None)
    return validator(manifest) if callable(validator) else []


def _resolve_one_data_path(value: str, manifest: AnalysisManifest) -> str:
    if Path(value).is_absolute():
        return value
    if manifest.manifest_dir is None or manifest.project_root is None:
        return value
    return resolve_data_path(manifest.project_root, manifest.manifest_dir, value)


def _resolve_path_container(value: Any, manifest: AnalysisManifest) -> Any:
    if isinstance(value, str):
        return _resolve_one_data_path(value, manifest)
    if isinstance(value, list):
        return [_resolve_path_container(item, manifest) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_path_container(item, manifest) for key, item in value.items()}
    return value


def resolve_tool_args(args: dict[str, Any], manifest: AnalysisManifest) -> dict[str, Any]:
    """Resolve manifest-relative file paths in tool arguments."""
    if manifest.manifest_dir is None or manifest.project_root is None:
        return args
    resolved = dict(args)
    for key in ("path", "pt2_path", "pt3_paths", "report_json", "target_report_json", "denominator_report_json"):
        if key in resolved:
            resolved[key] = _resolve_path_container(resolved[key], manifest)
    return resolved


def _fill_missing(target: dict[str, Any], defaults: dict[str, Any], keys: frozenset[str]) -> None:
    for key in keys:
        if key in defaults and (key not in target or target[key] is None):
            target[key] = defaults[key]


def _merge_correlator_grid_args(
    tool_name: str,
    args: dict[str, Any],
    manifest: AnalysisManifest,
) -> dict[str, Any]:
    grid = manifest.metadata.get("correlator_grid", {})
    if not isinstance(grid, dict):
        return args

    merged = dict(args)
    if tool_name in _CORRELATOR_PT2_TOOLS:
        _fill_missing(merged, grid, _CORRELATOR_PT2_KEYS)
        if "source_sink" not in merged or merged["source_sink"] is None:
            if "source_sink" in grid:
                merged["source_sink"] = grid["source_sink"]
        if "gamma" not in merged or merged["gamma"] is None:
            if "pt2_gamma" in grid:
                merged["gamma"] = grid["pt2_gamma"]
            elif "gamma" in grid:
                merged["gamma"] = grid["gamma"]
        if "momentum" not in merged or merged["momentum"] is None:
            if "momentum" in grid:
                merged["momentum"] = grid["momentum"]
    elif tool_name in _CORRELATOR_GRID_TOOLS:
        _fill_missing(merged, grid, _CORRELATOR_GRID_KEYS)
    return resolve_tool_args(merged, manifest)


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
    artifacts_dir: Path,
    _store: dict[str, Any],
) -> dict[str, Any]:
    """Resolve paths and force plot output under ``artifacts_dir``."""
    resolved = _merge_correlator_grid_args(tool_name, resolve_tool_args(args, manifest), manifest)
    renorm = manifest.metadata.get("renormalization", {})
    if isinstance(renorm, dict):
        if tool_name == "load_bare_matrix_element_grid":
            merged = dict(resolved)
            if "out" not in merged:
                merged["out"] = (
                    "target_bare_matrix_element"
                    if "target_bare_matrix_element" not in _store
                    else "denominator_bare_matrix_element"
                )
            if "report_json" not in merged:
                if merged.get("out") == "denominator_bare_matrix_element" and renorm.get("denominator_report_json"):
                    merged["report_json"] = renorm["denominator_report_json"]
                elif renorm.get("target_report_json"):
                    merged["report_json"] = renorm["target_report_json"]
            if "resample" not in merged and "resample" in renorm:
                merged["resample"] = renorm["resample"]
            resolved = resolve_tool_args(merged, manifest)
        elif tool_name == "apply_ratio_scheme_renormalization":
            merged = dict(resolved)
            merged.update({key: renorm[key] for key in _RENORM_APPLY_KEYS if key in renorm})
            resolved = merged
        elif tool_name == "plot_renormalized_matrix_element":
            merged = dict(resolved)
            if isinstance(renorm.get("plot"), dict):
                merged.update({key: renorm["plot"][key] for key in _RENORM_PLOT_KEYS | {"save_path"} if key in renorm["plot"]})
            resolved = merged

    fourier = manifest.metadata.get("fourier", {})
    if isinstance(fourier, dict):
        if tool_name == "load_renormalized_matrix_element_samples":
            merged = dict(resolved)
            merged.update({key: fourier[key] for key in _FOURIER_LOAD_KEYS if key in fourier})
            if "fourier_input" in manifest.metadata:
                merged["path"] = manifest.metadata["fourier_input"]
            resolved = resolve_tool_args(merged, manifest)
        elif tool_name == "run_fourier_transform":
            merged = dict(resolved)
            merged.update({key: fourier[key] for key in _FOURIER_RUN_KEYS if key in fourier})
            resolved = merged
        elif tool_name == "plot_fourier_result":
            merged = dict(resolved)
            if isinstance(fourier.get("plot_fourier"), dict):
                merged.update(fourier["plot_fourier"])
            resolved = merged
        elif tool_name == "plot_fourier_extension_quality_result":
            merged = dict(resolved)
            if isinstance(fourier.get("plot_extension"), dict):
                merged.update(fourier["plot_extension"])
            resolved = merged
    if tool_name in _RENORM_ARTIFACT_TOOLS:
        raw_save = resolved.get("save_path")
        if isinstance(raw_save, str) or raw_save is None:
            stem = "renormalized_matrix_element"
            default_stem = _run_scoped_plot_stem(manifest, stem)
            resolved["save_path"] = resolve_plot_save_path(
                raw_save if isinstance(raw_save, str) else None,
                artifacts_dir=artifacts_dir,
                default_stem=default_stem,
            )
        resolved["artifacts_dir"] = str(artifacts_dir)
    if tool_name in _PLOT_TOOLS:
        raw_save = resolved.get("save_path")
        if raw_save is None and tool_name == "fit_bare_matrix_grid":
            grid = manifest.metadata.get("correlator_grid", {})
            if isinstance(grid, dict) and isinstance(grid.get("save_path"), str):
                raw_save = grid["save_path"]
        if isinstance(raw_save, str) or raw_save is None:
            stem = "bare_matrix_elements" if tool_name == "fit_bare_matrix_grid" else "fit_on_data"
            default_stem = _run_scoped_plot_stem(manifest, stem)
            resolved["save_path"] = resolve_plot_save_path(
                raw_save if isinstance(raw_save, str) else None,
                artifacts_dir=artifacts_dir,
                default_stem=default_stem,
            )
        resolved["artifacts_dir"] = str(artifacts_dir)
    return resolved
