"""Stage tool-registry resolution and call preparation for the agent loop."""

from __future__ import annotations

import inspect
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from lamet_agent.manifest import AnalysisManifest, resolve_data_path

from .stages import resolve_stage_package

_PLOT_TOOLS = frozenset({"plot_fit_on_data", "plot_pt3_fit_on_data"})
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
    }
)


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


def resolve_tool_args(args: dict[str, Any], manifest: AnalysisManifest) -> dict[str, Any]:
    """Resolve manifest-relative file paths in tool arguments."""
    if manifest.manifest_dir is None or manifest.project_root is None:
        return args
    resolved = dict(args)
    path_value = resolved.get("path")
    if isinstance(path_value, str) and not Path(path_value).is_absolute():
        resolved["path"] = resolve_data_path(
            manifest.project_root,
            manifest.manifest_dir,
            path_value,
        )
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
    artifacts_dir: Path,
    _store: dict[str, Any],
) -> dict[str, Any]:
    """Resolve paths and force plot output under ``artifacts_dir``."""
    resolved = resolve_tool_args(args, manifest)
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
    if tool_name in _PLOT_TOOLS:
        raw_save = resolved.get("save_path")
        if isinstance(raw_save, str) or raw_save is None:
            resolved["save_path"] = resolve_plot_save_path(
                raw_save if isinstance(raw_save, str) else None,
                artifacts_dir=artifacts_dir,
            )
        resolved["artifacts_dir"] = str(artifacts_dir)
    return resolved
