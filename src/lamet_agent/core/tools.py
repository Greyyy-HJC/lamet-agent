"""Stage tool-registry resolution and call preparation for the agent loop."""

from __future__ import annotations

import inspect
import logging
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from lamet_agent.manifest import AnalysisManifest, ArtifactInput, StageJob, derive_job_kinematics
from lamet_agent.manifest_params import merge_stage_params

from .stages import resolve_stage_package

_PLOT_TOOLS = frozenset({"tune_ground_state", "tune_bare_matrix", "fit_bare_matrix_grid", "plot_matched_pdf"})
_RENORM_ARTIFACT_TOOLS = frozenset(
    {
        "apply_ratio_scheme_renormalization",
        "apply_self_renormalization",
        "fit_self_renormalization_factor",
        "plot_renormalized_matrix_element",
        "plot_self_renormalization_diagnostics",
        "load_bare_matrix_element_grid",
        "load_bare_matrix_element",
    }
)
_RENORM_SELF_FIT_PARAM_KEYS = frozenset({"LambdaQCD_gev", "d", "kernel_id", "mu", "svdcut"})
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
        "y_grid",
        "scheme_scan",
        "method",
        "order",
        "observable",
        "coord_unit",
        "momentum",
        "volume",
        "bz_direction",
        "ensemble",
        "momentum_gev",
        "final_momentum_gev",
        "lattice_spacing_fm",
        "im_flip_for_ft",
        "phase_shift",
        "sector",
        "target_observable",
        "hadron",
        "psi1_flavor_class",
        "psi2_flavor_class",
        "Lambda0_gev",
        "posterior_prior_error_scale",
        "sample_error_mode",
        "part",
        "output_scale",
        "plot_fourier",
        "plot_extension",
        "report",
    }
)
_MATCHING_KERNEL_KEYS = frozenset({"kernel_id", "momentum_gev", "mu", "zs_fm", "lc_x_ls"})
_MATCHING_LOAD_KEYS = frozenset({"quasi_y_ls"})
_MATCHING_APPLY_KEYS = frozenset({"endpoint_cut"})


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


def resolve_job_tools(
    stage: str,
    job: StageJob,
    effective_params: dict[str, Any],
    *,
    stage_tools: dict[str, Callable[..., dict[str, Any]]] | None = None,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Return only the stage tools that are valid for the current job contract."""
    tools = dict(stage_tools if stage_tools is not None else resolve_stage_tools(stage))
    if stage != "renormalization":
        return tools

    scheme = effective_params.get("scheme")
    roles = set(job.inputs)
    if scheme in {"ratio", "hybrid_ratio"} and roles == {"target", "denominator"}:
        allowed = {
            "apply_ratio_scheme_renormalization",
            "plot_renormalized_matrix_element",
        }
    elif scheme == "hybrid_self_renormalization" and roles == {"reference"}:
        allowed = {
            "fit_self_renormalization_factor",
            "plot_self_renormalization_diagnostics",
        }
    elif scheme == "hybrid_self_renormalization" and roles == {"target", "zR"}:
        allowed = {
            "apply_self_renormalization",
            "plot_self_renormalization_diagnostics",
            "plot_renormalized_matrix_element",
        }
    else:
        return tools
    return {name: tool for name, tool in tools.items() if name in allowed}


def required_job_tool_sequence(
    stage: str,
    job: StageJob,
    effective_params: dict[str, Any],
) -> tuple[str, ...]:
    """Return the successful tool order required before a job may finish."""
    if stage != "renormalization":
        return ()

    scheme = effective_params.get("scheme")
    roles = set(job.inputs)
    if scheme in {"ratio", "hybrid_ratio"} and roles == {"target", "denominator"}:
        return (
            "apply_ratio_scheme_renormalization",
            "plot_renormalized_matrix_element",
        )
    if scheme == "hybrid_self_renormalization" and roles == {"reference"}:
        return (
            "fit_self_renormalization_factor",
            "plot_self_renormalization_diagnostics",
        )
    if scheme == "hybrid_self_renormalization" and roles == {"target", "zR"}:
        return (
            "apply_self_renormalization",
            "plot_self_renormalization_diagnostics",
            "plot_renormalized_matrix_element",
        )
    return ()


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
        # The correlator z grid is runner-owned input metadata, not a stage/job
        # parameter. Ignore model-supplied copies and derive it below.
        resolved.pop("z_values", None)
        selected = [item for item in manifest.correlators if item.correlator_id in job.correlator_ids]
        defaults = merge_stage_params(effective_params, job.params)
        fitting_form = str(defaults.get("fitting_form", "Breit"))
        raw_scopes = defaults.get("fit_scope", ["3pt_ratio"])
        scope_values = raw_scopes if isinstance(raw_scopes, list) else [raw_scopes]
        is_qda = [str(value) for value in scope_values] == ["qda_ratio"]
        pt2_all = [item for item in selected if item.correlator_type == "2pt"]
        pt3_all = [item for item in selected if item.correlator_type == "3pt"]
        if fitting_form == "NonBreit":
            initial_momentum = defaults.get("initial_momentum")
            final_momentum = defaults.get("final_momentum")
            selected_momentum = final_momentum
        else:
            selected_momentum = defaults.get("momentum")
            initial_momentum = final_momentum = selected_momentum

        qda_input = None
        qda_denominator_mode = "local"
        if is_qda:
            qda_candidates = [
                item
                for item in pt2_all
                if initial_momentum in item.momentum
                and item.bz is not None
                and (
                    "_nonlocal" in item.source_operator
                    or "_nonlocal" in item.sink_operator
                )
            ]
            local_candidates = [
                item
                for item in pt2_all
                if initial_momentum in item.momentum
                and "_nonlocal" not in item.source_operator
                and "_nonlocal" not in item.sink_operator
            ]
            qda_input = qda_candidates[0] if len(qda_candidates) == 1 else None
            if len(local_candidates) == 1:
                pt2 = local_candidates[0]
            elif not local_candidates and qda_input is not None:
                pt2 = qda_input
                qda_denominator_mode = "nonlocal_bz0"
            else:
                pt2 = None
        else:
            pt2 = next((item for item in pt2_all if initial_momentum in item.momentum), None)
        pt2_out = pt2 if is_qda else next((item for item in pt2_all if final_momentum in item.momentum), None)
        pt3 = [item for item in pt3_all if selected_momentum in item.momentum]
        first_pt3 = pt3[0] if pt3 else None
        if "component" in defaults:
            defaults["part"] = defaults.pop("component")
        defaults["resample_mode"] = manifest.metadata.resample_mode
        defaults["sample_error_mode"] = manifest.metadata.sample_error_mode
        defaults["seed"] = manifest.metadata.random_seed
        if manifest.metadata.resample_mode == "bs":
            if manifest.metadata.bs_samples is None:
                raise ValueError("metadata.bs_samples is required when metadata.resample_mode is 'bs'")
            defaults["n_boot"] = manifest.metadata.bs_samples
        if manifest.metadata.bin_size is not None:
            defaults["bin_size"] = manifest.metadata.bin_size
        if pt2 is not None:
            defaults.update(
                {
                    "pt2_path": pt2.data_path,
                    "pt2_out_path": pt2_out.data_path if pt2_out is not None else pt2.data_path,
                    "source_operator": pt2.source_operator,
                    "sink_operator": pt2.sink_operator,
                    "temporal_extent": pt2.temporal_extent,
                    "volume": pt2.volume,
                    "ensemble": pt2.ensemble,
                    "tag": job.id,
                    "hadron": pt2.hadron,
                    "gfix": pt2.gfix,
                }
            )
            if is_qda and qda_input is not None:
                defaults["qda_path"] = qda_input.data_path
                defaults["qda_source_operator"] = qda_input.source_operator
                defaults["qda_sink_operator"] = qda_input.sink_operator
                defaults["z_values"] = qda_input.bz
                defaults["bz_direction"] = qda_input.bz_direction
                defaults["bT"] = qda_input.bT[0]
                defaults["qda_denominator_mode"] = qda_denominator_mode
                if qda_denominator_mode == "nonlocal_bz0":
                    defaults["pt2_bT"] = qda_input.bT[0]
                    defaults["pt2_bz"] = 0
            if fitting_form == "NonBreit":
                defaults["initial_momentum"] = initial_momentum
                defaults["final_momentum"] = final_momentum
            else:
                defaults["momentum"] = selected_momentum
        if pt3:
            first = pt3[0]
            paths_by_tsep: dict[str, str] = {}
            for item in pt3:
                for tsep in item.tsep or []:
                    key = str(tsep)
                    if key in paths_by_tsep and paths_by_tsep[key] != item.data_path:
                        raise ValueError(f"multiple 3pt files declare tsep={tsep} for momentum {selected_momentum}")
                    paths_by_tsep[key] = item.data_path
            defaults.update(
                {
                    "pt3_paths": paths_by_tsep,
                    "tsep_ls": sorted(int(value) for value in paths_by_tsep),
                    "z_values": first.bz,
                    "current_operator": first.current_operator,
                    "bz_direction": first.bz_direction,
                    "bT": first.bT[0],
                }
            )
        if tool_name == "tune_bare_matrix":
            if "nstate" in defaults:
                defaults["nstate_values"] = defaults.pop("nstate")
            if "fit_strategy" in defaults:
                defaults["fit_strategies"] = defaults.pop("fit_strategy")
            if "fit_scope" in defaults:
                defaults["fit_scope_values"] = defaults.pop("fit_scope")
        elif tool_name == "fit_bare_matrix_grid":
            use_model_average = bool(defaults.get("model_average", False))
            if use_model_average and isinstance(defaults.get("nstate"), list):
                defaults["nstate_values"] = defaults.pop("nstate")
                resolved.pop("nstate", None)
            elif isinstance(defaults.get("nstate"), list):
                if "nstate" in resolved and resolved["nstate"] is not None:
                    defaults.pop("nstate")
                else:
                    defaults["nstate_values"] = defaults.pop("nstate")
            if use_model_average and isinstance(defaults.get("prior_width"), list):
                resolved["prior_width"] = defaults["prior_width"]
            for key in ("fit_strategy", "fit_scope"):
                if isinstance(defaults.get(key), list):
                    values = defaults[key]
                    if len(values) == 1:
                        defaults[key] = values[0]
                    else:
                        defaults.pop(key)
            if "pt2_window" not in resolved and "tmin" in resolved and "tmax" in resolved:
                resolved["pt2_window"] = {"tmin": int(resolved["tmin"]), "tmax": int(resolved["tmax"])}
            if "pt3_window" not in resolved and "tau_cut" in resolved:
                resolved["pt3_window"] = {
                    "tsep_ls": [int(t) for t in resolved.get("tsep_ls", defaults.get("tsep_ls", []))],
                    "tau_cut": int(resolved["tau_cut"]),
                }
            defaults["save_path"] = str(artifacts_dir / job.id)
            defaults["job_id"] = job.id
            defaults["workers"] = manifest.metadata.workers
            defaults["lattice_spacing_fm"] = pt2.lattice_spacing_fm if pt2 is not None else None
            if fitting_form == "NonBreit":
                defaults["initial_momentum_gev"] = (
                    pt2.momentum_gev(initial_momentum) if pt2 is not None and initial_momentum is not None else None
                )
                defaults["momentum_gev"] = defaults["initial_momentum_gev"]
                defaults["final_momentum_gev"] = (
                    pt2_out.momentum_gev(final_momentum)
                    if pt2_out is not None and final_momentum is not None
                    else None
                )
            else:
                defaults["momentum_gev"] = (
                    pt2.momentum_gev(selected_momentum)
                    if pt2 is not None and selected_momentum is not None
                    else None
                )
        for key, value in defaults.items():
            if key not in resolved or resolved[key] is None:
                resolved[key] = value
        if tool_name == "fit_bare_matrix_grid" and "model_average" in defaults:
            resolved["model_average"] = defaults["model_average"]

    if stage == "renormalization":
        if tool_name in {
            "apply_ratio_scheme_renormalization",
            "fit_self_renormalization_factor",
            "apply_self_renormalization",
            "plot_self_renormalization_diagnostics",
            "plot_renormalized_matrix_element",
        }:
            # These job tools have runner-owned contracts. Ignore model-supplied
            # values (including explicit nulls) and rebuild arguments below.
            resolved = {}
        scheme_parameters = effective_params.get("scheme_parameters")
        if not isinstance(scheme_parameters, dict):
            scheme_parameters = {}
        renorm_kernels = [item for item in manifest.kernels if item.stage == "renormalization"]
        kernel_id = effective_params.get("kernel_id")
        kernel_parameters: dict[str, Any] = {}
        if kernel_id is None and len(renorm_kernels) == 1:
            kernel_id = renorm_kernels[0].kernel_id
        declaration = next((item for item in renorm_kernels if item.kernel_id == kernel_id), None)
        if declaration is not None:
            kernel_parameters = dict(declaration.kernel_parameters)

        if tool_name == "apply_ratio_scheme_renormalization":
            for key, value in effective_params.items():
                if key in {"normalization", "zs_fm", "scheme_parameters"}:
                    continue
                if key not in resolved or resolved[key] is None:
                    resolved[key] = value
            if effective_params.get("scheme") == "hybrid_ratio":
                resolved["scheme_parameters"] = {
                    **scheme_parameters,
                    "zs_fm": effective_params["zs_fm"],
                }
            else:
                resolved["scheme_parameters"] = {}
            resolved.update(
                {
                    "target": "target",
                    "denominator": "denominator",
                    "save_path": str(artifacts_dir / job.id),
                    "job_id": job.id,
                    "sample_error_mode": manifest.metadata.sample_error_mode,
                }
            )
        elif tool_name == "fit_self_renormalization_factor":
            resolved["reference"] = "reference"
            resolved["save_path"] = str(artifacts_dir / job.id)
            if kernel_id is not None:
                resolved["kernel_id"] = kernel_id
            for key, value in {**kernel_parameters, **scheme_parameters}.items():
                if key in _RENORM_SELF_FIT_PARAM_KEYS:
                    resolved[key] = value
            for key in _RENORM_SELF_FIT_PARAM_KEYS:
                if key in effective_params:
                    resolved[key] = effective_params[key]
        elif tool_name == "apply_self_renormalization":
            source = store.get("target")
            source_metadata = dict(
                source.resolved_metadata if isinstance(source, ArtifactInput) else getattr(source, "attrs", {})
            )
            if isinstance(source, ArtifactInput) and source.momentum_gev is not None:
                source_metadata["momentum_gev"] = source.momentum_gev
            if source is None:
                target_ref = job.inputs.get("target")
                artifact = next(
                    (
                        item
                        for item in manifest.inputs.artifacts
                        if isinstance(target_ref, str) and item.id == target_ref
                    ),
                    None,
                )
                if artifact is not None:
                    source_metadata.update(artifact.resolved_metadata)
                    if artifact.momentum_gev is not None:
                        source_metadata["momentum_gev"] = artifact.momentum_gev
            source_ensemble = getattr(source, "ensemble", None)
            if source_ensemble is not None:
                source_metadata.setdefault("ensemble", source_ensemble.id)
            source_metadata.update(derive_job_kinematics(manifest, job))
            if "ensemble" in effective_params:
                source_metadata["ensemble"] = effective_params["ensemble"]
            resolved.update(
                {
                    "target": "target",
                    "zR": "zR",
                    "save_path": str(artifacts_dir / job.id),
                    "job_id": job.id,
                    "sample_error_mode": manifest.metadata.sample_error_mode,
                    "metadata": {
                        key: value
                        for key, value in source_metadata.items()
                        if value is not None
                    },
                }
            )
            if kernel_id is not None:
                resolved["kernel_id"] = kernel_id
            for key, value in {**kernel_parameters, **scheme_parameters}.items():
                if key in {"mu", "LambdaQCD_gev", "d", "m0_gev", "z_coverage_policy"}:
                    resolved[key] = value
            for key in ("mu", "d", "m0_gev", "z_coverage_policy", "LambdaQCD_gev"):
                if key in effective_params:
                    resolved[key] = effective_params[key]
        elif tool_name == "plot_self_renormalization_diagnostics":
            is_fit_job = set(job.inputs) == {"reference"}
            resolved.update(
                {
                    "zR": "zR",
                    "fit": "self_renorm_fit",
                    "mode": "fit" if is_fit_job else "apply",
                    "save_path": str(artifacts_dir / job.id),
                    "sample_error_mode": manifest.metadata.sample_error_mode,
                }
            )
            if not is_fit_job:
                resolved["target"] = "target"
            if kernel_id is not None:
                resolved["kernel_id"] = kernel_id
            for key, value in {**kernel_parameters, **scheme_parameters}.items():
                if key in {"mu", "LambdaQCD_gev", "z_coverage_policy"}:
                    resolved[key] = value
            for key in ("LambdaQCD_gev", "mu", "z_coverage_policy"):
                if key in effective_params:
                    resolved[key] = effective_params[key]
            if not is_fit_job:
                apply_jobs = [
                    other
                    for other in manifest.stages["renormalization"].jobs
                    if set(other.inputs) == {"target", "zR"}
                ]
                siblings = []
                for other in apply_jobs:
                    if other.id == job.id:
                        continue
                    path = artifacts_dir / f"{other.id}.nc"
                    if path.is_file():
                        siblings.append(str(path))
                # After the current apply NetCDF exists, include it so the last
                # job can overlay all lattice spacings in one discrete_effect plot.
                current_path = artifacts_dir / f"{job.id}.nc"
                if current_path.is_file():
                    siblings.append(str(current_path))
                if "sibling_artifacts" not in resolved:
                    resolved["sibling_artifacts"] = siblings
                if "include_discrete_effect" not in resolved:
                    # Emit discrete_effect once on the last apply job when all
                    # sibling apply NetCDFs (including self) are present.
                    resolved["include_discrete_effect"] = (
                        bool(apply_jobs)
                        and job.id == apply_jobs[-1].id
                        and len(siblings) >= len(apply_jobs)
                    )
        elif tool_name == "load_bare_matrix_element_grid":
            role = None
            for candidate in ("target", "denominator"):
                if isinstance(store.get(candidate), ArtifactInput):
                    role = candidate
                    break
            if role is None:
                role = "target" if "target" in job.inputs else next(iter(job.inputs), None)
            if role is not None:
                source = store.get(role)
                if isinstance(source, ArtifactInput):
                    resolved.setdefault("netcdf_path", source.path)
                else:
                    artifact_path = _declared_artifact_path(manifest, job, role)
                    if artifact_path is not None:
                        resolved.setdefault("netcdf_path", artifact_path)
                resolved.setdefault("out", role)
        elif tool_name == "load_bare_matrix_element":
            source = store.get("reference")
            if isinstance(source, ArtifactInput):
                resolved.setdefault("path", source.path)
                resolved.setdefault("netcdf_path", source.path)
            else:
                artifact_path = _declared_artifact_path(manifest, job, "reference")
                if artifact_path is not None:
                    resolved.setdefault("path", artifact_path)
                    resolved.setdefault("netcdf_path", artifact_path)
            resolved.setdefault("out", "reference")
        elif tool_name == "plot_renormalized_matrix_element":
            resolved.update(
                {
                    "data": "output",
                    "save_path": str(artifacts_dir / job.id),
                    "sample_error_mode": manifest.metadata.sample_error_mode,
                }
            )
        if tool_name in _RENORM_ARTIFACT_TOOLS:
            resolved["artifacts_dir"] = str(artifacts_dir)
    if stage == "fourier_transform":
        fourier = dict(effective_params)
        if "component" in fourier and "part" not in fourier:
            fourier["part"] = fourier.pop("component")
        source = store.get("input")
        source_metadata = dict(
            source.resolved_metadata if isinstance(source, ArtifactInput) else getattr(source, "attrs", {})
        )
        if isinstance(source, ArtifactInput) and source.momentum_gev is not None:
            source_metadata["momentum_gev"] = source.momentum_gev
        source_metadata.update(derive_job_kinematics(manifest, job))
        for key in (
            "momentum",
            "volume",
            "bz_direction",
            "ensemble",
            "lattice_spacing_fm",
            "momentum_gev",
            "final_momentum_gev",
        ):
            if key in source_metadata:
                fourier[key] = source_metadata[key]
        for key in ("hadron", "gfix"):
            if key not in fourier and key in source_metadata:
                fourier[key] = source_metadata[key]
        if "method" not in fourier and str(fourier.get("gfix", "")).upper() in {"CG", "GI"}:
            fourier["method"] = str(fourier["gfix"]).upper()
        fourier.setdefault("target_observable", manifest.metadata.target_observable)
        fourier.setdefault("sample_error_mode", manifest.metadata.sample_error_mode)
        if "observable" not in fourier:
            target = manifest.metadata.target_observable
            parton = manifest.metadata.parton
            hadron = str(fourier.get("hadron", "")).lower()
            if target == "pdf" and hadron == "pion":
                fourier["observable"] = f"pion_{parton}_quasi_pdf"
            elif target == "da" and hadron == "pion":
                fourier["observable"] = "meson_quasi_da"
            elif target == "gpd" and hadron == "pion":
                fourier["observable"] = "pion_quark_quasi_gpd"
            elif target == "gpd" and hadron in {"proton", "nucleon"}:
                fourier["observable"] = "nucleon_quark_quasi_gpd"
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
            resolved["workers"] = manifest.metadata.workers
            resolved["save_path"] = str(artifacts_dir / job.id)
            resolved.setdefault("plot_fourier", {"save_path": f"{job.id}_xdep.pdf"})
            resolved.setdefault("plot_extension", {"save_path": f"{job.id}_re.pdf"})
        if tool_name in _FOURIER_ARTIFACT_TOOLS:
            resolved["artifacts_dir"] = str(artifacts_dir)

    if stage == "perturbative_matching":
        from lamet_agent.stages.matching.functions import resolve_kernel_id

        matching = dict(effective_params)
        quasi = store.get("quasi")
        quasi_metadata = dict(
            quasi.resolved_metadata if isinstance(quasi, ArtifactInput) else getattr(quasi, "attrs", {})
        )
        if isinstance(quasi, ArtifactInput) and quasi.momentum_gev is not None:
            quasi_metadata["momentum_gev"] = quasi.momentum_gev
        quasi_metadata.update(derive_job_kinematics(manifest, job))
        for key in ("momentum", "volume", "bz_direction", "ensemble", "lattice_spacing_fm", "momentum_gev"):
            if key in quasi_metadata:
                matching[key] = quasi_metadata[key]
        declared_id = str(matching.get("kernel_id", ""))
        declaration = next((item for item in manifest.kernels if item.kernel_id == declared_id), None)
        if declaration is not None:
            parameters = {key: value for key, value in declaration.kernel_parameters.items() if key != "zs_fm"}
            parameters.update(matching)
            matching = parameters
            matching["kernel_id"] = resolve_kernel_id(declared_id, declaration.scheme)
        if tool_name == "load_quasi_pdf":
            resolved["component"] = matching.get("component", "re")
            resolved.update({key: matching[key] for key in _MATCHING_LOAD_KEYS if key in matching})
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
            resolved.update({key: matching[key] for key in _MATCHING_APPLY_KEYS if key in matching})
        elif tool_name == "plot_matched_pdf":
            resolved.update({"save_path": str(artifacts_dir / job.id), "artifacts_dir": str(artifacts_dir)})
            plot = matching.get("plot", {})
            if isinstance(plot, dict):
                resolved.update({key: plot[key] for key in ("xlim", "ylim") if key in plot})
            resolved.update({key: matching[key] for key in ("xlim", "ylim") if key in matching})
            if "sector" in matching:
                resolved["sector"] = matching["sector"]
        elif tool_name == "report_matching_result":
            resolved.update({key: matching[key] for key in ("kernel_id", "momentum_gev", "mu", "zs_fm", "component") if key in matching})
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
