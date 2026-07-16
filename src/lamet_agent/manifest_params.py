"""Recursive contracts for user-authored stage manifest parameters."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from importlib import import_module
from typing import Any

from .stage_registry import resolve_stage_package


ParamSchema = dict[str, Any]


@dataclass(frozen=True)
class ListItems:
    """Apply a nested parameter schema to mapping items in a list."""

    schema: ParamSchema


_DERIVED_KINEMATICS_MESSAGE = (
    "is runner-derived from upstream discrete momentum, volume, and lattice_spacing_fm; "
    "remove it from stage defaults/params. For a partial run, declare momentum, volume, "
    "and lattice_spacing_fm on inputs.artifacts[]."
)
_COMMON_PARAMETER_MESSAGES = {
    key: _DERIVED_KINEMATICS_MESSAGE
    for key in (
        "a_fm",
        "bz_direction",
        "final_momentum",
        "final_momentum_gev",
        "initial_momentum",
        "initial_momentum_gev",
        "lattice_spacing_fm",
        "momentum",
        "momentum_gev",
        "pz_gev",
        "pz_out_gev",
        "volume",
    )
}
_COMMON_PARAMETER_MESSAGES.update(
    {
        "bin_size": "is run-wide; use metadata.bin_size.",
        "bs_samples": "is run-wide; use metadata.bs_samples.",
        "n_boot": "is run-wide; use metadata.bs_samples when metadata.resample_mode is 'bs'.",
        "random_seed": "is run-wide; use metadata.random_seed.",
        "resample_mode": "is run-wide; use metadata.resample_mode.",
        "sample_error_mode": "is run-wide; use metadata.sample_error_mode.",
        "seed": "is run-wide; use metadata.random_seed.",
        "workers": "is run-wide; use metadata.workers.",
    }
)


def _contract_for_stage(stage: str) -> tuple[ParamSchema, dict[str, str]]:
    package = resolve_stage_package(stage)
    if not package:
        raise ValueError(f"Registered stage {stage!r} has no package route.")
    module_name = f"lamet_agent.stages.{package}.params"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
            raise
        raise ValueError(f"Stage {stage!r} must provide parameter contract module {module_name}.") from exc
    schema = getattr(module, "MANIFEST_PARAM_SCHEMA", None)
    if not isinstance(schema, dict):
        raise ValueError(
            f"Stage {stage!r} must export a dict MANIFEST_PARAM_SCHEMA from "
            f"lamet_agent.stages.{package}.params."
        )
    removed = getattr(module, "REMOVED_MANIFEST_PARAMS", {})
    if not isinstance(removed, dict) or not all(
        isinstance(key, str) and isinstance(message, str) for key, message in removed.items()
    ):
        raise ValueError(f"Stage {stage!r} REMOVED_MANIFEST_PARAMS must map strings to strings.")
    return schema, removed


def _unknown_parameter_message(
    *,
    key: str,
    relative_path: str,
    full_path: str,
    schema: ParamSchema,
    removed: dict[str, str],
) -> str:
    migration = removed.get(relative_path) or removed.get(key) or _COMMON_PARAMETER_MESSAGES.get(key)
    if migration:
        return f"{full_path} {migration}"
    candidates = [candidate for candidate in schema if candidate != key]
    matches = get_close_matches(key, candidates, n=1, cutoff=0.72)
    suggestion = f"; did you mean {matches[0]!r}?" if matches else ""
    return f"{full_path} is not a supported stage parameter{suggestion}"


def _collect_parameter_issues(
    value: dict[str, Any],
    schema: ParamSchema,
    *,
    full_path: str,
    relative_path: str,
    removed: dict[str, str],
) -> list[str]:
    issues: list[str] = []
    for key, item in value.items():
        item_path = f"{full_path}.{key}"
        item_relative_path = f"{relative_path}.{key}" if relative_path else key
        if key not in schema:
            issues.append(
                _unknown_parameter_message(
                    key=key,
                    relative_path=item_relative_path,
                    full_path=item_path,
                    schema=schema,
                    removed=removed,
                )
            )
            continue
        child_schema = schema[key]
        if isinstance(child_schema, dict) and isinstance(item, dict):
            issues.extend(
                _collect_parameter_issues(
                    item,
                    child_schema,
                    full_path=item_path,
                    relative_path=item_relative_path,
                    removed=removed,
                )
            )
        elif isinstance(child_schema, ListItems) and isinstance(item, list):
            for index, child in enumerate(item):
                if not isinstance(child, dict):
                    continue
                issues.extend(
                    _collect_parameter_issues(
                        child,
                        child_schema.schema,
                        full_path=f"{item_path}[{index}]",
                        relative_path=f"{item_relative_path}[]",
                        removed=removed,
                    )
                )
    return issues


def validate_stage_parameter_mapping(
    stage: str,
    value: dict[str, Any],
    *,
    path: str,
) -> list[str]:
    """Return unknown-key issues for one stage defaults or params mapping."""
    schema, removed = _contract_for_stage(stage)
    return _collect_parameter_issues(
        value,
        schema,
        full_path=path,
        relative_path="",
        removed=removed,
    )
