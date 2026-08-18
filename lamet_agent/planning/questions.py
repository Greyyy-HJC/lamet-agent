"""Questions helpers for interactive planning."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from lamet_agent.manifest_params import render_required_planning_prompt

from .core import (
    PlanAgentState,
    _manifest_root,
    _resolve_manifest_path,
    _stage_parameter_gaps,
)
from .conversion import _standard_dataset_paths


def _stage_required_prompt(
    stage: str,
    payload: dict[str, Any],
    gaps: list[dict[str, Any]],
) -> str:
    del payload
    return render_required_planning_prompt(
        stage,
        [gap for gap in gaps if gap.get("stage") == stage],
    )


def _next_path_repair_question(state: PlanAgentState) -> dict[str, Any] | None:
    """Return the next invalid input path to repair after a run fallback."""
    if state.path_repair_project_root is None:
        return None
    expected_root = state.path_repair_project_root.expanduser().resolve()
    current_root = _manifest_root(state.manifest_path, state.candidate_payload)
    if current_root != expected_root:
        return {
            "question_id": "metadata.root_directory",
            "prompt": (
                "metadata.root_directory must be the lamet-agent project root. "
                f"Use {expected_root}?"
            ),
            "choices": [
                {
                    "label": "1",
                    "value": str(expected_root),
                    "description": f"Set metadata.root_directory to {expected_root}.",
                }
            ],
        }

    inputs = state.candidate_payload.get("inputs")
    if not isinstance(inputs, dict):
        return None
    path_groups = (
        ("correlators", "data_path", "correlator data"),
        ("artifacts", "path", "external artifact"),
        ("kernels", "kernel_path", "kernel"),
    )
    for collection, field, label in path_groups:
        items = inputs.get(collection)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            value = item.get(field)
            resolved = _resolve_manifest_path(state.manifest_path, state.candidate_payload, value)
            if resolved is not None and resolved.is_file():
                continue
            display = str(resolved) if resolved is not None else repr(value)
            return {
                "question_id": f"inputs.{collection}.{index}.{field}",
                "prompt": (
                    f"The {label} path inputs.{collection}[{index}].{field} is not an existing file: "
                    f"{display}. Enter the correct path."
                ),
            }
    return None


def _next_questions_for_state(state: PlanAgentState) -> list[dict[str, Any]]:
    payload = state.candidate_payload
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    path_question = _next_path_repair_question(state)
    if path_question is not None:
        return [path_question]
    missing_metadata = [key for key in ("random_seed", "resample_mode", "sample_error_mode") if key not in metadata]
    if missing_metadata:
        return [
            {
                "question_id": "metadata.required",
                "prompt": (
                    "metadata required choices: random_seed is a positive integer; "
                    "resample_mode options are jk/jackknife or bs/bootstrap. "
                    "sample_error_mode options are mean, median, or covariance. "
                    'Reply as JSON or key=value pairs, for example {"random_seed": 1984, "resample_mode": "jk", "sample_error_mode": "covariance"}.'
                ),
            }
        ]
    correlators = payload.get("inputs", {}).get("correlators", []) if isinstance(payload.get("inputs"), dict) else []
    if isinstance(correlators, list):
        required_by_kind = {
            "2pt": ["source_operator", "sink_operator", "volume", "lattice_spacing_fm", "momentum"],
            "3pt": ["source_operator", "sink_operator", "current_operator", "polarization", "bz_direction", "volume", "lattice_spacing_fm", "momentum", "bT", "bz", "tsep"],
        }
        for index, item in enumerate(correlators):
            if not isinstance(item, dict):
                continue
            kind = str(item.get("correlator_type", ""))
            for field_name in required_by_kind.get(kind, []):
                if field_name not in item:
                    label = str(item.get("correlator_id", index))
                    examples = {
                        "source_operator": "g5",
                        "sink_operator": "g5",
                        "current_operator": "gT_nonlocal",
                        "polarization": "unpolarized",
                        "bz_direction": "Z",
                        "volume": "S48T64",
                        "lattice_spacing_fm": "0.0574",
                        "momentum": '["PX0PY0PZ0"]',
                        "bT": "[0]",
                        "bz": "[0]",
                        "tsep": "[8]",
                    }
                    return [
                        {
                            "question_id": f"inputs.correlators.{index}.{field_name}",
                            "prompt": f"The {kind} correlator {label!r} is missing {field_name}. Please provide one value, for example {examples.get(field_name, 'a valid value')}.",
                        }
                    ]
    configured_stages = metadata.get("stages", [])
    configured_stage_list = [stage for stage in configured_stages if isinstance(stage, str)] if isinstance(configured_stages, list) else []
    stages_config = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
    unused_stages = [stage for stage in stages_config if isinstance(stage, str) and stage not in configured_stage_list]
    if unused_stages:
        stage = unused_stages[0]
        return [
            {
                "question_id": f"stage.unused.{stage}",
                "prompt": (
                    f"Stage `{stage}` is configured under stages but is not listed in metadata.stages, "
                    "so it will not run. Include it in the run, or remove the unused configuration?"
                ),
                "choices": [
                    {"label": "1", "value": "include", "description": f"Include `{stage}` in metadata.stages."},
                    {"label": "2", "value": "remove", "description": f"Remove unused stages.{stage}."},
                ],
            }
        ]
    gaps = _stage_parameter_gaps(payload, state.manifest_path)
    gap_stages = {str(gap.get("stage")) for gap in gaps}
    for stage in configured_stage_list:
        if stage not in state.stage_required_checked and stage in gap_stages:
            return [
                {
                    "question_id": f"stage_required.{stage}",
                    "prompt": _stage_required_prompt(stage, payload, gaps),
                }
            ]
        if stage not in state.stage_required_checked:
            state.stage_required_checked.add(stage)
    if gaps:
        gap = gaps[0]
        if not state.parameter_completion_checked:
            physics = f" Physical reason: {gap.get('physics')}" if gap.get("physics") else ""
            return [
                {
                    "question_id": str(gap.get("question_id") or f"stage_params.{gap.get('stage')}.{gap.get('job_id')}"),
                    "prompt": f"{gap.get('message')}{physics} {gap.get('suggested_fix')} Add or adjust this setting before building manifests?",
                    "choices": [
                        {"label": "1", "value": "yes", "description": "Yes, add the missing setting."},
                        {"label": "2", "value": "no", "description": "No, keep the manifest unchanged."},
                    ],
                }
            ]
        if state.parameter_completion_requested:
            physics = f" Physical reason: {gap.get('physics')}" if gap.get("physics") else ""
            return [{"question_id": str(gap.get("path")), "prompt": f"{gap.get('message')}{physics} {gap.get('suggested_fix')}"}]
    return []


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
        if (
            question_id in {"stage.add_remaining"}
            or question_id.startswith("stage_params.")
            or question_id.startswith("stage_required.")
            or question_id.startswith("stage_optional.")
            or question_id.startswith("stage.unused.")
        ):
            return question_id
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
    if (
        question_id.endswith(".lattice_spacing_fm")
        or question_id.endswith(".zs_fm")
    ):
        return float(value)
    if question_id.endswith(".data_path"):
        text = str(value).strip()
        if not re.search(r"\.(h5|hdf5|npy|npz|nc)$", text, flags=re.I):
            raise ValueError("data_path must point to a supported data file.")
        return text
    if question_id.endswith(".momentum"):
        if isinstance(value, list):
            return [str(item) for item in value]
        text = str(value).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in re.split(r"[,，\s]+", text) if part.strip()]
        if not isinstance(parsed, list):
            parsed = [parsed]
        return [str(item) for item in parsed]
    if question_id.endswith(".bT") or question_id.endswith(".bz") or question_id.endswith(".tsep"):
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
    if question_id.endswith(".scheme_parameters"):
        if isinstance(value, dict):
            return value
        text = str(value).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"zs_fm": float(text)}
        return parsed if isinstance(parsed, dict) else {"zs_fm": float(parsed)}
    return value
