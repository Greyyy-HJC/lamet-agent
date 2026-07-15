"""Questions helpers for interactive planning."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .core import PlanAgentState, _stage_parameter_gaps
from .conversion import _standard_dataset_paths


def _next_questions_for_state(state: PlanAgentState) -> list[dict[str, Any]]:
    payload = state.candidate_payload
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    if "random_seed" not in metadata:
        return [
            {
                "question_id": "metadata.random_seed",
                "prompt": "metadata.random_seed is required. Which integer seed should be used?",
                "choices": [
                    {"label": "1", "value": 1984, "description": "Use 1984, matching the repository examples."},
                    {"label": "2", "value": "__custom_int__", "description": "Enter a custom positive integer seed."},
                ],
                "custom_hint": "Enter random_seed as an integer: ",
                "skip_if_present": "metadata.random_seed",
            }
        ]
    correlators = payload.get("inputs", {}).get("correlators", []) if isinstance(payload.get("inputs"), dict) else []
    if isinstance(correlators, list):
        required_by_kind = {
            "2pt": ["source_operator", "sink_operator", "volume", "lattice_spacing_fm", "momentum"],
            "3pt": ["source_operator", "sink_operator", "current_operator", "bz_direction", "volume", "lattice_spacing_fm", "momentum", "bT", "bz", "tsep"],
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
    canonical_stages = ["correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"]
    configured_stages = metadata.get("stages", [])
    configured_stage_list = [stage for stage in configured_stages if isinstance(stage, str)] if isinstance(configured_stages, list) else []
    if configured_stage_list != canonical_stages and not state.stage_completion_checked:
        return [
            {
                "question_id": "stage.add_remaining",
                "prompt": "This manifest is not a full canonical flow. Which additional stages should be added? Answer none, all, or a subset such as renormalization and fourier_transform.",
            }
        ]
    gaps = _stage_parameter_gaps(payload)
    if gaps:
        gap = gaps[0]
        if not state.parameter_completion_checked:
            return [
                {
                    "question_id": str(gap.get("question_id") or f"stage_params.{gap.get('stage')}.{gap.get('job_id')}"),
                    "prompt": f"{gap.get('message')} {gap.get('suggested_fix')} Add or adjust this setting before building manifests?",
                    "choices": [
                        {"label": "1", "value": "yes", "description": "Yes, add the missing setting."},
                        {"label": "2", "value": "no", "description": "No, keep the manifest unchanged."},
                    ],
                }
            ]
        if state.parameter_completion_requested:
            return [{"question_id": str(gap.get("path")), "prompt": f"{gap.get('message')} {gap.get('suggested_fix')}"}]
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
