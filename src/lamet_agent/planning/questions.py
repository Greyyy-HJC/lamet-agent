"""Questions helpers for interactive planning."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .core import PlanAgentState, _stage_parameter_gaps
from .conversion import _standard_dataset_paths


def _stage_required_prompt(stage: str, payload: dict[str, Any]) -> str:
    target = str(payload.get("metadata", {}).get("target_observable", "pdf")) if isinstance(payload.get("metadata"), dict) else "pdf"
    if stage == "correlator_analysis":
        return (
            "correlator_analysis required choices: fit_scope options are 3pt_ratio, qda_ratio, FH, 3pt_ratio+FH; "
            "fitting_form options are Breit or NonBreit; NonBreit also needs initial_momentum and final_momentum. "
            "Reply as a JSON object or key=value pairs, or none to keep the current manifest."
        )
    if stage == "renormalization":
        return (
            "renormalization required choices: scheme options are ratio, hybrid_ratio, hybrid_self_renormalization. "
            "ratio/hybrid_ratio need target and denominator inputs; hybrid_ratio also needs zs_fm. "
            "hybrid_self_renormalization needs reference fit input or target plus zR inputs, LambdaQCD_gev, and fit parameter d. "
            "Reply as a JSON object or key=value pairs, or none to keep the current manifest."
        )
    if stage == "fourier_transform":
        sectors = "full" if target == "da" else "valence, total, full, sea"
        return (
            "fourier_transform required choices: input must be one renormalized job or artifact; "
            "order options are LA, NLA, or both; sector options are "
            f"{sectors}; part options are re, im, both. "
            "y_grid is required and may be a list or {start, stop, num}. "
            "coord_unit options are lattice, fm, gev_inv, lambda if not known from the input artifact. "
            "Reply as a JSON object or key=value pairs, or none to keep the current manifest."
        )
    if stage == "perturbative_matching":
        return (
            "perturbative_matching required choices: quasi input must be one Fourier job or artifact; "
            "kernel_id must select one declared kernel; component options are re or im; hybrid kernels need zs_fm. "
            "Reply as a JSON object or key=value pairs, or none to keep the current manifest."
        )
    if stage == "extrapolation":
        return (
            "extrapolation required choices: inputs.lightcone must list the matching jobs or artifacts to fit. "
            "If only one ensemble and one momentum are available the stage cannot run. "
            "Reply as a JSON object or key=value pairs, or none to keep the current manifest."
        )
    return "review has no required parameters. Reply none to continue."


def _stage_optional_prompt(stage: str, payload: dict[str, Any]) -> str:
    target = str(payload.get("metadata", {}).get("target_observable", "pdf")) if isinstance(payload.get("metadata"), dict) else "pdf"
    if stage == "correlator_analysis":
        return (
            "correlator_analysis optional choices: pt2_windows, pt3_windows, pt3_tau_cuts, nstate, prior_width, "
            "fit_strategy, model_average, component. Reply with values to set, or none to let run/stage decide."
        )
    if stage == "renormalization":
        return (
            "renormalization optional choices: normalization, scheme_parameters such as m0_gev and delta_m_gev for ratio branches, "
            "or z_coverage_policy and svdcut for hybrid_self_renormalization. Reply with values to set, or none."
        )
    if stage == "fourier_transform":
        sector_text = "sector is fixed to full for DA" if target == "da" else "sector options are valence, total, full, sea"
        return (
            "fourier_transform optional choices: scheme_scan, posterior_prior_error_scale, plot names, x/y limits, "
            f"method, observable; {sector_text}. Reply with values to set, or none."
        )
    if stage == "perturbative_matching":
        return (
            "perturbative_matching optional choices: mu, matching grids, x/y limits, sector, plot settings. "
            "Reply with values to set, or none."
        )
    if stage == "extrapolation":
        return (
            "extrapolation optional choices: allow_order_a, allow_order_1overp, allow_order_ap, fitting_param_xdep, pdep_gev. "
            "Reply with values to set, or none."
        )
    return "review has no optional manifest parameters. Reply none to continue."


def _next_questions_for_state(state: PlanAgentState) -> list[dict[str, Any]]:
    payload = state.candidate_payload
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    missing_metadata = [key for key in ("random_seed", "resample_mode") if key not in metadata]
    if missing_metadata:
        return [
            {
                "question_id": "metadata.required",
                "prompt": (
                    "metadata required choices: random_seed is a positive integer; "
                    "resample_mode options are jk/jackknife or bs/bootstrap. "
                    'Reply as JSON or key=value pairs, for example {"random_seed": 1984, "resample_mode": "jk"}.'
                ),
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
    gaps = _stage_parameter_gaps(payload, state.manifest_path)
    gap_stages = {str(gap.get("stage")) for gap in gaps}
    for stage in configured_stage_list:
        if stage not in state.stage_required_checked and stage in gap_stages:
            return [{"question_id": f"stage_required.{stage}", "prompt": _stage_required_prompt(stage, payload)}]
        if stage not in state.stage_required_checked:
            state.stage_required_checked.add(stage)
        if stage not in state.stage_optional_checked:
            return [{"question_id": f"stage_optional.{stage}", "prompt": _stage_optional_prompt(stage, payload)}]
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
        if (
            question_id in {"stage.add_remaining"}
            or question_id.startswith("stage_params.")
            or question_id.startswith("stage_required.")
            or question_id.startswith("stage_optional.")
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
