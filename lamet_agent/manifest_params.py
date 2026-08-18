"""Shared evaluator and lazy routing for stage-owned manifest contracts.

The global manifest envelope remains in :mod:`lamet_agent.manifest`; each
stage's ``validation.py`` is the sole authority for its parameter subtree.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from difflib import get_close_matches
from importlib import import_module
from typing import Any, Callable

ParamSchema = dict[str, Any]


@dataclass(frozen=True)
class ListItems:
    """Apply a nested parameter schema to mapping items in a list."""

    schema: ParamSchema
    summary: str = "List of structured parameter candidates."
    physics: str = "Each list item defines one candidate evaluated by the stage."
    examples: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ParameterSpec:
    """One user-authored parameter's shape and human-facing meaning."""

    summary: str
    physics: str
    expected: type | tuple[type, ...] | None = None
    items: type | tuple[type, ...] | None = None
    choices: tuple[Any, ...] = ()
    choice_descriptions: dict[Any, str] = field(default_factory=dict)
    unit: str | None = None
    default: str | None = None
    required: bool = False
    schema: ParamSchema | None = None
    examples: tuple[Any, ...] = ()
    validator: Callable[[Any], str | None] | None = None
    suggested_fix: str = ""
    coerce_scalar_to_list: bool = False


@dataclass(frozen=True)
class StageValidationContext:
    """Resolved job view shared by validation and incomplete-draft planning."""

    stage: str
    job_id: str
    job_path: str
    params: dict[str, Any]
    inputs: dict[str, Any]
    metadata: dict[str, Any]
    resources: dict[str, Any] = field(default_factory=dict)
    authored_params: dict[str, Any] | None = None
    parameter_base_path: str | None = None

    def parameter_path(self, parameter: str) -> str:
        """Return the manifest path for one effective job parameter."""
        base = self.parameter_base_path or f"{self.job_path}.params"
        return f"{base}.{parameter}"


@dataclass(frozen=True)
class RuleViolation:
    """Context-specific evidence that one declared constraint is not satisfied."""

    message: str
    path: str
    cause: str
    parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintSpec:
    """One executable cross-parameter or contextual rule."""

    code: str
    parameters: tuple[str, ...]
    rule: str
    physics: str
    suggested_fix: str
    check: Callable[[StageValidationContext], RuleViolation | list[RuleViolation] | None] | None = None


@dataclass(frozen=True)
class StageValidationIssue:
    """Structured stage issue suitable for CLI, planning, and LLM feedback."""

    code: str
    message: str
    path: str
    cause: str
    physics: str
    suggested_fix: str
    severity: str = "error"
    parameters: tuple[str, ...] = ()

    def detailed_message(self) -> str:
        """Render the issue with its immediate and physical causes."""
        details = [self.message]
        if self.cause:
            details.append(f"Cause: {self.cause}")
        if self.physics:
            details.append(f"Physics: {self.physics}")
        return " ".join(details)


@dataclass(frozen=True)
class StageParamContract:
    """Allowed parameter shape, semantics, and migration messages for a stage."""

    schema: ParamSchema
    removed: dict[str, str]
    summary: str = ""
    physics: str = ""
    constraints: tuple[ConstraintSpec, ...] = ()
    code_prefix: str = "stage"
    planning_notes: tuple[str, ...] = ()
    input_roles: tuple[str, ...] = ()
    input_role_descriptions: dict[str, str] = field(default_factory=dict)
    job_parameters: tuple[str, ...] = ()
    normalize_draft: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None

    def evaluate(self, context: StageValidationContext) -> list[StageValidationIssue]:
        """Evaluate parameter declarations and executable constraints once."""
        return evaluate_stage_contract(self, context)


def merge_stage_params(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge job parameter overrides onto stage defaults.

    Nested mappings merge recursively; all other values, including lists, are
    replaced as complete values by the job override.
    """
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_stage_params(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


STAGE_PARAM_CONTRACTS = {
    "correlator_analysis": "lamet_agent.stages.correlator.validation:STAGE_PARAM_CONTRACT",
    "renormalization": "lamet_agent.stages.renorm.validation:STAGE_PARAM_CONTRACT",
    "fourier_transform": "lamet_agent.stages.fourier.validation:STAGE_PARAM_CONTRACT",
    "perturbative_matching": "lamet_agent.stages.matching.validation:STAGE_PARAM_CONTRACT",
    "extrapolation": "lamet_agent.stages.extrapolation.validation:STAGE_PARAM_CONTRACT",
    "review": "lamet_agent.stages.review.validation:STAGE_PARAM_CONTRACT",
}


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


def get_stage_parameter_contract(stage: str) -> StageParamContract:
    """Resolve one contract, lazily loading stage-owned declarations."""
    source = STAGE_PARAM_CONTRACTS.get(stage)
    if source is None:
        raise ValueError(f"Stage {stage!r} must be registered in STAGE_PARAM_CONTRACTS.")
    if isinstance(source, StageParamContract):
        return source
    if not isinstance(source, str) or ":" not in source:
        raise ValueError(f"Stage {stage!r} has an invalid parameter-contract registration.")
    module_name, attribute = source.split(":", 1)
    contract = getattr(import_module(module_name), attribute, None)
    if not isinstance(contract, StageParamContract):
        raise ValueError(f"Stage {stage!r} did not provide a StageParamContract at {source!r}.")
    return contract


def _expected_type_name(expected: type | tuple[type, ...]) -> str:
    types = expected if isinstance(expected, tuple) else (expected,)
    names = []
    for candidate in types:
        names.append("number" if candidate is float else candidate.__name__)
    return " or ".join(names)


def _matches_expected(value: Any, expected: type | tuple[type, ...]) -> bool:
    types = expected if isinstance(expected, tuple) else (expected,)
    for candidate in types:
        if candidate is bool and type(value) is bool:
            return True
        if candidate is int and type(value) is int:
            return True
        if candidate is float and not isinstance(value, bool) and isinstance(value, (int, float)):
            return True
        if candidate not in {bool, int, float} and isinstance(value, candidate):
            return True
    return False


def _validate_parameter_spec(value: Any, spec: ParameterSpec, path: str) -> list[str]:
    issues: list[str] = []
    explanation = f" Parameter: {spec.summary} Physics: {spec.physics}"
    if spec.expected is not None and not _matches_expected(value, spec.expected):
        return [
            f"{path} must be {_expected_type_name(spec.expected)}; "
            f"got {type(value).__name__}.{explanation}"
        ]
    values = value if isinstance(value, list) else [value]
    if spec.items is not None and isinstance(value, list):
        for index, item in enumerate(value):
            if not _matches_expected(item, spec.items):
                issues.append(
                    f"{path}[{index}] must be {_expected_type_name(spec.items)}; "
                    f"got {type(item).__name__}.{explanation}"
                )
    if spec.choices:
        for index, item in enumerate(values):
            if item not in spec.choices:
                choice_path = f"{path}[{index}]" if isinstance(value, list) else path
                issues.append(
                    f"{choice_path} must be one of {list(spec.choices)!r}; "
                    f"got {item!r}.{explanation}"
                )
    return issues


def _parameter_suggested_fix(spec: ParameterSpec) -> str:
    if spec.suggested_fix:
        return spec.suggested_fix
    if spec.choices:
        return f"Choose one of {list(spec.choices)!r}."
    if spec.examples:
        return f"For example, use {spec.examples[0]!r}."
    return "Declare a value consistent with the parameter contract."


def _structured_parameter_issues(
    *,
    contract: StageParamContract,
    context: StageValidationContext,
    value: dict[str, Any],
    schema: ParamSchema,
    relative_path: str = "",
    include_required: bool = True,
) -> list[StageValidationIssue]:
    issues: list[StageValidationIssue] = []
    for key, child_schema in schema.items():
        if isinstance(child_schema, ListItems):
            parameter = f"{relative_path}.{key}" if relative_path else key
            if key not in value:
                continue
            item = value[key]
            path = context.parameter_path(parameter)
            if not isinstance(item, list):
                issues.append(
                    StageValidationIssue(
                        code=f"{contract.code_prefix}.{parameter}.invalid",
                        message=f"{path} must be list; got {type(item).__name__}.",
                        path=path,
                        cause=f"The effective value is {item!r}.",
                        physics=child_schema.physics,
                        suggested_fix="Declare a list of objects matching the documented item fields.",
                        parameters=(parameter,),
                    )
                )
                continue
            for index, child in enumerate(item):
                if isinstance(child, dict):
                    issues.extend(
                        _structured_parameter_issues(
                            contract=contract,
                            context=context,
                            value=child,
                            schema=child_schema.schema,
                            relative_path=f"{parameter}[{index}]",
                            include_required=include_required,
                        )
                    )
            continue
        if not isinstance(child_schema, ParameterSpec):
            continue
        parameter = f"{relative_path}.{key}" if relative_path else key
        path = context.parameter_path(parameter)
        code_base = f"{contract.code_prefix}.{parameter}"
        if key not in value:
            if child_schema.required and include_required:
                issues.append(
                    StageValidationIssue(
                        code=f"{code_base}.required",
                        message=(
                            f"{context.stage} job {context.job_id!r} is missing required "
                            f"parameter {parameter}."
                        ),
                        path=path,
                        cause="The parameter is absent from the effective stage defaults and job params.",
                        physics=child_schema.physics,
                        suggested_fix=_parameter_suggested_fix(child_schema),
                        parameters=(parameter,),
                    )
                )
            continue

        item = value[key]
        validator_message = (
            child_schema.validator(item)
            if child_schema.validator is not None
            else None
        )
        static_messages = (
            [validator_message]
            if validator_message
            else _validate_parameter_spec(item, child_schema, path)
        )
        for message in static_messages:
            issues.append(
                StageValidationIssue(
                    code=f"{code_base}.invalid",
                    message=message,
                    path=path,
                    cause=f"The effective value is {item!r}.",
                    physics=child_schema.physics,
                    suggested_fix=_parameter_suggested_fix(child_schema),
                    parameters=(parameter,),
                )
            )
        if child_schema.schema is not None and isinstance(item, dict):
            issues.extend(
                _structured_parameter_issues(
                    contract=contract,
                    context=context,
                    value=item,
                    schema=child_schema.schema,
                    relative_path=parameter,
                    include_required=include_required,
                )
            )
    return issues


def _structured_unknown_parameter_issues(
    *,
    contract: StageParamContract,
    context: StageValidationContext,
    value: dict[str, Any],
    schema: ParamSchema,
    relative_path: str = "",
) -> list[StageValidationIssue]:
    """Report unsupported authored keys without treating runner-derived values as manifest input."""
    issues: list[StageValidationIssue] = []
    for key, item in value.items():
        parameter = f"{relative_path}.{key}" if relative_path else key
        child_schema = schema.get(key)
        if key not in schema:
            migration = (
                contract.removed.get(parameter)
                or contract.removed.get(key)
                or _COMMON_PARAMETER_MESSAGES.get(key)
            )
            matches = get_close_matches(key, list(schema), n=1, cutoff=0.72)
            suggestion = f"; did you mean {matches[0]!r}?" if matches else "."
            message = f"{key} {migration}" if migration else f"{parameter} is not a supported {context.stage} parameter{suggestion}"
            issues.append(
                StageValidationIssue(
                    code=f"{contract.code_prefix}.{parameter}.unsupported",
                    message=message,
                    path=context.parameter_path(parameter),
                    cause="The key was authored in stage defaults or job params but is absent from this stage contract.",
                    physics=contract.physics,
                    suggested_fix="Remove the key or replace it with the contract-supported parameter named in the migration message.",
                    parameters=(parameter,),
                )
            )
            continue
        nested_schema = (
            child_schema.schema
            if isinstance(child_schema, ParameterSpec)
            else child_schema
            if isinstance(child_schema, dict)
            else None
        )
        if nested_schema is not None and isinstance(item, dict):
            issues.extend(
                _structured_unknown_parameter_issues(
                    contract=contract,
                    context=context,
                    value=item,
                    schema=nested_schema,
                    relative_path=parameter,
                )
            )
        elif isinstance(child_schema, ListItems) and isinstance(item, list):
            for index, child in enumerate(item):
                if isinstance(child, dict):
                    issues.extend(
                        _structured_unknown_parameter_issues(
                            contract=contract,
                            context=context,
                            value=child,
                            schema=child_schema.schema,
                            relative_path=f"{parameter}[{index}]",
                        )
                    )
    return issues


def evaluate_stage_contract(
    contract: StageParamContract,
    context: StageValidationContext,
) -> list[StageValidationIssue]:
    """Evaluate one stage contract for either a final manifest or a plan draft."""
    issues = _structured_unknown_parameter_issues(
        contract=contract,
        context=context,
        value=context.authored_params if context.authored_params is not None else context.params,
        schema=contract.schema,
    )
    issues.extend(
        _structured_parameter_issues(
            contract=contract,
            context=context,
            value=context.params,
            schema=contract.schema,
        )
    )
    for constraint in contract.constraints:
        if constraint.check is None:
            continue
        result = constraint.check(context)
        violations = result if isinstance(result, list) else [result] if result is not None else []
        for violation in violations:
            issues.append(
                StageValidationIssue(
                    code=constraint.code,
                    message=violation.message,
                    path=violation.path,
                    cause=violation.cause,
                    physics=constraint.physics,
                    suggested_fix=constraint.suggested_fix,
                    parameters=violation.parameters or constraint.parameters,
                )
            )
    return issues


def _parameter_guidance(spec: ParameterSpec) -> dict[str, Any]:
    guidance: dict[str, Any] = {
        "summary": spec.summary,
        "physics": spec.physics,
        "required": spec.required,
    }
    if spec.expected is not None:
        guidance["accepted_type"] = _expected_type_name(spec.expected)
    if spec.items is not None:
        guidance["item_type"] = _expected_type_name(spec.items)
    if spec.unit is not None:
        guidance["unit"] = spec.unit
    if spec.default is not None:
        guidance["default"] = spec.default
    if spec.choices:
        guidance["choices"] = list(spec.choices)
    if spec.choice_descriptions:
        guidance["choice_descriptions"] = {
            str(key): value for key, value in spec.choice_descriptions.items()
        }
    if spec.examples:
        guidance["examples"] = list(spec.examples)
    if spec.suggested_fix:
        guidance["suggested_fix"] = spec.suggested_fix
    if spec.coerce_scalar_to_list:
        guidance["planning_coercion"] = "A scalar answer is stored as a one-item list."
    if spec.schema:
        guidance["fields"] = {
            key: _parameter_guidance(child)
            for key, child in spec.schema.items()
            if isinstance(child, ParameterSpec)
        }
    return guidance


def _list_guidance(spec: ListItems) -> dict[str, Any]:
    guidance: dict[str, Any] = {
        "summary": spec.summary,
        "physics": spec.physics,
        "required": False,
        "item_fields": {
            key: _parameter_guidance(child)
            for key, child in spec.schema.items()
            if isinstance(child, ParameterSpec)
        },
    }
    if spec.examples:
        guidance["examples"] = list(spec.examples)
    return guidance


def stage_contract_guidance(stage: str) -> dict[str, Any]:
    """Serialize one authoritative stage contract for the planning LLM."""
    contract = get_stage_parameter_contract(stage)
    return {
        "summary": contract.summary,
        "physics": contract.physics,
        "planning_notes": list(contract.planning_notes),
        "input_roles": list(contract.input_roles),
        "input_role_descriptions": dict(contract.input_role_descriptions),
        "job_parameters": list(contract.job_parameters),
        "parameters": {
            key: _parameter_guidance(spec) if isinstance(spec, ParameterSpec) else _list_guidance(spec)
            for key, spec in contract.schema.items()
            if isinstance(spec, (ParameterSpec, ListItems))
        },
        "constraints": [
            {
                "code": constraint.code,
                "parameters": list(constraint.parameters),
                "rule": constraint.rule,
                "physics": constraint.physics,
                "suggested_fix": constraint.suggested_fix,
            }
            for constraint in contract.constraints
        ],
        "removed_or_renamed_parameters": dict(contract.removed),
    }


MANIFEST_PARAMETER_MAINTENANCE_POLICY = (
    "Treat each stage's STAGE_PARAM_CONTRACT in validation.py as the sole authority "
    "for manifest stage parameters. Do not invent unsupported parameters or infer new "
    "meanings for existing ones. Any repository change that adds, removes, renames, or "
    "changes a manifest parameter must update the corresponding validation.py contract, "
    "including its human-facing summary, physical explanation, allowed values, and "
    "affected constraints."
)


def _accepted_value_text(spec: ParameterSpec) -> str:
    """Return a compact human-facing description of accepted values."""
    if spec.expected is None:
        accepted = "value"
    else:
        accepted = _expected_type_name(spec.expected)
    if spec.items is not None:
        accepted += f" containing {_expected_type_name(spec.items)} values"
    if spec.choices:
        accepted += "; choices: " + ", ".join(repr(item) for item in spec.choices)
    return accepted


def _render_parameter_help(
    lines: list[str],
    path: str,
    spec: ParameterSpec | ListItems,
    *,
    indent: str = "",
) -> None:
    """Append one parameter and its nested fields to contract help text."""
    if isinstance(spec, ListItems):
        lines.extend(
            [
                f"{indent}- {path} [optional; list of objects]",
                f"{indent}  Meaning: {spec.summary}",
                f"{indent}  Physics: {spec.physics}",
            ]
        )
        if spec.examples:
            lines.append(f"{indent}  Example: {spec.examples[0]!r}")
        for key, child in spec.schema.items():
            if isinstance(child, (ParameterSpec, ListItems)):
                _render_parameter_help(lines, f"{path}[].{key}", child, indent=indent + "  ")
        return

    attributes = ["required" if spec.required else "optional", _accepted_value_text(spec)]
    if spec.unit is not None:
        attributes.append(f"unit: {spec.unit}")
    if spec.default is not None:
        attributes.append(f"default: {spec.default}")
    lines.extend(
        [
            f"{indent}- {path} [{'; '.join(attributes)}]",
            f"{indent}  Meaning: {spec.summary}",
            f"{indent}  Physics: {spec.physics}",
        ]
    )
    if spec.choice_descriptions:
        lines.append(f"{indent}  Choice behavior:")
        for choice in spec.choices:
            description = spec.choice_descriptions.get(choice)
            if description:
                lines.append(f"{indent}    - {choice!r}: {description}")
    if spec.examples:
        lines.append(f"{indent}  Example: {spec.examples[0]!r}")
    if spec.coerce_scalar_to_list:
        lines.append(f"{indent}  Planning: a scalar answer becomes a one-item list.")
    if spec.schema:
        for key, child in spec.schema.items():
            if isinstance(child, (ParameterSpec, ListItems)):
                _render_parameter_help(lines, f"{path}.{key}", child, indent=indent + "  ")


def render_stage_contract(stage: str) -> str:
    """Render one stage-owned contract for maintainers and manifest authors."""
    contract = get_stage_parameter_contract(stage)
    lines = [stage, "=" * len(stage), "", contract.summary, "", f"Physics: {contract.physics}"]
    if contract.input_roles:
        lines.extend(["", "Input roles"])
        for role in contract.input_roles:
            description = contract.input_role_descriptions.get(role, "Required upstream role used by this stage.")
            lines.append(f"- {role}: {description}")
    if contract.planning_notes:
        lines.extend(["", "Authoring notes"])
        lines.extend(f"- {note}" for note in contract.planning_notes)
    lines.extend(["", "Parameters"])
    if not contract.schema:
        lines.append("- This stage has no user-authored stage parameters.")
    for key, spec in contract.schema.items():
        if isinstance(spec, (ParameterSpec, ListItems)):
            _render_parameter_help(lines, key, spec)
    if contract.constraints:
        lines.extend(["", "Cross-parameter and context rules"])
        for constraint in contract.constraints:
            lines.extend(
                [
                    f"- {constraint.code}: {constraint.rule}",
                    f"  Physics: {constraint.physics}",
                    f"  Fix: {constraint.suggested_fix}",
                ]
            )
    if contract.removed:
        lines.extend(["", "Removed or renamed parameters"])
        lines.extend(f"- {key}: {message}" for key, message in contract.removed.items())
    return "\n".join(lines)


def render_required_planning_prompt(stage: str, gaps: list[dict[str, Any]]) -> str:
    """Render current contract violations as one stage-scoped planning prompt."""
    details = []
    for gap in gaps:
        detail = str(gap.get("message", ""))
        if gap.get("physics"):
            detail += f" Physical reason: {gap['physics']}"
        if gap.get("suggested_fix"):
            detail += f" Suggested fix: {gap['suggested_fix']}"
        details.append(detail)
    answer_format = "Reply as a JSON object or key=value pairs, or none to keep the current manifest."
    if len(gaps) == 1 and gaps[0].get("parameter"):
        answer_format = (
            f"Reply with {gaps[0]['parameter']}=value, a JSON object, or the bare value; "
            "reply none to keep the current manifest."
        )
    return (
        f"{stage} currently violates these declared stage rules: "
        + " ".join(details)
        + " "
        + answer_format
    )


def render_optional_planning_prompt(stage: str) -> str:
    """Render optional parameters directly from one stage contract."""
    contract = get_stage_parameter_contract(stage)
    optional = [
        key
        for key, spec in contract.schema.items()
        if isinstance(spec, ListItems) or isinstance(spec, ParameterSpec) and not spec.required
    ]
    notes = " ".join(contract.planning_notes)
    parameter_text = ", ".join(optional) if optional else "none"
    return (
        f"{stage} optional parameters: {parameter_text}. {notes} "
        "Reply with values to set, or none."
    )


def validate_stage_parameter_mapping(
    stage: str,
    value: dict[str, Any],
    *,
    path: str,
) -> list[str]:
    """Return shape and value issues for one stage defaults or params mapping."""
    contract = get_stage_parameter_contract(stage)
    context = StageValidationContext(
        stage=stage,
        job_id="<parameter-mapping>",
        job_path=path,
        params=value,
        inputs={},
        metadata={},
        authored_params=value,
        parameter_base_path=path,
    )
    issues = _structured_unknown_parameter_issues(
        contract=contract,
        context=context,
        value=value,
        schema=contract.schema,
    )
    issues.extend(
        _structured_parameter_issues(
            contract=contract,
            context=context,
            value=value,
            schema=contract.schema,
            include_required=False,
        )
    )
    return [
        issue.message if issue.path in issue.message else f"{issue.path} {issue.message}"
        for issue in issues
    ]
