"""Small manifest contract engine used by validation and planning alike.

Rules form a dependency graph.  The graph is traversed breadth first from the
contract root, so validators, defaults, and hooks are considered only after a
field has been reached through an active dependency.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence, get_args, get_origin


@dataclass(frozen=True)
class Depends:
    """Declare that a contract-owned mapping depends on one child key."""

    parent: str
    child: str
    physics: str
    question: str | None = None

    null_hook: Callable[..., object] | None = None

    @property
    def path(self) -> str:
        """Return the complete logical path owned by this dependency."""
        return _path_join(self.parent, self.child)


@dataclass(frozen=True)
class Recommends:
    """Declare a contract-owned child with a static fallback value."""

    parent: str
    child: str
    physics: str
    default: object
    question: str | None = None

    @property
    def path(self) -> str:
        """Return the complete logical path owned by this recommendation."""
        return _path_join(self.parent, self.child)


@dataclass(frozen=True)
class Suggests:
    """Copy a source mapping, then apply the target mapping as a shallow overlay."""

    parent: str
    source: str
    target: str
    physics: str
    question: str | None = None

    @property
    def source_path(self) -> str:
        """Return the optional source path relative to the owning parent."""
        return _path_join(self.parent, self.source)

    @property
    def target_path(self) -> str:
        """Return the mapping or virtual-item path filled before validation."""
        return self.target


@dataclass(frozen=True)
class Provides:
    """Activate a virtual rule branch without creating a document child."""

    parent: str
    child: str
    selector: str
    physics: str

    @property
    def path(self) -> str:
        """Return the virtual path activated when this provider is selected."""
        return _path_join(self.parent, self.child)

    @property
    def selector_path(self) -> str:
        """Return the complete logical selector path authored by the contract."""
        return self.selector


@dataclass(frozen=True)
class List:
    """Declare a list, its virtual item name, and optional list validation."""

    path: str
    item: str
    physics: str
    validator: Callable[[list[object]], bool] | None = None


@dataclass(frozen=True)
class Value:
    """Declare a type or literal value set and optional intrinsic validation."""

    path: str
    expected: Any
    physics: str
    validator: Callable[[object], bool] | None = None
    question: str | None = None


@dataclass(frozen=True)
class Source:
    """Declare one job, file, constant, or recursively listed input source."""

    path: str
    physics: str
    allow_job: bool = True
    allow_file: bool = True
    allow_constant: bool = False
    allow_list: bool = False
    question: str | None = None


@dataclass(frozen=True)
class Issue:
    """One deterministic validation or planning issue."""

    path: str
    message: str
    physics: str
    question: str | None = None


@dataclass(frozen=True)
class CheckContext:
    """Read-only values supplied to a stage physics check."""

    manifest: Mapping[str, Any]
    stage_id: str
    job_id: str | None
    params: Mapping[str, Any]
    inputs: Mapping[str, Any]
    unresolved: frozenset[str] = frozenset()


def _path_join(parent: str, child: str) -> str:
    if child == "$" or child.startswith("$."):
        return child
    return child if not parent else f"{parent}.{child}"


def _scope_path(root: str, path: str) -> str:
    if path == "$" or path.startswith("$."):
        return path
    return root if not path else _path_join(root, path)


def _scope_rules(root: str, rules: Sequence[_Rule]) -> tuple[_Rule, ...]:
    """Return contract rules rooted below one logical mapping path."""
    scoped: list[_Rule] = []
    for rule in rules:
        if isinstance(rule, Depends):
            scoped.append(
                Depends(
                    _scope_path(root, rule.parent),
                    rule.child,
                    rule.physics,
                    rule.question,
                    rule.null_hook,
                )
            )
        elif isinstance(rule, Recommends):
            scoped.append(
                Recommends(
                    _scope_path(root, rule.parent),
                    rule.child,
                    rule.physics,
                    rule.default,
                    rule.question,
                )
            )
        elif isinstance(rule, Suggests):
            scoped.append(
                Suggests(
                    _scope_path(root, rule.parent),
                    rule.source,
                    _scope_path(root, rule.target),
                    rule.physics,
                    rule.question,
                )
            )
        elif isinstance(rule, Provides):
            scoped.append(
                Provides(
                    _scope_path(root, rule.parent),
                    rule.child,
                    _scope_path(root, rule.selector),
                    rule.physics,
                )
            )
        elif isinstance(rule, List):
            scoped.append(
                List(
                    _scope_path(root, rule.path),
                    rule.item,
                    rule.physics,
                    rule.validator,
                )
            )
        elif isinstance(rule, Value):
            scoped.append(
                Value(
                    _scope_path(root, rule.path),
                    rule.expected,
                    rule.physics,
                    rule.validator,
                    rule.question,
                )
            )
        elif isinstance(rule, Source):
            scoped.append(
                Source(
                    _scope_path(root, rule.path),
                    rule.physics,
                    rule.allow_job,
                    rule.allow_file,
                    rule.allow_constant,
                    rule.allow_list,
                    rule.question,
                )
            )
    return tuple(scoped)


def _valid_job_id(value: object) -> bool:
    import re

    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value))


def stage_job_rules(
    param_rules: Sequence[_Rule],
    input_rules: Sequence[_Rule],
    job_rules: Sequence[_Rule] = (),
) -> tuple[_Rule, ...]:
    """Compose one stage contract, including optional cross-field job rules."""
    # ruff: disable[E501]
    # fmt: off
    base: tuple[_Rule, ...] = (
        Depends("", "jobs", physics="A stage declares a nonempty ordered job list."),
        List(
            "jobs",
            "job",
            physics="Stage jobs preserve authored order.",
            validator=lambda value: len(value) > 0,
        ),
        Suggests(
            "",
            "defaults",
            "jobs.job",
            physics="Authored stage defaults fill fields omitted by each job.",
        ),
        Depends("jobs.job", "id", physics="Every job has one identifier."),
        Value(
            "jobs.job.id",
            str,
            physics="Job ids use the shared graph identifier syntax.",
            validator=_valid_job_id,
        ),
        Recommends(
            "jobs.job",
            "inputs",
            physics="Jobs without dependencies use an empty input mapping.",
            default={},
        ),
        Value(
            "jobs.job.inputs",
            dict,
            physics="Job inputs form a role-to-source mapping.",
        ),
    )
    # fmt: on
    # ruff: enable[E501]

    return (
        *base,
        *_scope_rules("jobs.job", param_rules),
        *_scope_rules("jobs.job.inputs", input_rules),
        *_scope_rules("jobs.job", job_rules),
    )


def _display_path(logical_path: str, concrete_path: str) -> str:
    return concrete_path or logical_path or "<root>"


_Rule = Depends | Recommends | Suggests | Provides | List | Value | Source


@dataclass
class _TraversalResult:
    issues: list[Issue]
    unresolved: list[Depends]
    applied: dict[str, Any]


def _virtual_map(rules: Sequence[_Rule]) -> dict[str, str]:
    return {rule.path: rule.item for rule in rules if isinstance(rule, List)}


def _translate_provider_path(path: str, provider_aliases: Mapping[str, str]) -> str:
    """Collapse virtual provider prefixes to their concrete parent paths."""
    translated = path
    while True:
        matches = [
            virtual for virtual in provider_aliases if translated == virtual or translated.startswith(f"{virtual}.")
        ]
        if not matches:
            return translated
        virtual = max(matches, key=len)
        parent = provider_aliases[virtual]
        suffix = translated[len(virtual) :].removeprefix(".")
        replacement = _path_join(parent, suffix) if suffix else parent
        if replacement == translated:
            return translated
        translated = replacement


def _resolve(
    document: Any,
    path: str,
    list_rules: Mapping[str, str],
    root_document: Mapping[str, Any],
    provider_aliases: Mapping[str, str] | None = None,
) -> list[tuple[str, str, Any]]:
    """Resolve a logical path to ``(logical, concrete, value)`` tuples."""
    provider_aliases = {} if provider_aliases is None else provider_aliases

    path = _translate_provider_path(path, provider_aliases)
    list_rules = {_translate_provider_path(logical, provider_aliases): item for logical, item in list_rules.items()}
    absolute = path == "$" or path.startswith("$.")
    if path in {"", "$"}:
        value = root_document if absolute else document
        resolved = [("$" if absolute else "", "", value)]
    else:
        resolved = []
    parts = path[2:].split(".") if absolute else path.split(".")

    def walk(value: Any, remaining: list[str], logical: str, concrete: str) -> None:
        if not remaining:
            resolved.append((logical, concrete, value))
            return
        segment = remaining[0]
        if logical in list_rules and list_rules[logical] == segment:
            if not isinstance(value, list):
                return
            for index, item in enumerate(value):
                walk(item, remaining[1:], f"{logical}.{segment}", f"{concrete}[{index}]")
            return
        if isinstance(value, Mapping) and segment in value:
            next_logical = _path_join(logical, segment)
            next_concrete = _path_join(concrete, segment)
            walk(value[segment], remaining[1:], next_logical, next_concrete)

    if path not in {"", "$"}:
        walk(root_document if absolute else document, parts, "$" if absolute else "", "")
    return resolved


def _provider_index(rules: Sequence[_Rule]) -> dict[str, tuple[Provides, ...]]:
    """Index providers by their selector path before graph traversal."""
    indexed: dict[str, list[Provides]] = {}
    seen: set[tuple[str, str, str]] = set()
    for rule in rules:
        if not isinstance(rule, Provides):
            continue
        identity = (rule.parent, rule.child, rule.selector)
        if identity in seen:
            raise ValueError(f"duplicate Provides({rule.parent!r}, {rule.child!r}, {rule.selector!r})")
        seen.add(identity)
        indexed.setdefault(rule.selector_path, []).append(rule)
    return {path: tuple(providers) for path, providers in indexed.items()}


def _provider_aliases(rules: Sequence[_Rule]) -> dict[str, str]:
    """Return the unique virtual-branch-to-parent aliases in one contract."""
    aliases: dict[str, str] = {}
    for providers in _provider_index(rules).values():
        for provider in providers:
            previous = aliases.setdefault(provider.path, provider.parent)
            if previous != provider.parent:
                raise ValueError(f"conflicting virtual provider path {provider.path!r}")
    return aliases


def _path_parts(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split(".") if part)


def _suggestion_index(rules: Sequence[_Rule]) -> dict[str, Suggests]:
    """Index nonoverlapping, acyclic Suggests rules by target path."""
    suggestions = [rule for rule in rules if isinstance(rule, Suggests)]
    indexed: dict[str, Suggests] = {}
    targets: list[tuple[str, ...]] = []
    for rule in suggestions:
        target = rule.target_path
        if not target:
            raise ValueError("Suggests target must not be the contract root")
        if target in indexed:
            raise ValueError(f"duplicate Suggests target {target!r}")
        parts = _path_parts(target)
        for previous, previous_rule in zip(targets, indexed.values()):
            if parts[: len(previous)] == previous or previous[: len(parts)] == parts:
                raise ValueError(f"overlapping Suggests targets {previous_rule.target_path!r} and {target!r}")
        indexed[target] = rule
        targets.append(parts)

    adjacency = {rule.source_path: rule.target_path for rule in suggestions if rule.source_path in indexed}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(path: str) -> None:
        if path in visiting:
            raise ValueError(f"cyclic Suggests dependency at {path!r}")
        if path in visited:
            return
        visiting.add(path)
        target = adjacency.get(path)
        if target is not None:
            visit(target)
        visiting.remove(path)
        visited.add(path)

    for source in adjacency:
        visit(source)
    return indexed


def _walk_rules(
    document: Mapping[str, Any],
    rules: Sequence[_Rule],
    *,
    complete: bool,
    apply_defaults: bool,
    validate: bool,
    root_document: Mapping[str, Any] | None = None,
) -> _TraversalResult:
    """Traverse active dependencies breadth first from the contract root."""
    root_document = document if root_document is None else root_document
    list_rules = _virtual_map(rules)
    providers_by_selector = _provider_index(rules)
    provider_aliases = _provider_aliases(rules)
    potential_children_by_parent: dict[str, set[str]] = {}
    for rule in rules:
        if isinstance(rule, (Depends, Recommends)):
            concrete_parent = _translate_provider_path(rule.parent, provider_aliases)
            potential_children_by_parent.setdefault(concrete_parent, set()).add(rule.child)
    absolute_selectors_by_parent: dict[str, set[str]] = {}
    for selector, providers in providers_by_selector.items():
        if selector.startswith("$."):
            for provider in providers:
                absolute_selectors_by_parent.setdefault(provider.parent, set()).add(selector)
    suggestions_by_target = _suggestion_index(rules)
    suggestions_by_parent: dict[str, list[Suggests]] = {}
    dependencies_by_parent: dict[str, list[Depends | Recommends]] = {}
    values_by_path: dict[str, list[Value]] = {}
    sources_by_path: dict[str, list[Source]] = {}
    lists_by_path: dict[str, list[List]] = {}
    for rule in rules:
        if isinstance(rule, (Depends, Recommends)):
            dependencies_by_parent.setdefault(rule.parent, []).append(rule)
        elif isinstance(rule, Value):
            values_by_path.setdefault(rule.path, []).append(rule)
        elif isinstance(rule, List):
            lists_by_path.setdefault(rule.path, []).append(rule)
        elif isinstance(rule, Suggests):
            suggestions_by_parent.setdefault(rule.parent, []).append(rule)
        elif isinstance(rule, Source):
            sources_by_path.setdefault(rule.path, []).append(rule)

    issues: list[Issue] = []
    unresolved: list[Depends] = []
    applied: dict[str, Any] = {}
    queue = deque([("", True)])
    queued = {("", True)}
    checked_non_mapping: set[tuple[str, str]] = set()
    owned_mappings: dict[int, tuple[str, str, Mapping[str, Any], set[str]]] = {}

    def enqueue(path: str, active: bool) -> None:
        item = (path, active)
        if item not in queued:
            queued.add(item)
            queue.append(item)

    def declare(
        logical: str,
        concrete: str,
        parent: Mapping[str, Any],
        child: str,
    ) -> None:
        key = id(parent)
        if key not in owned_mappings:
            owned_mappings[key] = (
                concrete,
                logical,
                parent,
                set(),
            )
        owned_mappings[key][3].add(child)

    while queue:
        path, active = queue.popleft()
        resolved = _resolve(
            document,
            path,
            list_rules,
            root_document,
            provider_aliases,
        )
        if not resolved:
            continue

        suggestion = suggestions_by_target.get(path)
        if active and suggestion is not None:
            sources = _resolve(
                document,
                suggestion.source_path,
                list_rules,
                root_document,
                provider_aliases,
            )
            if len(sources) > 1:
                raise ValueError(f"Suggests source {suggestion.source_path!r} resolves ambiguously")
            source: Mapping[str, Any] = {}
            if sources:
                source_logical, source_concrete, source_value = sources[0]
                if not isinstance(source_value, Mapping):
                    source_key = (source_concrete, source_logical)
                    if validate and source_key not in checked_non_mapping:
                        checked_non_mapping.add(source_key)
                        issues.append(
                            Issue(
                                _display_path(source_logical, source_concrete),
                                "expected an object",
                                suggestion.physics,
                                suggestion.question,
                            )
                        )
                    source_value = {}
                source = source_value
            for logical, concrete, target in resolved:
                if not isinstance(target, dict):
                    target_key = (concrete, logical)
                    if validate and target_key not in checked_non_mapping:
                        checked_non_mapping.add(target_key)
                        issues.append(
                            Issue(
                                _display_path(logical, concrete),
                                "expected an object",
                                suggestion.physics,
                                suggestion.question,
                            )
                        )
                    continue
                merged = copy.deepcopy(dict(source))
                merged.update(copy.deepcopy(target))
                target.clear()
                target.update(merged)

        for selector in absolute_selectors_by_parent.get(path, ()):
            enqueue(selector, active)

        for rule in lists_by_path.get(path, ()):
            found_list = False
            for logical, concrete, value in resolved:
                display = _display_path(logical, concrete)
                if not isinstance(value, list):
                    if validate:
                        issues.append(Issue(display, "expected a list", rule.physics, None))
                    continue
                found_list = True
                if validate and rule.validator is not None and not rule.validator(value):
                    issues.append(
                        Issue(
                            display,
                            "failed its intrinsic value check",
                            rule.physics,
                            None,
                        )
                    )
            if found_list:
                enqueue(_path_join(rule.path, rule.item), active)

        for rule in values_by_path.get(path, ()):
            for logical, concrete, value in resolved:
                if not validate:
                    continue
                display = _display_path(logical, concrete)
                literal_values = get_args(rule.expected) if get_origin(rule.expected) is Literal else ()
                if literal_values:
                    if not any(type(value) is type(choice) and value == choice for choice in literal_values):
                        choices = ", ".join(repr(choice) for choice in literal_values)
                        issues.append(
                            Issue(
                                display,
                                f"must be one of {choices}",
                                rule.physics,
                                rule.question,
                            )
                        )
                        continue
                elif not _is_expected(value, rule.expected):
                    expected = _expected_name(rule.expected)
                    issues.append(
                        Issue(
                            display,
                            f"expected {expected}",
                            rule.physics,
                            rule.question,
                        )
                    )
                    continue
                if rule.validator is not None and not rule.validator(value):
                    issues.append(
                        Issue(
                            display,
                            "failed its intrinsic value check",
                            rule.physics,
                            rule.question,
                        )
                    )

        def validate_source(value: Any, rule: Source, display: str) -> None:
            if isinstance(value, str) and rule.allow_job:
                if not _valid_job_id(value):
                    issues.append(
                        Issue(
                            display,
                            "job source has an invalid identifier",
                            rule.physics,
                            rule.question,
                        )
                    )
                return
            if (
                isinstance(value, Mapping)
                and rule.allow_file
                and set(value) == {"file"}
                and isinstance(value["file"], str)
            ):
                return
            if isinstance(value, (int, float)) and not isinstance(value, bool) and rule.allow_constant:
                return
            if isinstance(value, list) and rule.allow_list:
                if not value:
                    issues.append(
                        Issue(
                            display,
                            "source list must be nonempty",
                            rule.physics,
                            rule.question,
                        )
                    )
                for index, item in enumerate(value):
                    validate_source(item, rule, f"{display}[{index}]")
                return
            issues.append(
                Issue(
                    display,
                    "is not an allowed input source",
                    rule.physics,
                    rule.question,
                )
            )

        if validate:
            for rule in sources_by_path.get(path, ()):
                for logical, concrete, value in resolved:
                    validate_source(value, rule, _display_path(logical, concrete))

        providers = providers_by_selector.get(path, ())
        if providers:
            for logical, concrete, selected in resolved:
                for provider in providers:
                    for parent_logical, parent_concrete, parent in _resolve(
                        document,
                        provider.parent,
                        list_rules,
                        root_document,
                        provider_aliases,
                    ):
                        if not isinstance(parent, Mapping):
                            continue
                        matched = isinstance(selected, str) and selected == provider.child
                        if active and matched:
                            enqueue(provider.path, True)

        outgoing = dependencies_by_parent.get(path, ())
        suggestions = suggestions_by_parent.get(path, ())
        if not outgoing and not suggestions:
            continue
        mapping_parents: list[tuple[str, str, Mapping[str, Any]]] = []
        for logical, concrete, parent in resolved:
            object_key = (concrete, logical)
            if not isinstance(parent, Mapping):
                if validate and object_key not in checked_non_mapping:
                    checked_non_mapping.add(object_key)
                    rule = outgoing[0]
                    issues.append(
                        Issue(
                            _display_path(logical, concrete),
                            "expected an object",
                            rule.physics,
                            rule.question,
                        )
                    )
                continue
            mapping_parents.append((logical, concrete, parent))

        for suggestion_rule in suggestions:
            for logical, concrete, parent in mapping_parents:
                declare(
                    logical,
                    concrete,
                    parent,
                    suggestion_rule.source,
                )

        for rule in outgoing:
            pending_hook = False
            child_exists = False
            for logical, concrete, parent in mapping_parents:
                declare(logical, concrete, parent, rule.child)
                missing_or_null = rule.child not in parent or parent[rule.child] is None
                if active and isinstance(rule, Recommends) and missing_or_null:
                    if apply_defaults and isinstance(parent, dict):
                        value = copy.deepcopy(rule.default)
                        parent[rule.child] = value
                        concrete_rule_path = _translate_provider_path(rule.path, provider_aliases)
                        applied[concrete_rule_path] = copy.deepcopy(value)
                        missing_or_null = False
                if isinstance(rule, Depends) and active and rule.null_hook is not None and missing_or_null:
                    pending_hook = True
                    continue
                if rule.child not in parent:
                    if validate and active and complete and isinstance(rule, Depends):
                        issues.append(
                            Issue(
                                _path_join(concrete, rule.child),
                                "is required",
                                rule.physics,
                                rule.question,
                            )
                        )
                    continue
                child_exists = True
            if active and pending_hook:
                if isinstance(rule, Depends) and rule not in unresolved:
                    unresolved.append(rule)
                continue
            if child_exists:
                enqueue(rule.path, active)

    if validate:
        for concrete, logical, mapping, declared in owned_mappings.values():
            for key in list(mapping):
                if key not in declared:
                    if isinstance(mapping, dict) and key in potential_children_by_parent.get(logical, set()):
                        del mapping[key]
                        continue
                    issues.append(
                        Issue(
                            _path_join(concrete, str(key)),
                            f"unknown key '{key}'",
                            "Remove the undeclared field or add it to the owning contract.",
                            None,
                        )
                    )
    return _TraversalResult(issues, unresolved, applied)


def _apply_recommended_defaults(
    document: dict[str, Any],
    rules: Sequence[_Rule],
    *,
    root_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill active missing or null recommendations and return applied values."""
    return _walk_rules(
        document,
        rules,
        complete=False,
        apply_defaults=True,
        validate=False,
        root_document=root_document,
    ).applied


def _unresolved_null_hooks(
    document: Mapping[str, Any],
    rules: Sequence[_Rule],
    *,
    root_document: Mapping[str, Any] | None = None,
) -> tuple[Depends, ...]:
    """Return active dependencies whose runtime hooks remain unresolved."""
    unresolved = _walk_rules(
        document,
        rules,
        complete=False,
        apply_defaults=False,
        validate=False,
        root_document=root_document,
    ).unresolved
    aliases = _provider_aliases(rules)
    return tuple(
        Depends(
            _translate_provider_path(rule.parent, aliases),
            rule.child,
            rule.physics,
            rule.question,
            rule.null_hook,
        )
        for rule in unresolved
    )


def _is_expected(value: Any, expected: type | tuple[type, ...]) -> bool:
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    if expected_types == (float,):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if int in expected_types and isinstance(value, bool):
        expected_types = tuple(candidate for candidate in expected_types if candidate is not int)
    return bool(expected_types) and isinstance(value, expected_types)


def evaluate_rules(
    document: Any,
    rules: Sequence[_Rule],
    *,
    complete: bool = True,
    root_document: Mapping[str, Any] | None = None,
) -> list[Issue]:
    """Evaluate the active dependency graph breadth first."""
    if not isinstance(document, Mapping):
        return [Issue("", "expected an object", "This contract owns an object mapping.", None)]

    return _walk_rules(
        document,
        rules,
        complete=complete,
        apply_defaults=True,
        validate=True,
        root_document=root_document,
    ).issues


def _expected_name(expected: type | tuple[type, ...]) -> str:
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    return " or ".join(candidate.__name__ for candidate in expected_types)


def evaluate_checks(
    checks: Sequence[Callable[[CheckContext], Issue | Sequence[Issue] | None]],
    context: CheckContext,
) -> list[Issue]:
    """Collect issues returned by ordinary stage-owned physics checks."""
    issues: list[Issue] = []
    for check in checks:
        result = check(context)
        if isinstance(result, Issue):
            issues.append(result)
        elif result is not None:
            if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
                raise TypeError(f"contract check {check.__name__} returned an invalid result")
            if not all(isinstance(issue, Issue) for issue in result):
                raise TypeError(f"contract check {check.__name__} returned a non-Issue value")
            issues.extend(result)
    return issues


__all__ = [
    "Depends",
    "Provides",
    "Recommends",
    "Suggests",
    "List",
    "Value",
    "Source",
    "Issue",
    "CheckContext",
    "evaluate_rules",
    "evaluate_checks",
    "stage_job_rules",
]
