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
    required: bool = True
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
class Provides:
    """Conditionally provide one real child as a selector implementation."""

    parent: str
    child: str
    selector: str
    physics: str

    @property
    def path(self) -> str:
        """Return the real path activated when this provider is selected."""
        return _path_join(self.parent, self.child)

    @property
    def selector_path(self) -> str:
        """Return the sibling selector whose value names this provider."""
        return _path_join(self.parent, self.selector)


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
    return child if not parent else f"{parent}.{child}"


def _display_path(logical_path: str, concrete_path: str) -> str:
    return concrete_path or logical_path or "<root>"


Rule = Depends | Recommends | Provides | List | Value


@dataclass
class _TraversalResult:
    issues: list[Issue]
    unresolved: list[Depends]
    applied: dict[str, Any]


def _virtual_map(rules: Sequence[Rule]) -> dict[str, str]:
    return {rule.path: rule.item for rule in rules if isinstance(rule, List)}


def _resolve(root: Any, path: str, list_rules: Mapping[str, str]) -> list[tuple[str, str, Any]]:
    """Resolve a logical path to ``(logical, concrete, value)`` tuples."""
    if path == "":
        return [("", "", root)]
    parts = path.split(".")
    resolved: list[tuple[str, str, Any]] = []

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

    walk(root, parts, "", "")
    return resolved


def _provider_index(rules: Sequence[Rule]) -> dict[str, tuple[Provides, ...]]:
    """Index providers by their selector path before graph traversal."""
    indexed: dict[str, list[Provides]] = {}
    seen: set[tuple[str, str, str]] = set()
    for rule in rules:
        if not isinstance(rule, Provides):
            continue
        identity = (rule.parent, rule.child, rule.selector)
        if identity in seen:
            raise ValueError(
                f"duplicate Provides({rule.parent!r}, {rule.child!r}, "
                f"{rule.selector!r})"
            )
        seen.add(identity)
        indexed.setdefault(rule.selector_path, []).append(rule)
    return {path: tuple(providers) for path, providers in indexed.items()}


def _walk_rules(
    document: Mapping[str, Any],
    rules: Sequence[Rule],
    *,
    complete: bool,
    apply_defaults: bool,
    validate: bool,
) -> _TraversalResult:
    """Traverse active dependencies breadth first from the contract root."""
    list_rules = _virtual_map(rules)
    providers_by_selector = _provider_index(rules)
    dependencies_by_parent: dict[str, list[Depends | Recommends]] = {}
    values_by_path: dict[str, list[Value]] = {}
    lists_by_path: dict[str, list[List]] = {}
    incoming_by_path: dict[str, list[Depends | Recommends]] = {}
    for rule in rules:
        if isinstance(rule, (Depends, Recommends)):
            dependencies_by_parent.setdefault(rule.parent, []).append(rule)
            incoming_by_path.setdefault(rule.path, []).append(rule)
        elif isinstance(rule, Value):
            values_by_path.setdefault(rule.path, []).append(rule)
        elif isinstance(rule, List):
            lists_by_path.setdefault(rule.path, []).append(rule)

    issues: list[Issue] = []
    unresolved: list[Depends] = []
    applied: dict[str, Any] = {}
    queue = deque([""])
    queued = {""}
    checked_non_mapping: set[tuple[str, str]] = set()
    owned_mappings: dict[
        tuple[str, str], tuple[Mapping[str, Any], set[str]]
    ] = {}

    def enqueue(path: str) -> None:
        if path not in queued:
            queued.add(path)
            queue.append(path)

    def declare(
        logical: str,
        concrete: str,
        parent: Mapping[str, Any],
        child: str,
    ) -> None:
        key = (concrete, logical)
        if key not in owned_mappings:
            owned_mappings[key] = (parent, set())
        owned_mappings[key][1].add(child)

    while queue:
        path = queue.popleft()
        resolved = _resolve(document, path, list_rules)
        if not resolved:
            continue

        for rule in lists_by_path.get(path, ()):
            found_list = False
            for logical, concrete, value in resolved:
                display = _display_path(logical, concrete)
                if not isinstance(value, list):
                    if validate:
                        issues.append(
                            Issue(display, "expected a list", rule.physics, None)
                        )
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
                enqueue(_path_join(rule.path, rule.item))

        for rule in values_by_path.get(path, ()):
            for logical, concrete, value in resolved:
                if not validate:
                    continue
                display = _display_path(logical, concrete)
                literal_values = (
                    get_args(rule.expected)
                    if get_origin(rule.expected) is Literal
                    else ()
                )
                if literal_values:
                    if not any(
                        type(value) is type(choice) and value == choice
                        for choice in literal_values
                    ):
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

        providers = providers_by_selector.get(path, ())
        if providers:
            allowed = tuple(rule.child for rule in providers)
            source = incoming_by_path.get(path, ())
            source_physics = (
                source[0].physics
                if source
                else "The selector must name one registered provider."
            )
            source_question = source[0].question if source else None
            for logical, concrete, selected in resolved:
                provider = next(
                    (
                        candidate
                        for candidate in providers
                        if isinstance(selected, str) and selected == candidate.child
                    ),
                    None,
                )
                if provider is None:
                    if validate:
                        choices = ", ".join(repr(choice) for choice in allowed)
                        issues.append(
                            Issue(
                                _display_path(logical, concrete),
                                f"must be provided by one of {choices}",
                                source_physics,
                                source_question,
                            )
                        )
                    continue
                for parent_logical, parent_concrete, parent in _resolve(
                    document, provider.parent, list_rules
                ):
                    if not isinstance(parent, Mapping):
                        continue
                    expected_selector = _path_join(parent_concrete, provider.selector)
                    if concrete != expected_selector:
                        continue
                    declare(
                        parent_logical,
                        parent_concrete,
                        parent,
                        provider.child,
                    )
                    if provider.child not in parent:
                        if validate and complete:
                            issues.append(
                                Issue(
                                    _path_join(parent_concrete, provider.child),
                                    f"is required when {provider.selector}={provider.child!r}",
                                    provider.physics,
                                    None,
                                )
                            )
                        continue
                    enqueue(provider.path)

        outgoing = dependencies_by_parent.get(path, ())
        if not outgoing:
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

        for rule in outgoing:
            pending_hook = False
            child_exists = False
            for logical, concrete, parent in mapping_parents:
                declare(logical, concrete, parent, rule.child)
                missing_or_null = (
                    rule.child not in parent or parent[rule.child] is None
                )
                if isinstance(rule, Recommends) and missing_or_null:
                    if apply_defaults and isinstance(parent, dict):
                        value = copy.deepcopy(rule.default)
                        parent[rule.child] = value
                        applied[rule.path] = copy.deepcopy(value)
                        missing_or_null = False
                if (
                    isinstance(rule, Depends)
                    and rule.null_hook is not None
                    and missing_or_null
                ):
                    pending_hook = True
                    continue
                if rule.child not in parent:
                    if (
                        validate
                        and complete
                        and isinstance(rule, Depends)
                        and rule.required
                    ):
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
            if pending_hook:
                if isinstance(rule, Depends) and rule not in unresolved:
                    unresolved.append(rule)
                continue
            if child_exists:
                enqueue(rule.path)

    if validate:
        for (concrete, _logical), (mapping, declared) in owned_mappings.items():
            for key in mapping:
                if key not in declared:
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
    rules: Sequence[Rule],
) -> dict[str, Any]:
    """Fill active missing or null recommendations and return applied values."""
    return _walk_rules(
        document,
        rules,
        complete=False,
        apply_defaults=True,
        validate=False,
    ).applied


def _unresolved_null_hooks(
    document: Mapping[str, Any],
    rules: Sequence[Rule],
) -> tuple[Depends, ...]:
    """Return active dependencies whose runtime hooks remain unresolved."""
    return tuple(
        _walk_rules(
            document,
            rules,
            complete=False,
            apply_defaults=False,
            validate=False,
        ).unresolved
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
    rules: Sequence[Rule],
    *,
    complete: bool = True,
) -> list[Issue]:
    """Evaluate the active dependency graph breadth first."""
    if not isinstance(document, Mapping):
        return [Issue("", "expected an object", "This contract owns an object mapping.", None)]

    candidate = copy.deepcopy(dict(document))
    return _walk_rules(
        candidate,
        rules,
        complete=complete,
        apply_defaults=True,
        validate=True,
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
    "List",
    "Value",
    "Issue",
    "CheckContext",
    "evaluate_rules",
    "evaluate_checks",
]
