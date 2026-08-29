"""Authored manifest state and validation evidence for Plan mode.

Purpose: own one reversible candidate, Issue contract packets, path discovery,
and guarded in-memory JSON patches without managing an LLM conversation.
Inputs: one parseable authored manifest document and its source/output paths.
Outputs: validator Issues, reversible edits, and an atomically saved candidate.
Example: ``PlanState(path, output, original, candidate)``.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, get_args, get_origin

from ..contract import Depends, Issue, List, Provides, Recommends, Source, Suggests, Value
from ..manifest import Manifest, _BASE_RULES, _load_stage_contract

_ALLOWED_PATCH_ROOTS = {"metadata", "stages", "systematics"}
_MISSING = object()
_PATH_PART = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")


def _json_value(value: Any) -> Any:
    if value is _MISSING:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _dotted_parts(path: str) -> list[str | int]:
    return [match.group(1) or int(match.group(2)) for match in _PATH_PART.finditer(path)]


def _get_dotted(document: Any, path: str) -> Any:
    current = document
    for part in _dotted_parts(path):
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return _MISSING
            current = current[part]
        else:
            if not isinstance(current, Mapping) or part not in current:
                return _MISSING
            current = current[part]
    return current


def _json_pointer_parts(path: str) -> list[str]:
    if path == "":
        return []
    if not path.startswith("/"):
        raise ValueError("JSON Patch paths must start with '/'")
    return [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]


def _pointer_get(document: Any, path: str) -> Any:
    current = document
    for part in _json_pointer_parts(path):
        if isinstance(current, list):
            if not part.isdigit() or int(part) >= len(current):
                return _MISSING
            current = current[int(part)]
        elif isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def _patch_parent(document: Any, path: str) -> tuple[Any, str]:
    parts = _json_pointer_parts(path)
    if not parts:
        raise ValueError("the manifest root cannot be patched directly")
    if parts[0] not in _ALLOWED_PATCH_ROOTS:
        raise ValueError(f"patch root must be one of {sorted(_ALLOWED_PATCH_ROOTS)}")
    current = document
    for part in parts[:-1]:
        if isinstance(current, list):
            if not part.isdigit() or int(part) >= len(current):
                raise ValueError(f"patch parent does not exist: {path}")
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError(f"patch parent does not exist: {path}")
    return current, parts[-1]


def apply_json_patches(document: Mapping[str, Any], patches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply add/replace/remove operations to a copy or raise without mutation."""
    candidate = copy.deepcopy(dict(document))
    if not patches:
        raise ValueError("patches must be a nonempty list")
    for patch in patches:
        if not isinstance(patch, Mapping):
            raise ValueError("every patch must be an object")
        operation = patch.get("op")
        path = patch.get("path")
        if operation not in {"add", "replace", "remove"} or not isinstance(path, str):
            raise ValueError("patches require op=add|replace|remove and a JSON Pointer path")
        parent, key = _patch_parent(candidate, path)
        if isinstance(parent, list):
            if operation == "add" and key == "-":
                parent.append(copy.deepcopy(patch.get("value")))
                continue
            if not key.isdigit():
                raise ValueError(f"list patch index must be an integer: {path}")
            index = int(key)
            if operation == "add":
                if index > len(parent):
                    raise ValueError(f"list add index is out of bounds: {path}")
                parent.insert(index, copy.deepcopy(patch.get("value")))
            elif index >= len(parent):
                raise ValueError(f"list patch index is out of bounds: {path}")
            elif operation == "replace":
                if "value" not in patch:
                    raise ValueError("replace patches require value")
                parent[index] = copy.deepcopy(patch["value"])
            else:
                parent.pop(index)
            continue
        if not isinstance(parent, dict):
            raise ValueError(f"patch parent is not an object or list: {path}")
        if operation == "add":
            if "value" not in patch:
                raise ValueError("add patches require value")
            parent[key] = copy.deepcopy(patch["value"])
        elif key not in parent:
            raise ValueError(f"patch target does not exist: {path}")
        elif operation == "replace":
            if "value" not in patch:
                raise ValueError("replace patches require value")
            parent[key] = copy.deepcopy(patch["value"])
        else:
            del parent[key]
    return candidate


def _expected_description(expected: Any) -> dict[str, Any]:
    if get_origin(expected) is Literal:
        choices = list(get_args(expected))
        return {"type": type(choices[0]).__name__ if choices else "literal", "choices": choices}
    values = expected if isinstance(expected, tuple) else (expected,)
    names = [value.__name__ if isinstance(value, type) else str(value) for value in values]
    return {"type": " or ".join(names)}


def _rule_path(rule: Any) -> str:
    if isinstance(rule, Suggests):
        return rule.target_path
    return str(getattr(rule, "path", ""))


def _rule_description(rule: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": type(rule).__name__,
        "path": _rule_path(rule),
        "physics": getattr(rule, "physics", ""),
    }
    question = getattr(rule, "question", None)
    if question:
        result["question"] = question
    if isinstance(rule, Value):
        result.update(_expected_description(rule.expected))
    elif isinstance(rule, Recommends):
        result["default"] = _json_value(rule.default)
    elif isinstance(rule, Depends):
        result["runtime_recommendation_available"] = rule.null_hook is not None
    elif isinstance(rule, List):
        result["item_name"] = rule.item
    elif isinstance(rule, Source):
        result["allowed_sources"] = [
            name
            for name, allowed in (
                ("job", rule.allow_job),
                ("file", rule.allow_file),
                ("constant", rule.allow_constant),
                ("list", rule.allow_list),
            )
            if allowed
        ]
    elif isinstance(rule, Provides):
        result["selector"] = rule.selector_path
        result["selected_value"] = rule.child
    elif isinstance(rule, Suggests):
        result["source"] = rule.source_path
    return result


def _issue_contract(issue_path: str) -> tuple[Sequence[Any], str, str]:
    stage_match = re.match(r"^stages\.([a-z][a-z0-9_]*)(?:\.(.*))?$", issue_path)
    if stage_match:
        stage_id, local = stage_match.groups()
        contract = _load_stage_contract(stage_id)
        logical = re.sub(r"^jobs\[\d+\]", "jobs.job", local or "")
        return contract.JOB_RULES, logical, stage_id
    systematics_match = re.match(r"^systematics\.([a-z][a-z0-9_]*)(?:\.(.*))?$", issue_path)
    if systematics_match:
        stage_id, logical = systematics_match.groups()
        contract = _load_stage_contract(stage_id)
        return getattr(contract, "SYSTEMATICS_RULES", ()), logical or "", stage_id
    return _BASE_RULES, issue_path, "manifest"


def _related_rules(issue_path: str) -> tuple[str, list[dict[str, Any]], list[str]]:
    try:
        rules, logical, _owner = _issue_contract(issue_path)
    except ValueError:
        return issue_path, [], []
    paths = [_rule_path(rule) for rule in rules]
    anchor = logical
    while anchor and not any(path == anchor or path.startswith(f"{anchor}.") for path in paths):
        anchor = anchor.rsplit(".", 1)[0] if "." in anchor else ""
    related = [
        _rule_description(rule)
        for rule in rules
        if not anchor
        or _rule_path(rule) == anchor
        or _rule_path(rule).startswith(f"{anchor}.")
        or (isinstance(rule, Provides) and rule.selector_path == anchor)
    ]
    parent = anchor.rsplit(".", 1)[0] if "." in anchor else ""
    allowed_children = sorted(
        {
            rule.child
            for rule in rules
            if isinstance(rule, (Depends, Recommends)) and rule.parent == parent
        }
    )
    return anchor or logical, related, allowed_children


def contract_packet(document: Mapping[str, Any], path: str) -> dict[str, Any]:
    """Describe the contract subtree for an existing or proposed manifest path."""
    anchor, rules, allowed_children = _related_rules(path)
    current = _get_dotted(document, path)
    return {
        "path": path,
        "current": {"exists": current is not _MISSING, "value": _json_value(current)},
        "contract_anchor": anchor,
        "allowed_children": allowed_children,
        "related_rules": rules,
    }


def issue_packet(document: Mapping[str, Any], issue: Issue) -> dict[str, Any]:
    """Attach current values and the complete related contract subtree to one Issue."""
    anchor, rules, allowed_children = _related_rules(issue.path)
    current = _get_dotted(document, issue.path)
    parent_path = issue.path.rsplit(".", 1)[0] if "." in issue.path else ""
    parent = _get_dotted(document, parent_path) if parent_path else document
    return {
        "issue": {
            "path": issue.path,
            "message": issue.message,
            "physics": issue.physics,
            "suggested_question": issue.question,
        },
        "current": {"exists": current is not _MISSING, "value": _json_value(current)},
        "parent": {"path": parent_path, "value": _json_value(parent)},
        "contract_anchor": anchor,
        "allowed_children": allowed_children,
        "related_rules": rules,
    }


def validate_authored_candidate(path: Path, document: Mapping[str, Any]) -> list[Issue]:
    """Validate a deep copy so successful normalization never mutates the draft."""
    return Manifest(path, copy.deepcopy(dict(document))).validate()


@dataclass
class PlanState:
    """Mutable authored candidate plus reversible edits and current Issue packets."""

    manifest_path: Path
    output_path: Path
    original: dict[str, Any]
    candidate: dict[str, Any]
    issues: list[Issue] = field(default_factory=list)
    packets: list[dict[str, Any]] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)

    def refresh(self) -> None:
        self.issues = validate_authored_candidate(self.manifest_path, self.candidate)
        self.packets = [issue_packet(self.candidate, issue) for issue in self.issues]

    def manifest_view(self, pointer: str = "") -> dict[str, Any]:
        value = _pointer_get(self.candidate, pointer)
        if value is _MISSING:
            return {"ok": False, "error": f"manifest path does not exist: {pointer}"}
        return {"ok": True, "path": pointer or "/", "value": _json_value(value)}

    def contract_view(self, path: str) -> dict[str, Any]:
        packet = contract_packet(self.candidate, path)
        return {"ok": bool(packet["related_rules"]), **packet}

    def apply(self, patches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        before = copy.deepcopy(self.candidate)
        try:
            candidate = apply_json_patches(self.candidate, patches)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        edits = []
        for patch in patches:
            path = str(patch["path"])
            edits.append(
                {
                    "path": path,
                    "before": _json_value(_pointer_get(before, path)),
                    "after": _json_value(_pointer_get(candidate, path)),
                }
            )
        self.revisions.append({"before": before, "patches": copy.deepcopy(list(patches)), "edits": edits})
        self.candidate = candidate
        self.refresh()
        return {
            "ok": True,
            "edits": edits,
            "remaining_issue_count": len(self.issues),
            "issues": self.packets,
        }

    def undo(self) -> bool:
        if not self.revisions:
            return False
        revision = self.revisions.pop()
        self.candidate = revision["before"]
        self.refresh()
        return True

    def replace_candidate(self, document: Mapping[str, Any], *, note: str) -> None:
        """Replace the draft from an explicit user edit while retaining undo state."""
        before = copy.deepcopy(self.candidate)
        candidate = copy.deepcopy(dict(document))
        self.revisions.append(
            {
                "before": before,
                "patches": [],
                "edits": [{"path": "/", "before": None, "after": None, "note": note}],
            }
        )
        self.candidate = candidate
        self.refresh()

    def find_paths(self, query: str, max_results: int = 30) -> list[str]:
        roots = [self.manifest_path.parent]
        root_value = self.candidate.get("metadata", {}).get("root_directory")
        if isinstance(root_value, str):
            root = Path(root_value).expanduser()
            root = root.resolve() if root.is_absolute() else (self.manifest_path.parent / root).resolve()
            if root not in roots and root.is_dir():
                roots.append(root)
        needle = query.lower()
        found: list[str] = []
        for root in roots:
            for directory, names, files in os.walk(root):
                names[:] = [name for name in names if not name.startswith(".") and name not in {"runs", "__pycache__"}]
                for name in [*names, *files]:
                    path = Path(directory) / name
                    relative = os.path.relpath(path, self.manifest_path.parent)
                    if needle in relative.lower() and relative not in found:
                        found.append(relative)
                        if len(found) >= max_results:
                            return found
        return found

    def save(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.candidate, indent=2, ensure_ascii=False) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.output_path.name}.", dir=self.output_path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
            os.replace(temporary, self.output_path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return self.output_path


def _default_output_path(manifest_path: Path) -> Path:
    suffix = manifest_path.suffix if manifest_path.suffix.lower() == ".json" else ".json"
    return manifest_path.with_name(f"{manifest_path.stem}.planned{suffix}")


def _acceptance_question(source: Path, target: Path) -> str:
    if target == source:
        return f"Accept this plan, overwrite {source}, and enter run mode?"
    if target.exists():
        return f"Accept this plan, overwrite existing {target}, and enter run mode?"
    return f"Accept this plan, write {target}, and enter run mode?"


__all__ = [
    "PlanState",
    "_acceptance_question",
    "_default_output_path",
    "apply_json_patches",
    "contract_packet",
    "issue_packet",
    "validate_authored_candidate",
]
