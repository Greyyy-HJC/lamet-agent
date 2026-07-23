"""Render helpers for interactive planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .conversion import convert_correlator_h5
from .core import CorrelatorH5Mapping, PlanIssue, PlanProposal, PlanRunResult, _strict_manifest_issues, _strip_jsonc


def _missing_parameters(issues: list[PlanIssue]) -> list[str]:
    return [
        f"{issue.manifest_path}: {issue.message}"
        for issue in issues
        if issue.manifest_path.startswith("metadata.") and "Missing required" in issue.message
    ]


def _inconsistent_settings(issues: list[PlanIssue]) -> list[str]:
    return [
        f"{issue.manifest_path}: {issue.message}"
        for issue in issues
        if (
            "differs from" in issue.message
            or "Duplicate" in issue.message
            or "Unavailable upstream" in issue.message
            or "Unknown correlator" in issue.message
            or "does not exist" in issue.message
            or "Strict manifest validation" not in issue.message and issue.severity == "warning"
        )
    ]


def _render_bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- none"]


def _short_repr(value: Any, *, limit: int = 220) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _manifest_change_lines(before: Any, after: Any, *, prefix: str = "") -> list[str]:
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        lines: list[str] = []
        for key in sorted(set(before) | set(after)):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in before:
                lines.append(f"{child_prefix}: <missing> -> {_short_repr(after[key])}")
            elif key not in after:
                lines.append(f"{child_prefix}: {_short_repr(before[key])} -> <removed>")
            else:
                lines.extend(_manifest_change_lines(before[key], after[key], prefix=child_prefix))
        return lines
    return [f"{prefix}: {_short_repr(before)} -> {_short_repr(after)}"]


def _render_proposal(proposal: PlanProposal, issues: list[PlanIssue]) -> str:
    summary = proposal.report.strip()
    if "\n" in summary:
        summary = summary.splitlines()[0].strip()
    lines = [summary, "", "Missing parameters:"]
    lines.extend(_render_bullets(_missing_parameters(issues)))
    lines.extend(["", "Inconsistent settings:"])
    lines.extend(_render_bullets(_inconsistent_settings(issues)))
    lines.extend(["", "Suggested modifications:"])
    if proposal.manifest_edits:
        rendered = []
        for item in proposal.manifest_edits:
            note = f" ({item['note']})" if "note" in item else ""
            rendered.append(f"{item['path']}: {item.get('old')!r} -> {item.get('new')!r}{note}")
        lines.extend(_render_bullets(rendered))
    else:
        lines.append("- none")
    lines.extend(["", "Data conversions:"])
    conversions = [item for item in proposal.data_conversions if not item.ambiguous and item.datasets]
    if conversions:
        lines.extend(f"- {item.correlator_id}: {item.source_file} -> {item.output_file}" for item in conversions)
    else:
        lines.append("- none")
    lines.extend(["", f"Quick manifest: {proposal.quick_manifest_path}", f"Full manifest: {proposal.full_manifest_path}"])
    return "\n".join(lines)


def _render_written_summary(result: PlanRunResult) -> str:
    lines = [
        f"Wrote quick manifest: {result.quick_manifest_path}",
        f"Wrote full manifest: {result.full_manifest_path}",
    ]
    for path in result.data_files:
        lines.append(f"Wrote converted data: {path}")
    lines.extend(["", "Quick manifest changes:"])
    lines.extend(_render_bullets(result.quick_manifest_changes))
    lines.extend(["", "Full manifest changes:"])
    lines.extend(_render_bullets(result.full_manifest_changes))
    if result.issues:
        lines.extend(["", "Validation issues:"])
        lines.extend(f"- {issue.severity}: {issue.message}" for issue in result.issues)
    return "\n".join(lines)


def write_planned_outputs(
    source_payload: dict[str, Any],
    quick: dict[str, Any],
    full: dict[str, Any],
    conversions: list[CorrelatorH5Mapping],
    quick_path: Path,
    full_path: Path,
) -> PlanRunResult:
    """Apply conversions, write manifests, and run strict validation."""
    for conversion in conversions:
        if not conversion.ambiguous and conversion.datasets:
            convert_correlator_h5(conversion)
    quick_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    quick_path.write_text(json.dumps(quick, indent=2) + "\n", encoding="utf-8")
    full_path.write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    issues: list[PlanIssue] = []
    for label, path in (("quick", quick_path), ("full", full_path)):
        payload = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
        issues.extend(
            PlanIssue(issue.severity, str(path), f"Generated {label} manifest failed strict validation: {issue.message}", issue.suggested_fix)
            for issue in _strict_manifest_issues(payload, path)
        )
    return PlanRunResult(
        quick_manifest_path=str(quick_path),
        full_manifest_path=str(full_path),
        data_files=[item.output_file for item in conversions if not item.ambiguous and item.datasets],
        issues=issues,
        quick_manifest_changes=_manifest_change_lines(source_payload, quick),
        full_manifest_changes=_manifest_change_lines(source_payload, full),
    )
