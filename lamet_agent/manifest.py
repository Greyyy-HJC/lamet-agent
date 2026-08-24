"""Plain JSON manifest loading, contract evaluation, and ordered job views."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Mapping, Sequence

from .contract import (
    CheckContext,
    Depends,
    Issue,
    Value,
    _apply_recommended_defaults,
    _unresolved_null_hooks,
    evaluate_checks,
    evaluate_rules,
)

_SAFE_STAGE = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_JOB = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_KERNEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _strip_json_comments(text: str) -> str:
    """Remove JSONC comments while preserving strings and source positions."""
    output = list(text)
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            index += 1
            continue
        if character != "/" or index + 1 >= len(text):
            index += 1
            continue
        marker = text[index + 1]
        if marker == "/":
            output[index] = output[index + 1] = " "
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                output[index] = " "
                index += 1
            continue
        if marker == "*":
            start = index
            output[index] = output[index + 1] = " "
            index += 2
            while index + 1 < len(text) and text[index : index + 2] != "*/":
                if text[index] not in "\r\n":
                    output[index] = " "
                index += 1
            if index + 1 >= len(text):
                raise json.JSONDecodeError("Unterminated block comment", text, start)
            output[index] = output[index + 1] = " "
            index += 2
            continue
        index += 1
    return "".join(output)


def _nonnegative(value: int) -> bool:
    return value >= 0


def _positive(value: int) -> bool:
    return value > 0


@dataclass(frozen=True)
class _ResolvedJob:
    """One authored job with effective parameters and its artifact cell."""

    stage_id: str
    stage_index: int
    job_index: int
    job_id: str
    params: Mapping[str, Any]
    inputs: Mapping[str, Any]
    artifact_directory: Path


@dataclass
class Manifest:
    """One parsed manifest document coupled to its source path."""

    path: Path
    document: dict[str, Any]

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()
        if not isinstance(self.document, dict):
            raise TypeError("Manifest document must be a JSON object")

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return validated-style metadata without copying the document."""
        metadata = self.document.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Manifest metadata must be an object")
        return metadata

    @property
    def root_directory(self) -> Path:
        """Resolve the authored root directory from the manifest location."""
        root = self.metadata.get("root_directory")
        if not isinstance(root, str):
            raise ValueError("Manifest root_directory must be a path string")
        return _resolve_root(self.path, root)

    def validate(self, *, stage_root: str | Path | None = None) -> list[Issue]:
        """Return all structural, stage-contract, and reference issues."""
        return _validate_document(self.document, manifest_path=self.path, stage_root=stage_root)

    def _resolved_jobs(self) -> list[_ResolvedJob]:
        """Return jobs in authored order with exact artifact paths."""
        return _build_resolved_jobs(self)

    def _resolve_source(
        self,
        source: Any,
        *,
        outputs: Mapping[str, Any],
        summaries: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        """Resolve one source relative to this manifest's root directory."""
        return _resolve_source(source, root=self.root_directory, outputs=outputs, summaries=summaries)


_BASE_RULES: tuple[Depends | Value, ...] = (
    Depends("", "metadata", physics="Run metadata names the physical analysis and its execution root."),
    Depends("", "stages", physics="The authored stage mapping is the sole execution order."),
    Depends("metadata", "run_id", physics="A run needs a stable human-readable identifier."),
    Depends("metadata", "root_directory", physics="Relative input paths are resolved from this directory."),
    Depends("metadata", "artifacts_directory", physics="Every job owns one artifact cell below this directory."),
    Depends("metadata", "random_seed", physics="Stochastic numerical work needs a reproducible root seed."),
    Depends("metadata", "workers", physics="All sample-wise fits share one run-level parallelism limit."),
    Depends("metadata", "target_observable", physics="The run has one final physical target observable."),
    Depends("metadata", "resample_mode", physics="All correlator jobs use one run-level resampling convention."),
    Depends("metadata", "sample_error_mode", physics="All stages use one authored sample center and error convention."),
    Depends("metadata", "bootstrap_samples", physics="Bootstrap sample count is explicit when bootstrap is selected.", required=False),
    Depends("metadata", "bin_size", physics="All raw correlators use one positive configuration bin size."),
    Value("metadata.run_id", str, physics="The run id is a human-readable string."),
    Value("metadata.root_directory", str, physics="The root directory is a path string."),
    Value("metadata.artifacts_directory", str, physics="The artifact base is a path string."),
    Value("metadata.random_seed", int, physics="The random seed is a nonnegative integer.", validator=_nonnegative),
    Value("metadata.workers", int, physics="The run-level worker count is a positive integer.", validator=_positive),
    Value("metadata.target_observable", Literal["pdf", "da"], physics="The migrated workflow targets a PDF or DA."),
    Value("metadata.resample_mode", Literal["jackknife", "bootstrap"], physics="The run-level sample plan is jackknife or bootstrap."),
    Value("metadata.sample_error_mode", Literal["covariance", "mean", "median"], physics="Sample statistics use covariance, mean-diagonal, or median-percentile errors."),
    Value("metadata.bootstrap_samples", int, physics="Bootstrap sample count is positive.", validator=_positive),
    Value("metadata.bin_size", int, physics="Configuration bin size is positive.", validator=_positive),
)


def _metadata_relationship_issues(metadata: Mapping[str, Any]) -> list[Issue]:
    """Validate relationships within the global sampling plan."""
    issues: list[Issue] = []
    mode = metadata.get("resample_mode")
    count = metadata.get("bootstrap_samples")
    if mode == "bootstrap" and count is None:
        issues.append(_issue("metadata.bootstrap_samples", "is required when resample_mode='bootstrap'", "Bootstrap requires an authored sample count."))
    elif mode == "jackknife" and count is not None:
        issues.append(_issue("metadata.bootstrap_samples", "must be omitted for jackknife", "Jackknife sample count is fixed by the binned configurations."))
    if metadata.get("sample_error_mode") == "median" and mode != "bootstrap":
        issues.append(_issue("metadata.sample_error_mode", "median errors require resample_mode='bootstrap'", "Median-percentile errors require bootstrap samples."))
    return issues


def load_manifest(path: str | Path) -> Manifest:
    """Parse one JSON or JSONC manifest and retain its resolved source path."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Manifest does not exist: {path}")
    text = manifest_path.read_text(encoding="utf-8")
    document = json.loads(_strip_json_comments(text))
    if not isinstance(document, dict):
        raise ValueError("Manifest root must be a JSON object")
    return Manifest(manifest_path, document)


def _merge_defaults(defaults: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalar values are replaced."""
    merged = copy.deepcopy(dict(defaults))
    for key, value in params.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _default_stage_root(stage_root: str | Path | None) -> Path:
    return Path(stage_root).expanduser().resolve() if stage_root is not None else (Path(__file__).parent / "stages").resolve()


def _kernel_file(stage_root: str | Path | None, kernel_id: str) -> Path:
    """Resolve a kernel stem lexically without importing its module."""
    return _default_stage_root(stage_root).parent / "kernels" / f"{kernel_id}.py"


def _kernel_document_file(stage_root: str | Path | None, kernel_id: str) -> Path:
    return _default_stage_root(stage_root).parent / "kernels" / f"{kernel_id}.md"


def _load_stage_contract(stage_id: str, stage_root: str | Path | None = None) -> ModuleType:
    """Load one selected stage contract directly from its directory."""
    if not _SAFE_STAGE.fullmatch(stage_id):
        raise ValueError(f"Invalid stage id '{stage_id}'")
    contract_path = _default_stage_root(stage_root) / stage_id / "contract.py"
    if not contract_path.is_file():
        raise ValueError(f"Stage '{stage_id}' has no contract.py")
    digest = __import__("hashlib").sha256(str(contract_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_lamet_agent_neo_contract_{stage_id}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load contract for stage '{stage_id}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("PARAM_RULES", "INPUT_RULES", "CHECKS"):
        if not hasattr(module, name):
            raise ValueError(f"Stage '{stage_id}' contract does not export {name}")
    return module


def _issue(path: str, message: str, physics: str = "Manifest structure is explicit.") -> Issue:
    return Issue(path, message, physics, None)


def _source_issues(value: Any, path: str, seen_jobs: set[str], root: Path, *, allow_constant: bool = False) -> list[Issue]:
    issues: list[Issue] = []
    if isinstance(value, list):
        if not value:
            issues.append(_issue(path, "must not be an empty source list"))
        for index, item in enumerate(value):
            issues.extend(_source_issues(item, f"{path}[{index}]", seen_jobs, root, allow_constant=allow_constant))
        return issues
    if allow_constant and isinstance(value, (int, float)) and not isinstance(value, bool):
        return []
    if not isinstance(value, Mapping):
        return [_issue(path, "must be a source object or a nonempty list of source objects")]
    if set(value) not in ({"job"}, {"file"}):
        issues.append(_issue(path, "must contain exactly one of 'job' or 'file'"))
        return issues
    if "job" in value:
        reference = value["job"]
        if not isinstance(reference, str):
            issues.append(_issue(f"{path}.job", "must be a string"))
        elif reference not in seen_jobs:
            issues.append(_issue(f"{path}.job", f"must reference an earlier job, not '{reference}'"))
    else:
        file_value = value["file"]
        if not isinstance(file_value, str):
            issues.append(_issue(f"{path}.file", "must be a string"))
        else:
            resolved = Path(file_value).expanduser()
            if not resolved.is_absolute():
                resolved = root / resolved
            resolved = resolved.resolve()
            if not resolved.is_file():
                issues.append(_issue(f"{path}.file", f"file does not exist: {resolved}"))
    return issues


def _resolve_root(manifest_path: Path, value: Any) -> Path:
    root = Path(value).expanduser()
    return root.resolve() if root.is_absolute() else (manifest_path.parent / root).resolve()


def _validate_document(
    document: Mapping[str, Any],
    *,
    manifest_path: Path,
    stage_root: str | Path | None = None,
) -> list[Issue]:
    """Return all structural, stage-contract, and ordered-reference issues."""
    issues: list[Issue] = []
    if not isinstance(document, Mapping):
        return [_issue("", "must be a JSON object")]
    # The common envelope is never a partial authored view.  Planning may leave
    # stage defaults and job parameters incomplete, but metadata and the
    # authored stage mapping are required before stage contracts can be loaded.
    issues.extend(evaluate_rules(document, _BASE_RULES, complete=True))
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        return issues
    issues.extend(_metadata_relationship_issues(metadata))
    root_value = metadata.get("root_directory")
    manifest_file = manifest_path
    root = _resolve_root(manifest_file, root_value) if isinstance(root_value, str) else manifest_file.parent
    if isinstance(root_value, str) and not root.is_dir():
        issues.append(_issue("metadata.root_directory", f"directory does not exist: {root}"))
    stage_blocks = document.get("stages")
    if not isinstance(stage_blocks, Mapping):
        issues.append(_issue("stages", "must be an object"))
        return issues
    if not stage_blocks:
        issues.append(_issue("stages", "must be a nonempty object"))
        return issues
    stages = list(stage_blocks)
    for stage_id in stages:
        if not isinstance(stage_id, str) or not _SAFE_STAGE.fullmatch(stage_id):
            issues.append(
                _issue(
                    f"stages.{stage_id}",
                    "stage id must match [a-z][a-z0-9_]*",
                )
            )

    seen_jobs: set[str] = set()
    root_for_inputs = root
    for stage_id in stages:
        if not isinstance(stage_id, str) or not _SAFE_STAGE.fullmatch(stage_id):
            continue
        block = stage_blocks.get(stage_id)
        if not isinstance(block, Mapping):
            issues.append(_issue(f"stages.{stage_id}", "must be an object"))
            continue
        allowed_block_keys = {"defaults", "jobs"}
        for key in block:
            if key not in allowed_block_keys:
                issues.append(_issue(f"stages.{stage_id}.{key}", "unknown stage key"))
        defaults = block.get("defaults", {})
        jobs = block.get("jobs")
        if not isinstance(defaults, Mapping):
            issues.append(_issue(f"stages.{stage_id}.defaults", "must be an object"))
            defaults = {}
        if not isinstance(jobs, list) or not jobs:
            issues.append(_issue(f"stages.{stage_id}.jobs", "must be a nonempty list"))
            continue
        try:
            contract = _load_stage_contract(stage_id, stage_root)
        except ValueError as exc:
            issues.append(_issue(f"stages.{stage_id}", str(exc)))
            contract = None
        if contract is not None:
            issues.extend(_prefix_issues(evaluate_rules(defaults, contract.PARAM_RULES, complete=False), f"stages.{stage_id}.defaults"))
        for job_index, job in enumerate(jobs):
            job_path = f"stages.{stage_id}.jobs[{job_index}]"
            if not isinstance(job, Mapping):
                issues.append(_issue(job_path, "must be an object"))
                continue
            for key in job:
                if key not in {"id", "inputs", "params"}:
                    issues.append(_issue(f"{job_path}.{key}", "unknown job key"))
            job_id = job.get("id")
            if not isinstance(job_id, str) or not _SAFE_JOB.fullmatch(job_id):
                issues.append(_issue(f"{job_path}.id", "must match [A-Za-z0-9][A-Za-z0-9_.-]*"))
                job_id = None
            elif job_id in seen_jobs:
                issues.append(_issue(f"{job_path}.id", f"job id '{job_id}' is not globally unique"))
            inputs = job.get("inputs", {})
            params = job.get("params", {})
            if not isinstance(inputs, Mapping):
                issues.append(_issue(f"{job_path}.inputs", "must be an object"))
                inputs = {}
            if not isinstance(params, Mapping):
                issues.append(_issue(f"{job_path}.params", "must be an object"))
                params = {}
            for role, source in inputs.items():
                issues.extend(_source_issues(source, f"{job_path}.inputs.{role}", seen_jobs, root_for_inputs, allow_constant=stage_id == "renormalization" and role == "denominator"))
            if contract is not None:
                input_rule_issues = evaluate_rules(inputs, contract.INPUT_RULES, complete=True)
                issues.extend(_prefix_issues(input_rule_issues, f"{job_path}.inputs"))
                effective = _merge_defaults(defaults, params)
                _apply_recommended_defaults(effective, contract.PARAM_RULES)
                param_rule_issues = evaluate_rules(effective, contract.PARAM_RULES, complete=True)
                issues.extend(_prefix_issues(param_rule_issues, f"{job_path}.params"))
                if stage_id == "perturbative_matching" and isinstance(effective.get("kernel_id"), str) and _SAFE_KERNEL.fullmatch(effective["kernel_id"]):
                    if not _kernel_file(stage_root, effective["kernel_id"]).is_file():
                        issues.append(_issue(f"{job_path}.params.kernel_id", f"kernel file does not exist: {_kernel_file(stage_root, effective['kernel_id'])}"))
                    elif not _kernel_document_file(stage_root, effective["kernel_id"]).is_file():
                        issues.append(_issue(f"{job_path}.params.kernel_id", f"kernel formula document does not exist: {_kernel_document_file(stage_root, effective['kernel_id'])}"))
                if job_id is not None and not input_rule_issues and not param_rule_issues:
                    unresolved = frozenset(
                        rule.path
                        for rule in _unresolved_null_hooks(
                            effective, contract.PARAM_RULES
                        )
                    )
                    context = CheckContext(
                        document,
                        stage_id,
                        job_id,
                        effective,
                        inputs,
                        unresolved,
                    )
                    issues.extend(_prefix_issues(evaluate_checks(contract.CHECKS, context), job_path))
            if job_id is not None:
                seen_jobs.add(job_id)
    return issues


def _prefix_issues(issues: Sequence[Issue], prefix: str) -> list[Issue]:
    output: list[Issue] = []
    for issue in issues:
        path = prefix if not issue.path else f"{prefix}.{issue.path}"
        output.append(Issue(path, issue.message, issue.physics, issue.question))
    return output


def _build_resolved_jobs(manifest: Manifest) -> list[_ResolvedJob]:
    """Return jobs in authored stage/job order with exact artifact paths."""
    document = manifest.document
    metadata = document["metadata"]
    root = manifest.root_directory
    artifact_base = Path(metadata["artifacts_directory"]).expanduser()
    artifact_base = artifact_base.resolve() if artifact_base.is_absolute() else (root / artifact_base).resolve()
    resolved: list[_ResolvedJob] = []
    for stage_index, (stage_id, block) in enumerate(document["stages"].items(), start=1):
        defaults = block.get("defaults", {})
        for job_index, job in enumerate(block["jobs"]):
            resolved.append(
                _ResolvedJob(
                    stage_id=stage_id,
                    stage_index=stage_index,
                    job_index=job_index,
                    job_id=job["id"],
                    params=_merge_defaults(defaults, job.get("params", {})),
                    inputs=copy.deepcopy(job.get("inputs", {})),
                    artifact_directory=artifact_base / f"{stage_index:02d}_{stage_id}" / job["id"],
                )
            )
    return resolved


def _resolve_source(source: Any, *, root: Path, outputs: Mapping[str, Any], summaries: Mapping[str, Any]) -> tuple[Any, Any]:
    """Resolve one authored source recursively for the ordered agent loop."""
    if isinstance(source, (int, float)) and not isinstance(source, bool):
        return source, None
    if isinstance(source, list):
        values: list[Any] = []
        source_summaries: list[Any] = []
        for item in source:
            value, summary = _resolve_source(item, root=root, outputs=outputs, summaries=summaries)
            values.append(value)
            source_summaries.append(summary)
        return values, source_summaries
    if not isinstance(source, Mapping) or set(source) not in ({"job"}, {"file"}):
        raise ValueError("Invalid input source")
    if "job" in source:
        job_id = source["job"]
        if job_id not in outputs:
            raise RuntimeError(f"Job output '{job_id}' is not available")
        return outputs[job_id], summaries.get(job_id)
    path = Path(source["file"]).expanduser()
    return (path if path.is_absolute() else (root / path).resolve()), None


__all__ = ["Manifest", "load_manifest"]
