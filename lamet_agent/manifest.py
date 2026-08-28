"""Plain JSON manifest loading, contract evaluation, and ordered job views."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Mapping, Sequence

from .contract import (
    CheckContext,
    Depends,
    Issue,
    Provides,
    Recommends,
    Value,
    _unresolved_null_hooks,
    evaluate_checks,
    evaluate_rules,
)

_SAFE_STAGE = re.compile(r"^[a-z][a-z0-9_]*$")


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
class Job:
    """One concrete globally ordered job owned by a parsed Manifest."""

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
    jobs: tuple[Job, ...] = field(default=(), repr=False)
    _systematics_expanded: bool = field(default=False, repr=False)
    _has_systematics: bool | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()
        if not isinstance(self.document, dict):
            raise TypeError("Manifest document must be a JSON object")
        if self._has_systematics is None:
            declarations = self.document.get("systematics")
            self._has_systematics = isinstance(declarations, Mapping) and any(
                isinstance(config, Mapping) and bool(config.get("variants")) for config in declarations.values()
            )

    @property
    def has_systematics(self) -> bool:
        """Return whether the authored manifest declared systematics."""
        return bool(self._has_systematics)

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return validated-style metadata without copying the document."""
        metadata = self.document.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Manifest metadata must be an object")
        return metadata

    @property
    def jobs_by_stage(self) -> Mapping[str, tuple[Job, ...]]:
        """Return a derived stage index over the canonical flat job graph."""
        return _group_jobs_by_stage(self.document, self.jobs)

    @property
    def root_directory(self) -> Path:
        """Resolve the authored root directory from the manifest location."""
        root = self.metadata.get("root_directory")
        if not isinstance(root, str):
            raise ValueError("Manifest root_directory must be a path string")
        return _resolve_root(self.path, root)

    def validate(self, *, stage_root: str | Path | None = None) -> list[Issue]:
        """Validate authored contracts, expand systematics, then validate generated jobs."""
        authored = Manifest(
            self.path,
            copy.deepcopy(self.document),
            _systematics_expanded=self._systematics_expanded,
            _has_systematics=self.has_systematics,
        )
        issues = _validate_document(authored.document, manifest_path=self.path, stage_root=stage_root)
        if not authored._systematics_expanded:
            issues.extend(_validate_systematics_contracts(authored.document, stage_root=stage_root))
        if issues:
            return issues
        try:
            expanded = authored.expand_systematics(stage_root=stage_root)
        except (TypeError, ValueError) as exc:
            return [
                _issue(
                    "systematics",
                    str(exc),
                    "Systematics declarations must compile to one deterministic concrete job graph.",
                )
            ]
        issues = _validate_document(expanded.document, manifest_path=self.path, stage_root=stage_root)
        if not issues:
            expanded.document.pop("systematics", None)
            self.document = expanded.document
            self._systematics_expanded = True
            self.jobs = tuple(_build_jobs(self))
        return issues

    def expand_systematics(self, *, stage_root: str | Path | None = None) -> "Manifest":
        """Return a copy whose stage-local systematics declarations are concrete jobs."""
        if self._systematics_expanded:
            return self
        document = copy.deepcopy(self.document)
        contract_issues = evaluate_rules(document, _BASE_RULES, complete=True)
        if isinstance(document.get("metadata"), Mapping):
            contract_issues.extend(evaluate_checks(_BASE_CHECKS, CheckContext(document, "", None, {}, {})))
        contract_issues.extend(_validate_systematics_contracts(document, stage_root=stage_root))
        if contract_issues:
            raise ValueError("; ".join(f"{issue.path}: {issue.message}" for issue in contract_issues))
        declarations = document["systematics"]
        stages = document["stages"]
        state: dict[str, Any] = {"root_directory": self.root_directory}
        for stage_id in stages:
            systematics_path = _default_stage_root(stage_root) / stage_id / "systematics.py"
            if not systematics_path.is_file():
                continue
            config = copy.deepcopy(dict(declarations.get(stage_id, {})))
            contract = _load_stage_contract(stage_id, stage_root)
            local = evaluate_rules(
                config,
                contract.SYSTEMATICS_RULES,
                complete=True,
                root_document=document,
            )
            if not local:
                local.extend(
                    evaluate_checks(
                        contract.SYSTEMATICS_CHECKS,
                        CheckContext(document, stage_id, None, config, {}),
                    )
                )
            if local:
                raise ValueError("; ".join(f"{stage_id}.{issue.path}: {issue.message}" for issue in local))
            module = _load_stage_systematics(stage_id, stage_root)
            try:
                module.expand(document, config, state)
            except KeyError as exc:
                raise ValueError(f"{stage_id} systematics expansion requires missing field {exc}") from exc
        document.pop("systematics")
        return Manifest(
            self.path,
            document,
            _systematics_expanded=True,
            _has_systematics=self.has_systematics,
        )

    def _resolved_jobs(self, *, stage_root: str | Path | None = None) -> list[Job]:
        """Return jobs in authored order with exact artifact paths."""
        if self.jobs:
            return list(self.jobs)
        resolved = self.expand_systematics(stage_root=stage_root)
        return _build_jobs(resolved)

    def _resolve_source(
        self,
        source: Any,
        *,
        outputs: Mapping[str, Any],
        summaries: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        """Resolve one source relative to this manifest's root directory."""
        return _resolve_source(source, root=self.root_directory, outputs=outputs, summaries=summaries)


# ruff: disable[E501]
# fmt: off
_BASE_RULES: tuple[Depends | Provides | Recommends | Value, ...] = (
    Value("", dict, physics="The manifest root is one JSON object."),
    Depends("", "metadata", physics="Run metadata names the physical analysis and its execution root."),
    Depends("", "stages", physics="The authored stage mapping is the sole execution order."),
    Recommends("", "systematics", physics="Runs without systematic variations use an empty declaration mapping.", default={}),
    Value("metadata", dict, physics="Run metadata is an object."),
    Value("stages", dict, physics="Stages form a nonempty ordered object.", validator=bool),
    Value("systematics", dict, physics="Systematics declarations form a stage-keyed object."),
    Depends("metadata", "run_id", physics="A run needs a stable human-readable identifier."),
    Depends("metadata", "root_directory", physics="Relative input paths are resolved from this directory."),
    Depends("metadata", "artifacts_directory", physics="Every job owns one artifact cell below this directory."),
    Depends("metadata", "random_seed", physics="Stochastic numerical work needs a reproducible root seed."),
    Depends("metadata", "workers", physics="All sample-wise fits share one run-level parallelism limit."),
    Depends("metadata", "target_observable", physics="The run has one final physical target observable."),
    Recommends("metadata", "parton", physics="The migrated workflows default to the reference quark species.", default="quark"),
    Depends("metadata", "resample_mode", physics="All correlator jobs use one run-level resampling convention."),
    Depends("metadata", "sample_error_mode", physics="All stages use one authored sample center and error convention."),
    Recommends("metadata", "parameter_recommendation_retries", physics="Each job permits one parameter recommendation plus this many bounded retries.", default=1),
    Provides("metadata", "bootstrap", "metadata.resample_mode", physics="Bootstrap sampling owns one explicit sample count."),
    Depends("metadata.bootstrap", "samples", physics="Bootstrap sample count is explicit when bootstrap is selected."),
    Depends("metadata", "bin_size", physics="All raw correlators use one positive configuration bin size."),
    Value("metadata.run_id", str, physics="The run id is a human-readable string."),
    Value("metadata.root_directory", str, physics="The root directory is a path string."),
    Value("metadata.artifacts_directory", str, physics="The artifact base is a path string."),
    Value("metadata.random_seed", int, physics="The random seed is a nonnegative integer.", validator=_nonnegative),
    Value("metadata.workers", int, physics="The run-level worker count is a positive integer.", validator=_positive),
    Value("metadata.target_observable", Literal["pdf", "da"], physics="The migrated workflow targets a PDF or DA."),
    Value("metadata.parton", Literal["quark"], physics="The migrated examples use quark distributions."),
    Value("metadata.resample_mode", Literal["jackknife", "bootstrap"], physics="The run-level sample plan is jackknife or bootstrap."),
    Value("metadata.sample_error_mode", Literal["covariance", "variance", "one_sigma"], physics="Sample statistics use covariance, variance-only mean, or median one-sigma errors."),
    Value("metadata.parameter_recommendation_retries", int, physics="The per-job parameter recommendation retry limit is nonnegative.", validator=_nonnegative),
    Value("metadata.bootstrap.samples", int, physics="Bootstrap sample count is positive.", validator=_positive),
    Value("metadata.bin_size", int, physics="Configuration bin size is positive.", validator=_positive),
)
# fmt: on
# ruff: enable[E501]


def _check_manifest_relations(context: CheckContext) -> list[Issue]:
    """Validate top-level sampling and stage-key relationships."""
    issues: list[Issue] = []
    metadata = context.manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        return issues
    mode = metadata.get("resample_mode")
    if metadata.get("sample_error_mode") == "one_sigma" and mode != "bootstrap":
        issues.append(
            _issue(
                "metadata.sample_error_mode",
                "one_sigma errors require resample_mode='bootstrap'",
                "Median one-sigma errors require bootstrap samples.",
            )
        )
    stages = context.manifest.get("stages")
    if isinstance(stages, Mapping):
        for stage_name in stages:
            if not isinstance(stage_name, str) or not _SAFE_STAGE.fullmatch(stage_name):
                issues.append(
                    _issue(
                        f"stages.{stage_name}",
                        "stage id must match [a-z][a-z0-9_]*",
                    )
                )
            elif not isinstance(stages[stage_name], Mapping):
                issues.append(
                    _issue(
                        f"stages.{stage_name}",
                        "must be an object",
                        "Every stage owns one defaults/jobs configuration object.",
                    )
                )
    declarations = context.manifest.get("systematics")
    if isinstance(declarations, Mapping) and isinstance(stages, Mapping):
        for stage_id, config in declarations.items():
            if stage_id not in stages:
                issues.append(
                    _issue(
                        f"systematics.{stage_id}",
                        "declares a stage absent from stages",
                        "Systematic variations can only extend an authored stage.",
                    )
                )
            if not isinstance(config, Mapping):
                issues.append(
                    _issue(
                        f"systematics.{stage_id}",
                        "must be an object",
                        "Each stage owns one systematics configuration object.",
                    )
                )
    return issues


_BASE_CHECKS = (_check_manifest_relations,)


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


def _default_stage_root(stage_root: str | Path | None) -> Path:
    return (
        Path(stage_root).expanduser().resolve()
        if stage_root is not None
        else (Path(__file__).parent / "stages").resolve()
    )


def _load_stage_contract(stage_id: str, stage_root: str | Path | None = None) -> ModuleType:
    """Load one selected stage contract directly from its directory."""
    if not _SAFE_STAGE.fullmatch(stage_id):
        raise ValueError(f"Invalid stage id '{stage_id}'")
    contract_path = _default_stage_root(stage_root) / stage_id / "contract.py"
    if not contract_path.is_file():
        raise ValueError(f"Stage '{stage_id}' has no contract.py")
    digest = __import__("hashlib").sha256(str(contract_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_lamet_agent_contract_{stage_id}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load contract for stage '{stage_id}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("JOB_RULES", "PARAM_RULES", "CHECKS"):
        if not hasattr(module, name):
            raise ValueError(f"Stage '{stage_id}' contract does not export {name}")
    systematics_path = contract_path.with_name("systematics.py")
    if systematics_path.is_file():
        for name in ("SYSTEMATICS_RULES", "SYSTEMATICS_CHECKS"):
            if not hasattr(module, name):
                raise ValueError(f"Stage '{stage_id}' contract does not export {name}")
    return module


def _load_stage_systematics(stage_id: str, stage_root: str | Path | None = None) -> ModuleType:
    """Load one optional stage-owned systematics compiler."""
    if not _SAFE_STAGE.fullmatch(stage_id):
        raise ValueError(f"invalid systematics stage id '{stage_id}'")
    path = _default_stage_root(stage_root) / stage_id / "systematics.py"
    if not path.is_file():
        raise ValueError(f"stage '{stage_id}' does not support systematics declarations")
    digest = __import__("hashlib").sha256(str(path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_lamet_agent_systematics_{stage_id}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load systematics compiler for stage '{stage_id}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(module.expand):
        raise ValueError(f"stage '{stage_id}' systematics expand is not callable")
    return module


def _validate_systematics_contracts(
    document: Mapping[str, Any],
    *,
    stage_root: str | Path | None = None,
) -> list[Issue]:
    """Validate each authored stage-systematics object with its owned contract."""
    declarations = document.get("systematics")
    stages = document.get("stages")
    if not isinstance(declarations, Mapping) or not isinstance(stages, Mapping):
        return []
    issues: list[Issue] = []
    for stage_id in stages:
        config = declarations.get(stage_id)
        if config is None or not isinstance(config, Mapping):
            continue
        try:
            contract = _load_stage_contract(str(stage_id), stage_root)
            _load_stage_systematics(str(stage_id), stage_root)
        except ValueError as exc:
            issues.append(_issue(f"systematics.{stage_id}", str(exc)))
            continue
        local = evaluate_rules(
            config,
            contract.SYSTEMATICS_RULES,
            complete=True,
            root_document=document,
        )
        issues.extend(_prefix_issues(local, f"systematics.{stage_id}"))
        if not local:
            context = CheckContext(document, str(stage_id), None, config, {})
            issues.extend(
                _prefix_issues(
                    evaluate_checks(contract.SYSTEMATICS_CHECKS, context),
                    f"systematics.{stage_id}",
                )
            )
    return issues


def _issue(path: str, message: str, physics: str = "Manifest structure is explicit.") -> Issue:
    return Issue(path, message, physics, None)


def _source_issues(value: Any, path: str, seen_jobs: set[str], root: Path) -> list[Issue]:
    issues: list[Issue] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_source_issues(item, f"{path}[{index}]", seen_jobs, root))
        return issues
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return []
    if isinstance(value, str):
        if value not in seen_jobs:
            return [_issue(path, f"must reference an earlier job, not '{value}'")]
        return []
    if not isinstance(value, Mapping) or set(value) != {"file"} or not isinstance(value.get("file"), str):
        return issues
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
    issues.extend(
        evaluate_checks(
            _BASE_CHECKS,
            CheckContext(document, "", None, {}, {}),
        )
    )
    root_value = metadata.get("root_directory")
    manifest_file = manifest_path
    root = _resolve_root(manifest_file, root_value) if isinstance(root_value, str) else manifest_file.parent
    if isinstance(root_value, str) and not root.is_dir():
        issues.append(_issue("metadata.root_directory", f"directory does not exist: {root}"))
    stage_blocks = document.get("stages")
    if not isinstance(stage_blocks, Mapping):
        issues.append(_issue("stages", "must be an object"))
        return issues
    stages = list(stage_blocks)

    for stage_id in stages:
        if not isinstance(stage_id, str) or not _SAFE_STAGE.fullmatch(stage_id):
            continue
        block = stage_blocks.get(stage_id)
        if not isinstance(block, Mapping):
            continue
        jobs = block.get("jobs")
        try:
            contract = _load_stage_contract(stage_id, stage_root)
        except ValueError as exc:
            issues.append(_issue(f"stages.{stage_id}", str(exc)))
            contract = None
        stage_rule_issues: list[Issue] = []
        if contract is not None:
            if isinstance(jobs, list) and jobs:
                parsed_jobs: list[Any] = []
                for job_index, job in enumerate(jobs):
                    single = {key: copy.deepcopy(value) for key, value in block.items() if key != "jobs"}
                    single["jobs"] = [copy.deepcopy(job)]
                    local_issues = evaluate_rules(
                        single,
                        contract.JOB_RULES,
                        complete=True,
                        root_document=document,
                    )
                    for issue in local_issues:
                        path = issue.path
                        if path == "jobs[0]":
                            path = f"jobs[{job_index}]"
                        elif path.startswith("jobs[0]."):
                            path = f"jobs[{job_index}].{path[len('jobs[0].') :]}"
                        stage_rule_issues.append(Issue(path, issue.message, issue.physics, issue.question))
                    parsed_jobs.append(single["jobs"][0])
                if isinstance(block, dict):
                    block["jobs"] = parsed_jobs
            else:
                stage_rule_issues = evaluate_rules(
                    block,
                    contract.JOB_RULES,
                    complete=True,
                    root_document=document,
                )
            issues.extend(_prefix_issues(stage_rule_issues, f"stages.{stage_id}"))
        jobs = block.get("jobs")
        if not isinstance(jobs, list):
            continue
        for job_index, job in enumerate(jobs):
            job_path = f"stages.{stage_id}.jobs[{job_index}]"
            if not isinstance(job, Mapping):
                continue
            job_id = job.get("id")
            if not isinstance(job_id, str):
                job_id = None
            inputs = job.get("inputs", {})
            params = {key: value for key, value in job.items() if key not in {"id", "inputs"}}
            if contract is not None:
                job_rule_issues = [
                    issue
                    for issue in stage_rule_issues
                    if issue.path == f"jobs[{job_index}]" or issue.path.startswith(f"jobs[{job_index}].")
                ]
                if job_id is not None and not job_rule_issues and isinstance(inputs, Mapping):
                    unresolved = frozenset(rule.path for rule in _unresolved_null_hooks(params, contract.PARAM_RULES))
                    context = CheckContext(
                        document,
                        stage_id,
                        job_id,
                        params,
                        inputs,
                        unresolved,
                    )
                    issues.extend(_prefix_issues(evaluate_checks(contract.CHECKS, context), job_path))
    jobs_for_graph = _build_jobs_from_document(document, manifest_path=manifest_path)
    issues.extend(_job_graph_issues(jobs_for_graph, root=root))
    return issues


def _prefix_issues(issues: Sequence[Issue], prefix: str) -> list[Issue]:
    output: list[Issue] = []
    for issue in issues:
        path = prefix if not issue.path else f"{prefix}.{issue.path}"
        output.append(Issue(path, issue.message, issue.physics, issue.question))
    return output


def _build_jobs(manifest: Manifest) -> list[Job]:
    """Return jobs in authored stage/job order with exact artifact paths."""
    return _build_jobs_from_document(manifest.document, manifest_path=manifest.path)


def _build_jobs_from_document(document: Mapping[str, Any], *, manifest_path: Path) -> list[Job]:
    metadata = document.get("metadata")
    stages = document.get("stages")
    if not isinstance(metadata, Mapping) or not isinstance(stages, Mapping):
        return []
    root_value = metadata.get("root_directory")
    artifacts_value = metadata.get("artifacts_directory")
    if not isinstance(root_value, str) or not isinstance(artifacts_value, str):
        return []
    root = _resolve_root(manifest_path, root_value)
    artifact_base = Path(artifacts_value).expanduser()
    artifact_base = artifact_base.resolve() if artifact_base.is_absolute() else (root / artifact_base).resolve()
    resolved: list[Job] = []
    for stage_index, (stage_id, block) in enumerate(stages.items(), start=1):
        if not isinstance(block, Mapping) or not isinstance(block.get("jobs"), list):
            continue
        for job_index, job in enumerate(block["jobs"]):
            if not isinstance(job, Mapping):
                continue
            job_id = job.get("id")
            inputs = job.get("inputs", {})
            if not isinstance(job_id, str) or not isinstance(inputs, Mapping):
                continue
            resolved.append(
                Job(
                    stage_id=stage_id,
                    stage_index=stage_index,
                    job_index=job_index,
                    job_id=job["id"],
                    params={key: copy.deepcopy(value) for key, value in job.items() if key not in {"id", "inputs"}},
                    inputs=copy.deepcopy(dict(inputs)),
                    artifact_directory=artifact_base / f"{stage_index:02d}_{stage_id}" / job["id"],
                )
            )
    return resolved


def _group_jobs_by_stage(document: Mapping[str, Any], jobs: Sequence[Job]) -> dict[str, tuple[Job, ...]]:
    """Group the concrete flat job graph by authored stage order."""
    blocks = document.get("stages")
    if not isinstance(blocks, Mapping):
        return {}
    grouped: dict[str, list[Job]] = {str(stage_id): [] for stage_id in blocks}
    for job in jobs:
        if job.stage_id not in grouped:
            raise ValueError(f"job '{job.job_id}' belongs to absent stage '{job.stage_id}'")
        grouped[job.stage_id].append(job)
    return {stage_id: tuple(stage_jobs) for stage_id, stage_jobs in grouped.items()}


def _job_graph_issues(jobs: Sequence[Job], *, root: Path) -> list[Issue]:
    """Validate global uniqueness and prior-job/file references in one ordered pass."""
    issues: list[Issue] = []
    seen: set[str] = set()
    for job in jobs:
        job_path = f"stages.{job.stage_id}.jobs[{job.job_index}]"
        if job.job_id in seen:
            issues.append(
                _issue(
                    f"{job_path}.id",
                    f"job id '{job.job_id}' is not globally unique",
                )
            )
        for role, source in job.inputs.items():
            issues.extend(
                _source_issues(
                    source,
                    f"{job_path}.inputs.{role}",
                    seen,
                    root,
                )
            )
        seen.add(job.job_id)
    return issues


def _resolve_source(
    source: Any, *, root: Path, outputs: Mapping[str, Any], summaries: Mapping[str, Any]
) -> tuple[Any, Any]:
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
    if isinstance(source, str):
        if source not in outputs:
            raise RuntimeError(f"Job output '{source}' is not available")
        return outputs[source], summaries.get(source)
    if not isinstance(source, Mapping) or set(source) != {"file"}:
        raise ValueError("Invalid input source")
    path = Path(source["file"]).expanduser()
    return (path if path.is_absolute() else (root / path).resolve()), None


__all__ = ["Job", "Manifest", "load_manifest"]
