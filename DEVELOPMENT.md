# Development

This is the development reference for both human contributors and coding
agents. It starts with the system design, then defines module boundaries and
extension protocols, and ends with the expected change and verification
workflow. Installation, command-line use, manifests, and data formats are
documented in [`README.md`](README.md).

## Contents

- Understand the system: [Architecture](#architecture),
  [Manifest and contract lifecycle](#manifest-and-contract-lifecycle),
  [Component ownership](#component-ownership), and
  [Job execution lifecycle](#job-execution-lifecycle).
- Extend the system: [Extending a stage](#extending-a-stage),
  [LLM extension points](#llm-extension-points),
  [Numerical data and parallel work](#numerical-data-and-parallel-work),
  [UI and progress](#ui-and-progress), and
  [Matching kernels](#matching-kernels).
- Make and verify changes: [Development workflow](#development-workflow),
  [Tests and packaging](#tests-and-packaging), and
  [Documentation ownership](#documentation-ownership).

## Development environment

The project supports Python 3.10 and newer. With uv:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

The equivalent pip installation is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install `.[codex]` or `.[literature]` only when working on those integrations.

## Architecture

The repository has one runtime framework, extended by Plan and numerical
stages:

- `validate`, `plan`, and `run` share one manifest and contract system.
- Contracts and Python validators own hard constraints. Prompts explain the
  physical judgement required from the model.
- Numerical inspection, fitting, selection, publication, and reporting are
  deterministic workflow code, not model-callable tools.
- Every job owns its parameters, resolved inputs, state, RNG, artifact
  directory, and LLM session.
- Stage-specific code stays under its stage; reusable code moves to shared
  modules only when multiple features need it.
- All terminal output, prompts, cancellation, and progress go through `ui.py`.

### Dependency direction

```text
__main__.py
    └── agent.py
        ├── manifest.py ── contract.py
        ├── llm.py ── structured.py
        ├── ui.py
        ├── plan/ ── plan/tools/
        └── stages/<stage>/workflow.py
                ├── private job helpers
                ├── physics.py ── data.py / parallel/ / kernels/
                ├── ask/ ── LlmSession
                └── reporting.py
```

The dependency direction expresses ownership:

- `__main__.py` translates CLI arguments into framework calls; it contains no
  manifest policy or physics.
- `agent.py` schedules jobs and conversations; it does not implement stage
  algorithms or stage-specific parameter policy.
- `manifest.py` discovers contracts and resolves sources; it does not import
  stage workflows.
- `contract.py` is domain-neutral. Stage contracts may reference stage-owned
  null hooks, but the shared rule engine knows nothing about LaMET stages.
- `workflow.py` may use `ToolContext`, stage-private helpers, and shared
  numerical modules. It owns sequencing and publication, not reusable formulas.
- `physics.py` operates on explicit numerical inputs and parameters. It should
  remain callable outside the agent; renderer-neutral progress and an optional
  shared pool are the only execution-framework concerns it should accept.
- `reporting.py` consumes completed records. It never changes decisions or
  reruns numerical work.
- `parallel/` contains execution primitives and emits no terminal output.
- `ui.py` contains no stage knowledge.

The common control flow is:

```text
manifest
   │
   ▼
validate ── issues ──► Plan conversation ── acceptance ──► saved manifest
   │                                           │
   │ valid                                     └─ Run continues when it entered
   ▼                                              Plan after failed validation
ordered jobs
   │
   ▼
resolve inputs → ToolContext → recommendations → stage workflow
   │
   ▼
output + job summaries → stage report
```

Standalone Plan stops after writing the accepted manifest. It is not a second
agent framework.

## Manifest and contract lifecycle

`load_manifest()` parses JSON or JSONC and retains the absolute source path.
Relative `metadata.root_directory` values are resolved from that path;
root-relative file inputs and `artifacts_directory` are then resolved from the
resulting root directory.

Validation runs in the following order:

1. Validate the common manifest envelope and metadata.
2. Load each authored stage's `contract.py` from the filesystem.
3. Apply stage defaults independently to every job and validate its rules.
4. Run stage `CHECKS` after structural issues for that job have cleared.
5. Check global job-ID uniqueness, prior-job references, file sources, and DAG
   order.
6. Validate each stage-owned systematics declaration.
7. Expand systematic variants into a copy of the job graph and validate the
   generated graph again.

Only a fully successful validation updates `Manifest.document` with defaults
and expanded jobs. The authored `systematics` block is removed from that
resolved document, which is later written as `resolved_manifest.json`. Plan
validates a deep copy, so validation never silently normalizes the editable
candidate.

### Contract rules

Contracts are dependency graphs traversed from their root. Rules outside an
inactive branch are not evaluated.

| Rule | Meaning |
| --- | --- |
| `Depends(parent, child, ...)` | The child is required when its parent branch is active. It may carry a runtime `null_hook`. |
| `Recommends(parent, child, default=...)` | Fill a deterministic default when the child is omitted. |
| `Suggests(parent, source, target, ...)` | Shallow-copy a defaults mapping into a target mapping before target values override it. |
| `Provides(parent, child, selector, ...)` | Activate a virtual branch when a selector chooses `child`; no extra manifest object is created. |
| `List(path, item, ...)` | Declare a list and the logical item path used by descendant rules. |
| `Value(path, expected, ...)` | Validate a Python type, `Literal` choice, and optional intrinsic predicate. |
| `Source(path, ...)` | Declare allowed job, file, constant, or recursive-list input forms. |

Every rule carries a `physics` explanation. `question` is optional wording for
Plan, but it is not a substitute for the physical explanation. Cross-field
checks receive a read-only `CheckContext` containing the complete manifest,
stage/job identity, effective parameters, inputs, and unresolved null-hook
paths. They return `Issue` objects rather than raising for ordinary validation
failures.

`stage_job_rules()` scopes parameter and input rules under each item in the
stage's ordered `jobs` list. It also applies the stage `defaults` mapping before
explicit job values, supplies an empty `inputs` mapping when omitted, and
validates job IDs.

## Component ownership

| Component | Responsibility |
| --- | --- |
| `__main__.py` | CLI parsing and dispatch. |
| `agent.py` | Shared session, Plan conversation, job loop, `ToolContext`, transcripts, and report boundaries. |
| `manifest.py` | Loading, stage discovery, validation, DAG construction, source resolution, and systematics expansion. |
| `contract.py` | Declarative rules, defaults, null hooks, issues, and cross-field checks. |
| `llm.py`, `structured.py` | Provider adapters, message history, usage, and JSON schemas. |
| `ui.py`, `banner.py` | Terminal interaction, semantic output, progress, and cancellation. |
| `data.py` | `EnsembleInfo`, `EnsembleData`, resampling, and NetCDF serialization. |
| `plan/` | Reversible manifest state, controller prompt, and Plan tools. |
| `stages/` | Stage contracts, workflows, physics, recommendations, systematics, and reports. |
| `parallel/` | Shared process pool, fitting, Fourier, and Lanczos primitives. |
| `kernels/` | Renormalization and matching kernels plus formula documents. |
| `literature/` | Review catalogs and classification utilities. |
| `plotting.py`, `stages/_reporting.py` | Shared plotting and report helpers. |

The package root is reserved for framework-wide concerns. Feature code belongs
under `plan/`, `stages/`, `parallel/`, `kernels/`, or `literature/`.

When deciding where a change belongs:

- a new manifest field starts in the owning stage contract, not in Plan or the
  CLI;
- a new fit or transform starts in stage physics, then is orchestrated by the
  workflow;
- a new question to the model starts in stage `ask/`, not as a general tool;
- a model-selected action belongs in a tool package only when choosing whether
  and when to invoke it is part of the model's task;
- a reusable numerical routine moves to `parallel/` only after at least two
  execution paths need the same semantics;
- a user-visible message or progress renderer belongs in `ui.py`; callers emit
  semantic events instead of terminal escape sequences;
- files prefixed with `_` are private implementation units and must not become
  manifest or plugin identifiers.

## Job execution lifecycle

Jobs are flattened in authored stage order. Each receives an artifact directory
at `<artifacts>/<stage-index>_<stage-id>/<job-id>/`. A string input resolves to
an earlier job's in-memory output and summary; `{ "file": ... }` resolves to a
path; `{ "json": ..., "id": ... }` preserves the selected descriptor record;
lists are resolved recursively. Forward job references are rejected during
validation.

For every job, the agent constructs a `ToolContext` with:

| Field | Contents |
| --- | --- |
| `manifest`, `manifest_path` | Resolved manifest and its authored source. |
| `stage_id`, `job_id` | Stable job identity. |
| `params` | Effective stage parameters after defaults and runtime recommendations. |
| `inputs`, `input_summaries` | Resolved values and upstream job summaries. |
| `state` | Private mutable scratch state for this job only. |
| `artifact_directory` | The job-owned output directory. |
| `rng` | A reproducible NumPy generator derived from run seed and job position. |
| `workers`, `_parallel` | Run-wide concurrency limit and shared process pool. |
| `output`, `summary` | Terminal fields set once by `context.finish()`. |

The lifecycle is:

1. Resolve authored inputs from completed jobs or files.
2. Create the context, RNG, transcript, and per-job `LlmSession`.
3. Apply `Recommends` defaults and resolve any active `Depends.null_hook` in
   dependency order.
4. Run the deterministic workflow, or the Review preflight and tool loop.
5. Require `context.finish()` and verify every declared artifact exists.
6. Write `summary.json` and compact `review_summary.json`.
7. Retain output and summary for downstream source resolution.
8. At the end of the stage, pass all `StageReportRecord` objects to its reporter.

The shared summary envelope separates three concerns:

- `result`: a short result type;
- `decisions`: selected physical/numerical choices;
- `diagnostics`: quality metrics, warnings, and candidate evidence;
- `artifacts`: relative paths produced by the job.

Do not put large sample arrays in the summary. Publish sample-bearing results as
`EnsembleData`/NetCDF and keep detailed candidate tables in job artifacts.

### Artifact layout

A typical resolved run has:

```text
artifacts/
├── resolved_manifest.json
├── 01_correlator_analysis/
│   ├── <job-id>/
│   │   ├── output.nc
│   │   ├── summary.json
│   │   ├── review_summary.json
│   │   ├── llm_transcript.md
│   │   ├── diagnostics/
│   │   └── plots/
│   └── report.md
└── ...
```

Not every job produces every optional directory. `summary.json` is the
canonical terminal record; `review_summary.json` is the bounded evidence view
used by Review. Stage reports aggregate completed records but do not replace
job-level diagnostics.

## Extending a stage

A numerical stage normally has this shape:

```text
<stage>/
├── contract.py        manifest and input authority
├── workflow.py        deterministic orchestration
├── physics.py         reusable numerical implementation
├── ask/               optional typed recommendations
├── systematics.py     optional variant expansion
├── reporting.py       optional aggregate report
└── _*.py              private inspection, fitting, selection, and publication
```

### Contract

Define `PARAM_RULES`, `INPUT_RULES`, `JOB_RULES`, and `CHECKS`. Use shared rule
types for structural constraints and `CHECKS` for relationships that cannot be
expressed declaratively. Do not duplicate these rules in prompts.

Stages with systematic variants also define `SYSTEMATICS_RULES` and
`SYSTEMATICS_CHECKS`. Their `systematics.py` exports:

```python
def expand(document: dict, config: dict, state: dict) -> None:
    ...
```

Expansion must be deterministic, preserve authored central jobs, reject ID
collisions, and record downstream mappings in `state`.

### Workflow and artifacts

The five standard numerical stages are listed in `_WORKFLOW_STAGES` in
`agent.py`. Their `workflow.py` exports:

```python
def run(context: ToolContext, session: LlmSession) -> None:
    ...
```

Add a new deterministic stage to `_WORKFLOW_STAGES`; otherwise it is treated as
a model-driven tool stage. Complete a workflow exactly once with
`context.finish(output, summary)`. The summary keys are:

```text
stage_id, job_id, result, decisions, diagnostics, artifacts
```

Artifact paths are relative to the job directory and must exist before
`finish`. Reusable mathematics belongs in `physics.py`; job state and artifact
orchestration belong in the workflow or private helpers.

An optional `reporting.py` exports
`write_stage_report(*, records, artifact_directory)`. Reporting consumes
completed records and must not rerun fits or change published results.

### Adding a numerical stage

Use a lowercase identifier matching `[a-z][a-z0-9_]*`; stage discovery is based
on the directory name and requires `contract.py`. A minimal implementation
sequence is:

1. Add the stage package and contract exports.
2. Implement numerical functions independently of `ToolContext` where practical
   so they can be unit-tested directly.
3. Add `workflow.run()` and register the stage in `_WORKFLOW_STAGES`.
4. Preserve physical metadata needed by downstream stages in output attrs.
5. Publish a complete summary and declared artifacts through `context.finish()`.
6. Add `reporting.py` when more than the per-job summary is needed.
7. Add a systematics compiler only for an approved deterministic variant policy.
8. Cover contract failures, physics, workflow publication, and report rendering
   in focused tests.

The ordered keys under `manifest.stages` define execution order. No central
stage registry owns that order, and a stage must not depend on a later job.

## LLM extension points

The three LLM surfaces intentionally use different interfaces:

| Surface | Invocation | Purpose | Schema source | Prompt file |
| --- | --- | --- | --- | --- |
| Runtime `ask` | The deterministic workflow requests a recommendation. | Fill or revise a small typed parameter set. | `TypedDict`/annotations plus task-specific JSON-schema constraints. | `prompt.md` under the specific ask package. |
| Stage `tools` | The model chooses a tool during a bounded conversation. | Inspect selected evidence or perform an explicitly agentic action. | Keyword-only annotations on `run()`. | `prompt.md` beside the tool. |
| Plan `tools` | The model chooses a tool while editing a manifest candidate. | Read, patch, validate, undo, discover paths, save, cancel, or finish Plan. | Explicit `PARAMETERS` mapping. | `prompts.md` beside the tool. |

The distinction is semantic, not only directory naming. Use `ask` when the
program already knows that it needs a recommendation and only the returned
value is uncertain. Use a stage tool when deciding whether, when, or with what
arguments to perform an action is part of the model's task. Plan tools are a
separate controller interface because their state and safety rules differ from
numerical jobs.

### Runtime ask packages

Correlator and Fourier recommendations live under:

```text
<stage>/ask/
├── __init__.py             shared evidence, caching, initial/retry routing
└── ask_for_<task>/
    ├── __init__.py         request and response contract
    └── prompt.md           physical meaning and judgement criteria
```

The request sequence is:

1. A contract `null_hook` or workflow decides that a recommendation is needed.
2. `ask.ensure(context, session)` adds job evidence under one stable context key.
3. The task builds a request containing requested fields, fixed parameters, and
   previous candidate diagnostics when retrying.
4. A response schema is derived from the return annotation and narrowed to the
   fields requested in this turn.
5. `LlmSession.complete(..., response_schema=...)` makes a structured call with
   no model tools.
6. The decoded value is checked again with `validate_value()` and task-specific
   invariants.
7. Null-hook values are applied atomically and the complete contract/check set
   is rerun; invalid values restore the previous parameters.

An ask function does not let the model choose or execute a fit. It supplies only
parameters to the deterministic candidate workflow. Evidence added with
`session.add_context()` is sent once, while task prompts added with
`add_system_prompt()` remain in that job's history. Initial suggestions may be
cached in `context.state` so multiple null hooks do not cause duplicate calls.

`metadata.parameter_recommendation_retries` sets the number of extra structured
recommendation calls allowed after the initial request. A retry should include
the attempted parameter-to-quality mapping or numerical failure, not merely ask
the same question again.

### Stage tools

Model-driven stages have a stage `prompt.md` and immediate tool directories:

```text
<stage>/
├── prompt.md
└── tools/<name>/
    ├── __init__.py
    └── prompt.md
```

Tool directory names match `[a-z][a-z0-9_]*`. The module exports a callable
`run()` whose first argument is `context`; all model-visible arguments are
keyword-only and type-annotated. Supported annotations are converted by
`structured.annotation_schema()`, and additional properties are rejected.

A tool returns an observation mapping containing at least a string `summary`.
It may update stage-owned `context.state`, write artifacts, or call
`context.finish()` when it is the terminal publication action. The conversation
continues until the context is finished or its turn/tool budget is exhausted.

Review is the only current model-driven stage. Its deterministic preflight runs
in this order:

```text
inspect selected results and reports
    → check manifest-chain consistency
    → rank literature candidates
    → start the Review tool conversation
```

The preflight outputs, including consistency findings, are injected into the
initial Review context. Ordinary `review_summary.json` evidence is capped to 60
plot-grid points; `read_full_resolution` lets the model request one selected
job's complete grid. `read_papers` controls full-text literature access, and
`write_review` is the terminal tool that authors and publishes the report.
Deterministic preflight functions stay private and are never exposed as tools.

### Plan tools and state

Plan tools operate on one `PlanState` rather than `ToolContext`:

```text
plan/tools/<name>/
├── __init__.py    exports PARAMETERS and run(state, arguments)
└── prompts.md     capability, usage, and observation guidance
```

The plural filename is deliberate and distinct from stage tool prompts. Add the
module to the stable registry in `plan/tools/__init__.py`; Plan tools are not
implicitly enabled merely because a directory exists.

`PlanState` owns four related views:

- `original`: the authored document loaded at entry;
- `candidate`: the current in-memory document;
- `issues`/`packets`: validator results plus related contract subtrees;
- `revisions`: prior candidates and edit descriptions used by undo.

Issue packets include the failing path, current and parent values, physical
explanation, allowed children, and all related descendant rules. This lets the
model ask coherent questions without placing field-specific policy in the Plan
controller prompt.

Manifest mutation uses the `add`, `replace`, and `remove` subset of JSON Patch.
Patches are restricted to `/metadata`, `/stages`, and `/systematics`, applied to
a deep copy, recorded for undo, and followed immediately by a complete
validation refresh. `/edit` replaces the candidate only after parsing the
external editor result; saves use a temporary file and atomic replacement.

The LLM may interpret natural-language requests as edits or as `/show`,
`/issues`, `/undo`, `/save`, or cancellation intent. Only explicit final user
acceptance permits the planned manifest to be written as the accepted result.

### Prompt ownership

| Prompt | Owns | Must not own |
| --- | --- | --- |
| `plan/prompt.md` | General planning dialogue, sequencing questions, interpreting user intent, and acceptance behavior. | Stage-specific parameter rules or copied schemas. |
| Plan tool `prompts.md` | One manifest operation and how to interpret its observation. | Other tools' policies or direct file mutation outside `PlanState`. |
| Ask `prompt.md` | Meaning of incoming data, physical fit/range judgement, and quality interpretation for one task. | Invocation policy, deterministic candidate enumeration, or duplicated response validation. |
| Stage `prompt.md` | Goal and decision policy of a model-driven stage. | Numerical implementation details already enforced by code. |
| Stage tool `prompt.md` | Capability, appropriate use, argument meaning, and returned evidence. | Global conversation policy. |

### Sessions and providers

`LlmSession` owns message history, one-time system prompts, one-time numerical
context, call counts, recommendation budgets, and transcript recording. Calls
using `user_message` extend the retained job history; calls supplying an
explicit `messages` list are owned by the surrounding conversation loop.

Codex maps a stable prompt digest and first user request to a persistent,
read-only, ephemeral cached-login thread. Existing threads receive only the new
turn. OpenAI-compatible providers receive the accumulated message history on
each request. Both paths normalize responses into the same text, tool-call,
structured-value, and usage representation.

Registered API providers are defined in `_OPENAI_COMPATIBLE_API` in `llm.py` as
base URL, environment-variable name, and default model. Adding one requires:

1. a registry entry;
2. request/model-discovery tests;
3. README provider and environment-variable documentation.

Keep provider authentication and transport behavior inside `llm.py`. No stage
may branch on provider identity.

## Numerical data and parallel work

Numerical stages exchange `EnsembleData`, whose first dimension is `resample`:

```python
from lamet_agent.data import EnsembleData

data.to_netcdf("output.nc")
loaded = EnsembleData.from_netcdf("output.nc")
```

NetCDF uses native complex support. `gvar` values are stored as aligned mean and
standard-deviation variables and reconstructed on load.

`EnsembleData` enforces these invariants:

- physical dimensions never include the reserved `resample` name;
- every physical dimension has an explicitly sized coordinate;
- `raw`, `jackknife`, and `bootstrap` data contain a list of samples;
- `gvar` data use a length-one resample dimension around scalar/array gvars;
- ensemble metadata is either one `EnsembleInfo` record or `None`;
- stage provenance belongs in JSON-compatible attrs and must be preserved when a
  transformation does not intentionally change it.

Correlators selected together are resampled with one plan and carry a shared
`resample_id`. Downstream code must not silently align unrelated replicas by
array index. When a stage intentionally changes grids, units, schemes, or sample
semantics, update attrs so Review consistency checks can distinguish a physical
transformation from lost provenance.

The agent supplies one bounded process pool for a run. Stage workflows should
pass it to shared fitting, Fourier, or Lanczos functions rather than create
nested pools. `parallel/` owns no UI; the stage that knows the logical unit of
work owns progress reporting.

Use `FitNumericalError` for a numerically attempted candidate that failed and
may participate in an approved retry/selection policy. Use `ValueError` or
`TypeError` for invalid inputs, broken provenance, or programmer-facing contract
violations. Do not catch broad exceptions to turn structural bugs into low-fit
quality.

## UI and progress

`PlainUi` supports redirected output, tests, and terminals without the full TUI.
`TerminalUi` owns prompt-toolkit history, completion, multiline input, colors,
progress rendering, and cancellation. `create_ui()` selects between them from
terminal capabilities, while `use_ui()` binds the active instance through a
context variable.

Code outside `ui.py` should use:

- `log()` for ordinary semantic messages;
- `warning()` for user attention;
- `track()` for a renderer-neutral iterable progress unit;
- `ToolContext.state["show_job_progress"]` when a stage needs to honor the
  selected progress ownership.

Progress totals describe logical stage or physics work, not worker-process
events. Sample fits may run in parallel underneath one logical z/model/job unit.
Do not update the same progress task from child processes. Cancellation is
reported through `UiCancelled` and must unwind pools and UI contexts cleanly.

## Matching kernels

Public matching kernel IDs are filename stems in `lamet_agent/kernels/`. Each
matching kernel has a pair:

```text
<kernel_id>.py    exports kernel(...)
<kernel_id>.md    formula and provenance
```

The Python signature defines accepted `kernel_parameters`; shared private
helpers stay in `implementation.py`. Coordinate-space renormalization kernels
such as `z_msbar_*` are loaded separately and need no paired document.

## Development workflow

This workflow applies to both human contributors and coding agents. Current
contracts, workflows, and tests are the primary authorities. Generated runs,
old artifacts, and the legacy repository may provide context, but they do not
define this repository's behavior.

### Locate the owner

| Change | Primary owner | Related work |
| --- | --- | --- |
| Manifest field or validation | Stage `contract.py` | Examples, Plan packets, contract tests. |
| Numerical method or fit policy | Stage `physics.py` and private helpers | Workflow, diagnostics, reporting, physics tests. |
| Stage sequencing | Stage `workflow.py` | Inspection, selection, publication, agent-loop tests. |
| Fit/range recommendation | Stage `ask/` | Null hook, schema, prompt, retry evidence, transcript tests. |
| Model-selected action | Stage `tools/` | Stage prompt, schema discovery, conversation tests. |
| Plan behavior | `plan/state.py`, `plan/tools/` | Controller prompt, UI, planning tests. |
| CLI or run lifecycle | `__main__.py`, `agent.py` | Manifest, UI, README, core tests. |
| Output or progress | `ui.py` and the owning stage | UI/reporting tests. |
| Provider or structured response | `llm.py`, `structured.py` | Usage display, provider tests, README. |
| Matching kernel | Kernel `.py`/`.md` pair | Contract validation and numerical tests. |
| Packaging or dependency | `pyproject.toml`, `MANIFEST.in` | Editable install and distribution build. |

### Make the change

1. Inspect `git status` and the existing diff. Preserve unrelated tracked and
   untracked work.
2. Read the owning module, its boundary modules, and focused tests before
   editing.
3. Decide whether the request changes a public contract, numerical policy, LLM
   boundary, or only an implementation detail.
4. Implement the smallest coherent change at the owning layer.
5. Add or update tests at the same abstraction level.
6. Update README for user-visible behavior and this document for architecture or
   extension changes.

Do not add unrequested fallback policy, retries, compatibility layers, or model
authority. Do not move stage code into the framework root, expose deterministic
functions as tools, create hidden random state or nested pools, or bypass the UI
with direct ANSI/progress output.

Unit tests use temporary manifests and data rather than modifying examples,
local datasets, or generated runs. `uv.lock` is intentionally untracked. Do not
commit or push unless explicitly requested; before a requested commit, inspect
the staged diff and include only approved changes.

## Tests and packaging

Run the full suite and linter from the repository root:

```bash
python -m pytest
python -m ruff check .
```

Tests should be layered with the implementation:

- contract tests exercise missing fields, wrong types/choices, defaults,
  inactive branches, and cross-field checks;
- manifest tests use temporary stage roots and files to cover discovery, DAG,
  path, and systematics behavior without coupling unit tests to examples;
- physics tests call numerical functions directly and compare stable physical
  quantities or invariants rather than incidental formatting;
- workflow tests use a temporary `ToolContext`, mock recommendations where
  needed, and verify selection, `context.finish()`, provenance, and artifacts;
- LLM tests use fake backends and assert the exact request/schema/turn behavior;
  unit tests never call external providers;
- reporting tests consume representative completed records and verify rendered
  scientific content and declared links;
- UI tests use `PlainUi` or isolated prompt-toolkit inputs and do not depend on
  the developer's terminal.

| Area | Tests |
| --- | --- |
| Manifest, contracts, agent loop | `tests/unit/test_core.py` |
| Plan | `tests/unit/test_planning.py` |
| Data and serialization | `tests/unit/test_data.py` |
| Numerical stages | `tests/unit/test_stage_physics.py`, `tests/unit/test_physics.py` |
| Review | `tests/unit/test_review.py`, `tests/unit/test_review_evidence.py` |
| Reports and UI | `tests/unit/test_reporting.py`, `tests/unit/test_ui.py` |

After contract changes, validate all bundled manifests:

```bash
for manifest in examples/*_manifest.json; do
  lamet-agent validate "$manifest" || exit 1
done
```

After changing package data, build both distributions:

```bash
uv build
```

Root documents included in source distributions are listed in `MANIFEST.in`.
Package-owned prompts, catalogs, and kernel documents are declared in
`pyproject.toml`.

Inspect the built archives when adding a new prompt, catalog, formula document,
or root Markdown file. An editable checkout can hide missing package-data rules
that only fail after installation from a wheel.

## Documentation ownership

- `README.md` is for users: installation, CLI, manifests, input files, outputs,
  and supported physics workflows.
- `DEVELOPMENT.md` is the coding-agent and contributor contract: architecture,
  ownership, extension protocols, and verification.
- Contract `physics` strings explain why a manifest field or check exists and
  are supplied to Plan as structured evidence.
- Ask/tool prompts are instructions for one model task, not general developer
  documentation.
- Kernel Markdown files document the formula paired with a public kernel ID.
- Generated reports and transcripts document one run; they do not define future
  behavior.

When behavior changes, update the narrowest authoritative source first, then any
user or developer documentation that describes that behavior. Avoid copying the
same rule into several prompts and documents.
