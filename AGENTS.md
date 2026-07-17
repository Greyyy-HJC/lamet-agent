# AGENTS.md

Project-specific instructions for coding agents working in this repository.

## Purpose

`lamet-agent` is a Python-first LaMET workflow framework. The repository should stay explicit, readable, and easy to extend as more domain-specific analysis logic is integrated.

## Think Before Coding

Do not guess when requirements are ambiguous.

- State assumptions explicitly.
- Surface tradeoffs when multiple implementations are valid.
- Ask for clarification before implementing if key intent is unclear.
- Prefer a simpler path when it satisfies the same goal.

## Simplicity First

Implement only what is needed for the task at hand.

- No speculative abstractions.
- No optional features unless requested.
- Prefer locally understandable logic over clever indirection.
- Keep stage tool contracts stable unless related stages are intentionally evolved together.

## Surgical Changes

Touch only what is required for the current task.

- Do not refactor unrelated files without an explicit request.
- Preserve existing style and conventions in touched files.
- Keep reusable logic in `src/lamet_agent/`; keep `examples/` scripts as thin wrappers.
- Add comments only where logic is non-obvious.

## Goal-Driven Execution

Define success before coding and verify outcomes after coding.

- Prefer tests or smoke checks when interface or behavior changes.
- Validate that stage tool outputs remain consumable downstream.
- Ensure changed documentation and code paths stay consistent.

## Workflow Hygiene

- Before any `git add` or `git commit`, check whether `.gitignore` needs updates.
- Before any `git add` or `git commit`, check whether every relevant `README` file needs updates.
- After each meaningful implementation pass, check whether `PROJECT_LOG.md` should receive an append-only entry.

## Project-Specific Rules

- Keep repository documentation, comments, and docstrings in English.
- Use the repository-root `.venv` as the default Python environment.
- Keep dependency and setup guidance consistent with the active package workflow (`pyproject.toml` extras and editable installs).
- Every executable Python script must start with a module docstring that includes:
  - script purpose
  - expected inputs and outputs
  - example usage

## Documentation Maintenance

- Keep `README.md` as the human-facing project entry point (setup, CLI, file map).
- Keep `PLAN.md` for long-form product and physics workflow notes.
- Keep `PROJECT_LOG.md` as an append-only engineering log.
- Keep `AGENTS.md` as the durable, primary ruleset for coding agents.

## Module Map

Top-level layout:

```text
.
├── examples/
│   ├── fake_data/generate_fake_data.py
│   └── pion_pdf_cg_manifest.json
├── src/lamet_agent/
│   ├── agent.py
│   ├── cli.py
│   ├── kernels.py
│   ├── manifest.py
│   ├── core/
│   └── stages/
└── tests/unit/
```

Package modules:

- `src/lamet_agent/cli.py`: CLI for `validate` and `run`.
- `src/lamet_agent/agent.py`: stage/job DAG runner and per-job LLM tool loop.
- `src/lamet_agent/manifest.py`: `metadata`/`inputs`/`stages` schema, path resolution, and DAG validation.
- `src/lamet_agent/manifest_params.py`: central `STAGE_PARAM_CONTRACTS` registry and recursive stage `defaults`/job `params` validation.
- `src/lamet_agent/kernels.py`: built-in matching kernels.
- `src/lamet_agent/core/stages.py`: stage-id → package routing.
- `src/lamet_agent/core/tools.py`: resolves `STAGE_TOOLS`, prepares tool args, plot paths under `artifacts/`.
- `src/lamet_agent/core/llm.py`: `LlmSession` backends (`mock`, `external`, `api`, `codex`); OpenAI-compatible HTTP providers in `PROVIDERS`; `parse_api_model()` for `provider/model_id` CLI specs.
- `src/lamet_agent/core/prompting.py`: system prompt and per-stage static context assembly.
- `src/lamet_agent/core/trace.py`: optional ReAct-style stdout trace (`--verbose`).
- `src/lamet_agent/core/data.py`: typed ensemble containers and cross-stage data helpers.
- `src/lamet_agent/core/plotting.py`: shared plotting conventions and helpers.
- `src/lamet_agent/stages/`: stage packages, each with `functions.py`, `prompts.py`, and `skills.py`:
  - `correlator` (`correlator_analysis`)
  - `renorm` (`renormalization`)
  - `fourier` (`fourier_transform`)
  - `matching` (`perturbative_matching`)
  - `extrapolation` (`extrapolation`)
  - `review` (`review`)
- `examples/`: smoke manifests, fake data generator, and local workflow examples.
- `tests/unit/`: schema, CLI, agent loop, tools, trace, and stage tests.
- `runs/`: typical output location for logged runs (gitignored); artifact placement comes from the manifest.

## How To Add A New Stage

1. Add the stage id to `StageId` in `manifest.py`, `STAGE_TO_PACKAGE` in `core/stages.py`, and `STAGE_PARAM_CONTRACTS` in `manifest_params.py`.
2. Create `src/lamet_agent/stages/<package>/` with:
   - `functions.py`: stage tools and a `STAGE_TOOLS` dict mapping tool names to callables `(store, **kwargs) -> dict`.
   - `prompts.py`: stage instruction text and action protocol for the LLM.
   - `skills.py`: `STAGE_SKILL` strategy text, `tool_catalog()`, and `validate_stage_inputs(manifest, job)`.
3. Register tools only through `STAGE_TOOLS`; `core/tools.resolve_stage_tools()` imports them dynamically.
4. Extend `core/prompting.py` if the new stage needs shared prompt fragments.
5. Add unit tests under `tests/unit/` and, when appropriate, extend a dedicated example manifest.

## How To Add A New Script Or Example

1. Put reusable logic in the package, not in the example script.
2. Keep example scripts as thin wrappers around package APIs.
3. Start the file with a module docstring that includes example usage.
4. Prefer manifests under `examples/` for runnable workflow demos.

## How To Integrate Existing Analysis Code

- Land exploratory or legacy code outside `src/lamet_agent/` only when it is not yet ready for the tool-registry contract.
- Prefer thin wrappers that expose fixed Python tools in `stages/<package>/functions.py` over copying large procedural scripts into the agent loop.
- Keep file-format assumptions localized to the stage that reads them (or to `core/data.py` when shared).
- Convert legacy conventions at tool boundaries so manifest paths, store keys, and observations stay uniform.
- Preserve per-stage store keys and observation shapes unless a coordinated contract update is explicitly intended.

## Manifest Conventions

- Required top-level fields are `metadata`, `inputs`, and `stages`.
- `metadata.stages` is the sole execution order; stage selection is not a CLI override.
- Stage entries contain `defaults` and `jobs`; job `params` shallow-merge over defaults.
- Stage `defaults` and job `params` reject keys not declared for that stage in `STAGE_PARAM_CONTRACTS`; never add a manifest parameter without adding its consumption path and contract entry together.
- Correlator jobs group `inputs.correlators` by `correlator_ids`; downstream jobs reference earlier job ids through role-named `inputs`.
- All ids are globally unique. External partial-run sources are declared in `inputs.artifacts`.
- Paths resolve from `metadata.root_directory`; default job artifacts are `<artifacts_directory>/<stage>/<job_id>.nc`.
- Terminal stage tools must place their primary in-memory result in `store["output"]`.
- Fourier jobs consume role `input`; matching jobs consume role `quasi`. Standard partial-run NetCDF artifacts read discrete `momentum`, `volume`, and `lattice_spacing_fm` from attrs; legacy files may declare the complete triple under `inputs.artifacts`, and physical `momentum_gev` is derived.
- `inputs.kernels[].stage` must be a full `StageId` (`renormalization` or `perturbative_matching`), not a package shorthand such as `matching`.
- Hybrid `zs_fm` is a flat stage/job parameter under `renormalization` and `perturbative_matching`; do not place it in `inputs.kernels[].kernel_parameters` or renormalization `scheme_parameters`.

## Plotting Conventions

- All stage plots must use `src/lamet_agent/core/plotting.py`.
- Use `default_plot()` and the exported helpers instead of direct `plt.subplots()` or `plt.figure()` in stage code.
- Reuse exported style constants (`COLOR_CYCLE`, `ERRORBAR_STYLE`, `FIG_SIZE`, `LEGEND_SETS`) for consistent publication-style output.
- Correlator plot tools must write PDFs under the job's manifest-controlled stage artifact directory.

## Testing Expectations

- Add or update unit tests for manifest schema, CLI, tools, and stage behavior when interfaces change.
- Add or extend correlator tool tests when changing `stages/correlator/functions.py`.
- Prefer small toy arrays and deterministic smoke kernels for tests.
- Install `[dev]` and optional `[analysis]` extras from `pyproject.toml` when tests need `gvar`, `lsqfit`, or HDF5 I/O.
