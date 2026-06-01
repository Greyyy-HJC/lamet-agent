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
- Keep stage contracts stable unless all related stages are intentionally evolved together.

## Surgical Changes

Touch only what is required for the current task.

- Do not refactor unrelated files without an explicit request.
- Preserve existing style and conventions in touched files.
- Keep reusable logic in `src/lamet_agent/`; keep `scripts/` as thin wrappers.
- Add comments only where logic is non-obvious.

## Goal-Driven Execution

Define success before coding and verify outcomes after coding.

- Prefer tests or smoke checks when interface or behavior changes.
- Validate that stage outputs remain consumable downstream.
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

- Keep `README.md` as the human-facing project entry point.
- Keep `DEVELOPMENT.md` focused on engineering plan and implementation details.
- Keep `SPEC.md` aligned with repository structure when structure changes.
- Keep `AGENTS.md` as the durable, primary ruleset for coding agents.
- Keep `CLAUDE.md` minimal and consistent with `AGENTS.md`.

## Module Map

- `src/lamet_agent/cli.py`: CLI surface for `validate`, `workflow`, and `run`.
- `src/lamet_agent/schemas.py`: manifest model and validation logic.
- `src/lamet_agent/planners/`: workflow planner implementations.
- `src/lamet_agent/workflows.py`: workflow execution entry point.
- `src/lamet_agent/stages/`: stage protocol, registry, and concrete stage implementations. The `evaluation` stage performs cross-family (cross-momentum) aggregation rather than per-family processing.
- `src/lamet_agent/loaders.py`: built-in correlator loaders.
- `src/lamet_agent/kernel.py`: inline hard-kernel compilation and validation.
- `src/lamet_agent/constants.py`: shared physics constants and perturbative running helpers.
- `src/lamet_agent/plotting.py`: shared plotting conventions and helpers.
- `src/lamet_agent/reporting.py`: markdown and JSON report generation.
- `src/lamet_agent/extensions/`: reusable low-level analysis helpers that stages compose.
- `incoming/analysis_steps/`: temporary intake area for legacy, draft, or not-yet-integrated analysis code.
- `examples/`: curated end-to-end workflows and tracked example data slices.
- `docs/analysis_model.md`: structured metadata contract and analysis taxonomy.

## How To Add A New Stage

1. Create a new module in `src/lamet_agent/stages/`.
2. Define a stage class with `name`, `description`, and `run(context)`.
3. Decorate the class with `@register_stage`.
4. Return a `StageResult` containing:
   - a concise summary string
   - structured payload data for downstream stages
   - normalized artifact records
5. Import the module from `src/lamet_agent/stages/__init__.py` so it registers automatically.
6. Update the rule-based planner if the new stage changes the default workflow.

## How To Add A New Script

1. Put reusable logic in the package, not directly in the script.
2. Keep the script as a thin wrapper around package APIs.
3. Start the file with a module docstring that includes example usage.
4. If the script is repo-local, ensure `src/` is importable from repository root.

## How To Integrate Existing Analysis Code

- Land raw or legacy code in `incoming/analysis_steps/` before moving it into the package.
- Prefer wrappers and adapters over copying large procedural scripts into the workflow engine.
- Keep file-format assumptions localized to `loaders.py` or dedicated adapter modules.
- Convert legacy input/output conventions at stage boundaries so the rest of the workflow remains uniform.
- Preserve stage payload contracts unless a coordinated contract update is explicitly intended.

## Analysis Model Conventions

- Preserve the three-layer structure:
  - reusable helpers in `src/lamet_agent/`
  - stage implementations in `src/lamet_agent/stages/`
  - end-to-end workflows in `examples/`
- Keep physics metadata in the manifest, not in ad hoc stage parameters.
- For full workflows, prefer structured `metadata.analysis` and `metadata.setups` over legacy free-form metadata.
- When adding new correlator families, keep selectors and emitted payload metadata unambiguous across `setup_id`, momentum, smearing, and operator choices.

## Plotting Conventions

- All plots must use `extensions/plot_presets`.
- Use `default_plot()` or `default_sub_plot()` instead of direct `plt.subplots()` or `plt.figure()` calls in stage/extension code.
- Reuse exported style constants (`PALETTE`, `COLOR_CYCLE`, `MARKER_CYCLE`, `ERRORBAR_STYLE`, `ERRORBAR_CIRCLE_STYLE`, `AXIS_FONT`, `SMALL_AXIS_FONT`) for consistent publication-style output.
- Apply this rule to every stage output figure (including cs-kernel, Fourier-transform, and effective-mass plots).

## Testing Expectations

- Add or update unit tests for schema, planner, and stage behavior when interfaces change.
- Add an end-to-end smoke test when a change affects the full workflow path.
- Prefer small toy arrays and deterministic smoke kernels for tests.
