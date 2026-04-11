# AGENTS.md

This document describes how to work in the `lamet-agent` repository when extending the scaffold through iterative coding sessions.

## Purpose

The repository is a Python-first LaMET workflow framework. The primary goal is to keep analysis code explicit, readable, and easy to upgrade as more domain-specific functions are integrated.

## Core Conventions

- Keep repository documentation, comments, and docstrings in English.
- Prefer adding reusable logic inside `src/lamet_agent/` and keep `scripts/` thin.
- Every executable Python script must start with a module docstring that states:
  - the script purpose
  - expected inputs and outputs
  - example usage
- Add comments only where logic is non-obvious.
- Keep stage interfaces stable. New analysis logic should fit the existing stage contract unless there is a clear reason to evolve the contract for every stage together.
- Before any `git add` or `git commit`, check whether the change requires updates to every relevant `README` file and to `.gitignore`.

## Module Map

- `src/lamet_agent/cli.py`: CLI surface for `validate`, `workflow`, and `run`.
- `src/lamet_agent/schemas.py`: manifest model and validation logic.
- `src/lamet_agent/planners/`: workflow planner implementations.
- `src/lamet_agent/workflows.py`: workflow execution entry point.
- `src/lamet_agent/stages/`: stage protocol, registry, and concrete stage implementations.  The `evaluation` stage is notable because it performs cross-family (cross-momentum) aggregation rather than processing each family independently.
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

1. Put the reusable logic in the package, not directly in the script.
2. Make the script a thin wrapper around package APIs.
3. Start the file with a module docstring including example usage.
4. If the script is meant for repo-local execution, ensure `src/` is importable when run from the repository root.

## How To Integrate Existing Analysis Code

- Land raw or legacy code in `incoming/analysis_steps/` before moving it into the package.
- Prefer wrapping existing functions behind stage helpers or extension modules instead of copying large procedural scripts into the workflow engine.
- Keep file-format assumptions localized to `loaders.py` or dedicated adapter modules.
- If an existing function has its own conventions, convert inputs and outputs at the stage boundary so the rest of the workflow stays uniform.
- When in doubt, preserve the stage payload contract and adapt the legacy code to it.

## Analysis Model Conventions

- Preserve the three-layer structure:
  - reusable helpers in `src/lamet_agent/`
  - stage implementations in `src/lamet_agent/stages/`
  - end-to-end workflows in `examples/`
- Keep physics metadata in the manifest, not in ad hoc stage parameters.
- For full workflows, prefer structured `metadata.analysis` and `metadata.setups` over legacy free-form metadata.
- When adding new correlator families, ensure selectors and emitted payload metadata remain unambiguous across `setup_id`, momentum, smearing, and operator choices.

## Plotting Conventions

- **All plots must use `extensions/plot_presets`.**  Call `default_plot()` to create a figure and axis with the shared publication style (Times New Roman, golden-ratio size, inward ticks on all sides, dotted grid).  Use the exported constants `PALETTE`, `COLOR_CYCLE`, `MARKER_CYCLE`, `ERRORBAR_STYLE`, `ERRORBAR_CIRCLE_STYLE`, `AXIS_FONT`, and `SMALL_AXIS_FONT` for consistent styling.
- Do not call `plt.subplots()` or `plt.figure()` directly in stage or extension code; go through `default_plot()` or `default_sub_plot()`.
- This rule applies to every plot produced by a stage (cs_kernel, Fourier transform, effective mass, etc.) and to any helper that generates a standalone figure.

## Testing Expectations

- Add or update unit tests for schema, planner, and stage behavior when interfaces change.
- Add an end-to-end smoke test when a change affects the full workflow path.
- Prefer small toy arrays and deterministic smoke kernels for tests.