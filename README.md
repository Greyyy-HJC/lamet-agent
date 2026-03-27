# lamet-agent

`lamet-agent` is a CLI-first scaffold for building agentic LaMET analysis workflows without requiring an LLM backend in the first iteration.

The repository is organized around a rule-based workflow engine that validates a JSON manifest, loads correlator inputs, executes staged analysis steps, writes normalized intermediate artifacts, and exports a final distribution-like observable.

## Goals

- Keep the analysis code in plain Python modules that are easy to read and extend.
- Make workflow resolution explicit and inspectable before model-based planning is introduced.
- Standardize inputs, outputs, stage contracts, and plotting conventions from the start.
- Leave clean extension points for future user analysis code and future agent backends.
- Keep future backend authentication explicit, with first-class OAuth support for `codex` and `claude_code`.

## Initial Workflow Stages

The default pipeline covers:

1. `correlator_analysis`
2. `renormalization`
3. `fourier_transform`
4. `perturbative_matching`
5. `physical_limit`

The current implementations are intentionally simple demo versions. They exist to make the package runnable end-to-end while preserving stable interfaces for later physics-specific upgrades.

## Repository Layout

- `src/lamet_agent/`: package code, workflow engine, schema validation, reporting, plotting, and stage implementations.
- `src/lamet_agent/auth/`: OAuth provider configuration, PKCE flow helpers, and token storage.
- `src/lamet_agent/backends/`: descriptors for future provider-backed agent integrations.
- `scripts/`: thin executable wrappers for local development.
- `examples/`: toy correlator inputs, an example manifest, and a reference hard-kernel module.
- `incoming/analysis_steps/`: temporary landing zone for existing or draft analysis code before it is normalized into the package.
- `tests/`: schema, kernel, workflow, stage, and end-to-end smoke tests.

## Quickstart

Install the package and its dependencies in your preferred environment:

```bash
python -m pip install -e .
```

If you want test tooling as well:

```bash
python -m pip install -e .[dev]
```

Run the example workflow:

```bash
lamet-agent validate examples/demo_manifest.json
lamet-agent workflow examples/demo_manifest.json
lamet-agent run examples/demo_manifest.json
lamet-agent auth providers
```

If you are developing from the repository without installing the package:

```bash
python scripts/run_manifest.py run examples/demo_manifest.json
```

## Manifest Contract

The workflow manifest is JSON with the following top-level sections:

- `goal`: one of `parton_distribution_function`, `distribution_amplitude`, or `custom`.
- `correlators`: a list of correlator inputs with `kind`, `path`, `file_format`, `label`, and optional per-input metadata.
- `metadata`: run-level metadata. The initial scaffold requires at least `ensemble` and `conventions`.
- `kernel`: inline Python source plus `callable_name`. This source is executed dynamically and is treated as trusted input in v0.
- `workflow`: optional explicit stage list and per-stage parameters.
- `outputs`: output directory plus requested plot and data formats.

## Hard-Kernel Contract

The inline hard-kernel callable should accept at least two positional arguments:

```python
def my_kernel(coordinate_axis, values, metadata):
    ...
```

The current perturbative matching stage calls the kernel as:

```python
kernel(momentum_axis, fourier_magnitude, manifest_metadata)
```

Returning an array with the same shape as the input values is required.

## Outputs

Each run produces:

- stage-level data files and plots under `stages/<stage-name>/`
- short stage summaries under `summaries/`
- a top-level `report.md`
- a top-level `report.json`

Default formats are `png` for plots and `csv` for exported numeric data. The included example uses `svg` so it can run in environments without Matplotlib.

## OAuth Support For Agent Backends

The scaffold now includes provider-aware OAuth helpers for:

- `codex`
- `claude_code`

The implementation uses standard OAuth 2.0 authorization-code flow with PKCE, but it does not hardcode vendor endpoints or client registrations into the repository. Instead, you configure each provider through environment variables so the project can track provider-side changes without source rewrites.

Example configuration:

```bash
export LAMET_AGENT_CODEX_CLIENT_ID="your-codex-client-id"
export LAMET_AGENT_CODEX_AUTH_URL="https://example.com/oauth/authorize"
export LAMET_AGENT_CODEX_TOKEN_URL="https://example.com/oauth/token"
export LAMET_AGENT_CODEX_SCOPES="openid profile offline_access"

export LAMET_AGENT_CLAUDE_CODE_CLIENT_ID="your-claude-client-id"
export LAMET_AGENT_CLAUDE_CODE_AUTH_URL="https://example.com/oauth/authorize"
export LAMET_AGENT_CLAUDE_CODE_TOKEN_URL="https://example.com/oauth/token"
export LAMET_AGENT_CLAUDE_CODE_SCOPES="openid profile offline_access"
```

Available commands:

```bash
lamet-agent auth providers
lamet-agent auth status codex
lamet-agent auth login codex
lamet-agent auth logout codex
```

Stored tokens are written to `~/.config/lamet-agent/oauth/`.

## Future Direction

The current CLI uses a rule-based planner. The planner and backend interfaces are intentionally narrow so a future natural-language or model-based backend can reuse the same manifest, workflow, artifact, and authentication contracts.

## Incremental Code Migration

If you already have analysis code, place it under `incoming/analysis_steps/` first. After that, describe which script or function implements which LaMET step, and it can be refactored into the appropriate package location with cleaner interfaces and documentation.
