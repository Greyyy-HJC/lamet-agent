# lamet-agent

`lamet-agent` is a Python-first scaffold for a LaMET/LQCD analysis agent.

## Core Idea

Keep only one necessary input contract:

- `correlators`: correlation-function datasets
- `kernels`: perturbative kernel Python functions

Kernel references use `module:function`, e.g. `lamet_agent.kernels:identity_kernel`.

## Minimal Structure

```text
.
├── examples/
│   ├── fake_data/
│   │   └── generate_fake_data.py
│   └── workflow_smoke_manifest.json
├── src/lamet_agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── cli.py
│   ├── kernels.py
│   ├── manifest.py
│   ├── prompts.py
│   └── skills.py
└── tests/unit/
    ├── test_schemas.py
    └── test_validation.py
```

`temp/` is ignored and not part of this structure.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Generate fake data:

```bash
python examples/fake_data/generate_fake_data.py
```

Validate and run manifest:

```bash
lamet-agent validate examples/workflow_smoke_manifest.json
lamet-agent run examples/workflow_smoke_manifest.json
```

Run with a real-model placeholder switch:

```bash
lamet-agent run examples/workflow_smoke_manifest.json --model mock
```

## File Responsibilities

- `src/lamet_agent/manifest.py`
  - Defines manifest schema (`correlators`, `kernels`, metadata).
  - Validates kernel references in `module:function` format.
- `src/lamet_agent/prompts.py`
  - Stores `SYSTEM_PROMPT` and per-stage prompt templates.
- `src/lamet_agent/skills.py`
  - Resolves stage sequence for a workflow goal.
  - Builds stage-specific prompt payloads from manifest + state.
- `src/lamet_agent/agent.py`
  - Main agent loop over stages.
  - `call_llm_api()` is the single place for model API integration.
  - `run_agent()` manages stage iteration, resume, and action collection.
- `src/lamet_agent/cli.py`
  - Exposes `validate`, `workflow`, `run` commands.
  - Parses CLI args, validates manifest, calls `run_agent()`.
- `src/lamet_agent/kernels.py`
  - Built-in kernel function examples for smoke tests.
- `examples/fake_data/generate_fake_data.py`
  - Generates fake correlator-style datasets used for local testing.
- `examples/workflow_smoke_manifest.json`
  - Minimal runnable manifest example.

## Current Status

- `validate` already enforces schema + kernel import checks.
- `run` executes the stage loop and collects structured actions.
- Real provider API wiring is intentionally centralized in `agent.py::call_llm_api`.
