# lamet-agent

`lamet-agent` is a Python-first scaffold for a LaMET/LQCD analysis agent.

## Core Idea

Expected inputs:

- `correlators`: lattice correlation-function datasets
- `kernels`: perturbative kernels provided as Python functions

Kernel references use `module:function`, e.g. `lamet_agent.kernels:identity_kernel`.

Expected agent behavior:

- Automatically run the full LaMET analysis workflow from correlators and kernels.
- Emit intermediate stage outputs so users can track progress and understand the
  analysis path.
- Produce final physics distribution functions (for example DA, PDF, and TMDs),
  including plots in PDF format and final result arrays in `.npy` files.

Ordered five-stage workflow:

1. `correlator_analysis` -> `stages/correlator/`
2. `renormalization` -> `stages/renorm/`
3. `fourier_transform` -> `stages/fourier/`
4. `perturbative_matching` -> `stages/matching/`
5. `extrapolation` -> `stages/extrapolation/`

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
│   ├── core/
│   │   ├── prompting.py
│   │   ├── data.py
│   │   └── stages.py
│   ├── kernels.py
│   ├── manifest.py
│   └── stages/
│       ├── correlator/
│       │   ├── prompts.py
│       │   ├── skills.py
│       │   └── functions.py
│       ├── renorm/
│       ├── fourier/
│       ├── matching/
│       └── extrapolation/
└── tests/unit/
    ├── test_agent.py
    ├── test_stage_core.py
    ├── test_schemas.py
    └── test_validation.py
```

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
- `src/lamet_agent/core/stages.py`
  - Resolves stage sequence for a workflow goal.
  - Maps stage IDs to concrete stage packages.
- `src/lamet_agent/core/data.py`
  - Defines typed data containers (`EnsembleInfo`, `EnsembleData`) for resampled
    lattice data.
  - Provides common data operations (resampling, coordinate transforms, and
    cross-stage arithmetic/alignment helpers).
- `src/lamet_agent/core/prompting.py`
  - Stores `SYSTEM_PROMPT` and shared output-format hint.
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
- `src/lamet_agent/stages/*`
  - Each stage owns `prompts.py`, `skills.py`, and `functions.py`.
  - `prompts.py` contains the stage instruction text.
  - `skills.py` performs stage-local checks and strategy scaffolding.
  - `functions.py` holds stage-local execution placeholders.
- `examples/fake_data/generate_fake_data.py`
  - Generates fake correlator-style datasets used for local testing.
- `examples/workflow_smoke_manifest.json`
  - Minimal runnable manifest example.

## Agent Workflow

1. API or CLI receives a manifest path and runtime options (`model`, `resume_from`,
   `max_steps`).
2. `manifest.py` validates the input contract and resolves each kernel callable from
   `module:function`.
3. `agent.py` asks `core/stages.py` for the ordered stage workflow.
4. For each stage, `core/prompting.py` assembles one prompt from:
   - shared system prompt text,
   - stage instruction in `stages/<stage>/prompts.py`,
   - run context (run ID, completed stages, correlator IDs, kernel IDs).
5. `agent.py` sends the prompt to `call_llm_api()` and records the returned
   structured action.
6. Stage-local `skills.py`/`functions.py` are the extension points where stage
   checks and stage execution logic are implemented as the project matures.
7. The run ends with a structured summary including completed stages and collected
   actions.

## Current Status

- `validate` already enforces schema + kernel import checks.
- `run` executes the stage loop and collects structured actions.
- Real provider API wiring is intentionally centralized in `agent.py::call_llm_api`.
