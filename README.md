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

Print each agent cycle (prompt, model action, tool observation) while the run
executes:

```bash
lamet-agent run examples/workflow_smoke_manifest.json --model deepseek --verbose
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
  - Builds static stage context once per stage; incremental tool observations are
    appended as separate user turns in the DeepSeek multi-turn session.
- `src/lamet_agent/core/tools.py`
  - Resolves a stage's `STAGE_TOOLS` registry for the agent loop.
  - `resolve_plot_save_path()` forces correlator plot PDFs under `artifacts/` in
    the current working directory.
- `src/lamet_agent/core/plotting.py`
  - Self-contained publication-style plotting (default plot, 2pt fit-on-data).
- `src/lamet_agent/agent.py`
  - Main agent loop over stages with an intra-stage tool-execution loop.
  - Pluggable session: `mock` (deterministic scaffold), `external` (replays a
    JSONL action transcript), or `deepseek` (multi-turn chat per stage via the
    DeepSeek chat-completions API).
  - Correlator plots are written to `artifacts/` under the process working
    directory (created automatically).
  - `_request_llm_action()` is the single backend entry for mock and DeepSeek
    responders; add new providers there or in `_post_chat_completion`.
  - `run_agent()` resolves which stages to run (explicit `stages` subset or the
    default sequence), validates per-stage inputs, and collects tool results.
- `src/lamet_agent/cli.py`
  - Exposes `validate`, `workflow`, `run` commands.
  - `run` accepts `--stages` (comma-separated subset), `--resume-from`,
    `--model` (`mock`/`external`/`deepseek`), `--verbose` / `-v` (ReAct-style
    trace to stdout), `--actions-path` (for `external`), and
    `--api-key-file`/`--deepseek-model` (for `deepseek`).
- `src/lamet_agent/kernels.py`
  - Built-in kernel function examples for smoke tests.
- `src/lamet_agent/stages/*`
  - Each stage owns `prompts.py`, `skills.py`, and `functions.py`.
  - `prompts.py` contains the stage instruction text and action protocol.
  - `skills.py` performs stage-local checks plus `STAGE_SKILL` strategy text and
    a `tool_catalog()`.
  - `functions.py` holds the stage tools and a `STAGE_TOOLS` registry.
  - `stages/correlator/` is the first worked example: read 2pt data, resample,
    fit ground-state windows, logGBF model-average `E0`/`z0`, and plot the
    fit on data (requires the `analysis` optional dependencies).
- `examples/fake_data/generate_fake_data.py`
  - Generates fake correlator-style datasets used for local testing.
- `examples/workflow_smoke_manifest.json`
  - Minimal runnable manifest example.

## Agent Workflow

1. CLI receives a manifest path and runtime options (`--model`, `--stages`,
   `--resume-from`, `--verbose`).
2. `manifest.py` validates the input contract and resolves each kernel callable from
   `module:function`.
3. `agent.py` asks `core/stages.py` for the ordered stage workflow (explicit
   `--stages` subset or the default sequence for the manifest goal).
4. For each stage:
   - `core/tools.validate_stage_inputs()` surfaces missing inputs as
     `input_issues`.
   - `core/prompting.build_stage_static_prompt()` assembles static context once
     (system prompt, stage instruction, run context, tool catalog).
   - A pluggable `LlmSession` drives a multi-turn loop (up to `max_tool_steps`,
     default 30): the model emits one JSON action per cycle; on `call_tool`,
     `core/tools.resolve_stage_tools()` runs the tool and returns an observation
     as the next user turn; on `finish` (or other non-tool actions) the stage
     ends.
5. Session backends: `mock` (deterministic scaffold), `external` (JSONL
   transcript replay via `--actions-path`), or `deepseek` (chat-completions API
   via `_request_llm_action`).
6. The run ends with a compact JSON summary on stdout (`run_id`, `status`,
   `summary`, manifest paths, etc.). Full action traces are not printed; use
   `--verbose` for per-cycle ReAct-style logging. Programmatic callers using
   `run_agent()` still receive `actions` and `stage_results` in the return dict.

## Current Status

- `validate` already enforces schema + kernel import checks.
- `run` executes the stage loop and collects structured actions.
- Real provider API wiring lives in `agent.py::_request_llm_action` and
  `_post_chat_completion` (DeepSeek today).
