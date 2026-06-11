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

The CG qPDF example manifests can run a connected correlator -> ratio-renormalization -> Fourier smoke flow. The renormalization stage reads correlator bare-matrix txt grids, applies the Eq. 15 ratio/hybrid scheme while preserving every resampled sample, writes a compatible `.npz`, and hands `matrix_element_data` directly to Fourier when stages run in one agent process.

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
│   │   ├── llm.py
│   │   ├── prompting.py
│   │   ├── tools.py
│   │   ├── trace.py
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

Stage plots and other artifacts are written under `artifacts/` in the **current
working directory**. For real runs, create a directory under `runs/` and execute
from there so outputs stay isolated (see `runs/ds_pdf_cont/run.sh`):

```bash
mkdir -p runs/my_run && cd runs/my_run
lamet-agent run ../../examples/workflow_cg_qpdf_cont_manifest.json \
  --stages renormalization,fourier_transform
# artifacts -> runs/my_run/artifacts/
```

Paths inside the manifest (correlator datasets, precomputed NPZ files, etc.)
are resolved relative to the manifest file, with a fallback to the repo root for
paths like `examples/fake_data/...`.

Run a subset of stages with `--stages` (comma-separated stage IDs, in the order
you want them executed). Omit `--stages` to run the full default pipeline for
the manifest `goal` (see `lamet-agent workflow`).

Valid stage IDs:

| Stage ID | Package |
| --- | --- |
| `correlator_analysis` | correlator |
| `renormalization` | renorm |
| `fourier_transform` | fourier |
| `perturbative_matching` | matching |
| `extrapolation` | extrapolation |

Examples:

```bash
# Correlator stage only
lamet-agent run examples/workflow_cg_qpdf_p0_manifest.json --stages correlator_analysis

# Renormalization then Fourier (continuation from pre-computed bare matrix elements)
lamet-agent run examples/workflow_cg_qpdf_cont_manifest.json \
  --stages renormalization,fourier_transform

# Two stages in one run
lamet-agent run examples/workflow_cg_qpdf_p5_manifest.json \
  --stages correlator_analysis,renormalization
```

To start from a middle stage without listing every later stage explicitly, use
`--resume-from` instead. It slices the default goal sequence from that stage
onward (for example `--resume-from fourier_transform` runs Fourier, matching,
and extrapolation). `--stages` takes precedence when both are set.

When you skip earlier stages, the manifest must already supply that stage's
inputs (for example bare-matrix report paths under `metadata.renormalization`,
or a renormalized NPZ path under `metadata.fourier_input`). Missing inputs
surface as `input_issues` and the run stops with
`status: waiting_for_user_input`.

Print each agent cycle (prompt, model action, tool observation) while the run
executes:

```bash
lamet-agent run examples/workflow_smoke_manifest.json --model deepseek --verbose
```

Choose the LLM provider with `--model` (`deepseek` or `openai`). The API key is
read from `--api-key-file` (default `api.key`) or the provider environment
variable (`DEEPSEEK_API_KEY` / `OPENAI_API_KEY`). Each provider defaults to a
cost-effective model (`deepseek-chat` / `gpt-4o-mini`); override with
`--llm-model` and, if needed, `--base-url`:

```bash
lamet-agent run examples/workflow_smoke_manifest.json --model openai --verbose
lamet-agent run examples/workflow_smoke_manifest.json --model openai --llm-model gpt-4o
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
- `src/lamet_agent/core/llm.py`
  - Pluggable `LlmSession` backends: `mock`, `external` (JSONL transcript), and the
    OpenAI-compatible providers `deepseek` and `openai` (multi-turn chat per stage).
  - `PROVIDERS` holds each provider's base URL, default model, and API-key env var;
    `make_llm_session()` selects a backend and shared HTTP lives in
    `_post_chat_completion` (add new OpenAI-compatible providers to `PROVIDERS`).
- `src/lamet_agent/core/tools.py`
  - Resolves a stage's `STAGE_TOOLS` registry for the agent loop.
  - `prepare_tool_args()` / `filter_tool_kwargs()` normalize LLM tool calls
    (manifest paths, plot `save_path` under `artifacts/`).
  - `resolve_plot_save_path()` forces correlator plot PDFs under `artifacts/` in
    the current working directory.
- `src/lamet_agent/core/trace.py`
  - Optional ReAct-style stdout trace (`--verbose`).
- `src/lamet_agent/core/plotting.py`
  - Self-contained publication-style plotting (default plot, 2pt fit-on-data).
- `src/lamet_agent/agent.py`
  - Stage orchestration only: `run_agent()` resolves stages, validates inputs,
    and runs the per-stage tool loop (`_run_stage`).
  - Correlator plots are written to `artifacts/` under the process working
    directory (created automatically).
- `src/lamet_agent/cli.py`
  - Exposes `validate`, `workflow`, `run` commands.
  - `run` accepts `--stages` (comma-separated subset), `--resume-from`,
    `--model` (`mock`/`external`/`deepseek`/`openai`), `--verbose` / `-v`
    (ReAct-style trace to stdout), `--actions-path` (for `external`), and
    `--api-key-file`/`--llm-model`/`--base-url` (for `deepseek`/`openai`).
- `src/lamet_agent/kernels.py`
  - Built-in kernel function examples for smoke tests.
- `src/lamet_agent/stages/*`
  - Each stage owns `prompts.py`, `skills.py`, and `functions.py`.
  - `prompts.py` contains the stage instruction text and action protocol.
  - `skills.py` performs stage-local checks plus `STAGE_SKILL` strategy text and
    a `tool_catalog()`.
  - `functions.py` holds the stage tools and a `STAGE_TOOLS` registry.
  - `stages/correlator/` is the first worked example and exposes four agentic
    tools (requires the `analysis` optional dependencies):
    `inspect_correlator_scale` (choose a `correlator_rescale`), `tune_ground_state`
    (2pt-only window scan + model average), `tune_bare_matrix` (scan bare-matrix fit
    windows on sample-average data for one representative z), and
    `fit_bare_matrix_grid` (apply one shared tuned window to every z and every
    resampled sample, then export `bare_qpdf/*.txt`, fit-on-data PDFs, split logs,
    and a JSON report). The agent tunes once on sample-average data, then applies the
    same setting everywhere; pass a single `pt2_window`/`pt3_window` or
    `model_average=true` to BMA-combine the window grid.
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
   - `core/llm.make_llm_session()` provides a pluggable `LlmSession` that drives a
     multi-turn loop (up to `max_tool_steps`, default 40): the model emits one
     JSON action per cycle; on `call_tool`, `core/tools.prepare_tool_args()` and
     `resolve_stage_tools()` run the tool and return an observation as the next
     user turn; on `finish` (or other non-tool actions) the stage ends.
5. Session backends: `mock` (deterministic scaffold), `external` (JSONL
   transcript replay via `--actions-path`), or `deepseek` (chat-completions API
   in `core/llm.py`).
6. The run ends with a compact JSON summary on stdout (`run_id`, `status`,
   `summary`, manifest paths, etc.). Full action traces are not printed; use
   `--verbose` for per-cycle ReAct-style logging. Programmatic callers using
   `run_agent()` still receive `actions` and `stage_results` in the return dict.

## Current Status

- `validate` already enforces schema + kernel import checks.
- `run` executes the stage loop and collects structured actions.
- Real provider API wiring lives in `core/llm.py` (DeepSeek today).
