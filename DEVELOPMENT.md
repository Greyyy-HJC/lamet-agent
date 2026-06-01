# Development Guide

This file is developer-facing and plan-oriented.

- User-facing overview: `README.md`
- Durable agent rules: `AGENTS.md`
- Physics/product intent: `PLAN.md`
- Engineering backlog and milestones: `TODO.md`
- Manifest contract and taxonomy: `docs/analysis_model.md`

## 1) Development Goals

`lamet-agent` aims to provide an agent-assisted LaMET/LQCD analysis workflow that is:

- reproducible (manifest-driven and stateful)
- extensible (stage interfaces and reusable analysis helpers)
- auditable (intermediate artifacts, diagnostics, and reports)
- uncertainty-aware (resampling propagation and systematic scans)

## 2) Architecture: Current vs Target

### Current implementation (implemented)

- CLI entry points: validate, workflow inspect, and run
- deterministic stage engine under `src/lamet_agent/stages/`
- shared analysis helpers under `src/lamet_agent/extensions/`
- run outputs with stage summaries and report artifacts

### Target expansion (planned)

- orchestrated agent runtime with structured actions
- skill-oriented prompting for stage-specific strategy/execution
- richer workflow state for pause/resume/re-run with diagnostics
- expanded schema objects for analysis products and metadata

## 3) Stage Responsibility Boundaries

The project should keep clear boundaries between user choices, agent assistance, and deterministic tools.

- `correlator_analysis`: strategy-assisted (fit models, windows, priors, diagnostics)
- `renormalization`: execution-oriented (scheme/user inputs must be explicit; agent validates and executes)
- `fourier_transform`: mixed (agent may help with asymptotic fit-window choices; transform conventions explicit)
- `perturbative_matching`: execution-oriented (kernel/scheme/scale specified by user)
- `physical_limit` / extrapolation: strategy-assisted (ansatz selection, range scans, prior sensitivity)

## 4) Milestone Roadmap (M1-M5)

### Milestone 1: Minimal agent runtime

Deliverables:

- `main.py` style orchestration entry
- controller loop (`orchestrator`)
- model client abstraction (`llm_client`)
- structured action schema and tool dispatch contract
- persisted state save/resume

Exit criteria:

- agent emits valid structured actions
- at least one deterministic tool call round-trip succeeds

### Milestone 2: Correlator analysis prototype

Deliverables:

- mock 2pt/3pt generators and fit utilities
- effective mass + plateau + summation + simultaneous-fit baseline
- fit-window scan and stability diagnostics
- bare matrix element output contract

Exit criteria:

- reproducible `2pt + 3pt -> bare matrix element` on synthetic data

### Milestone 3: Renormalization + Fourier prototype

Deliverables:

- constant and ratio renormalization baseline
- z-space convention handling
- asymptotic extrapolation and Fourier transform pipeline
- quasi distribution outputs with uncertainty propagation

Exit criteria:

- reproducible `bare -> renormalized h(z) -> quasi distribution` mock pipeline

### Milestone 4: Matching + extrapolation prototype

Deliverables:

- matching kernel interfaces + convolution path
- continuum/chiral/volume extrapolation baselines
- model averaging and systematic scan support

Exit criteria:

- simplified end-to-end LaMET workflow reaches physical-observable output

### Milestone 5: Robustness and documentation

Deliverables:

- expanded unit/integration/regression tests
- curated example configs and smoke datasets
- stronger user/developer docs and benchmark references

Exit criteria:

- collaborators can run, validate, and extend the workflow with minimal onboarding

## 5) Stage-by-Stage Development Details

### 5.1 Correlator analysis

Development focus:

- extract spectra/matrix elements with robust fit diagnostics
- compare plateau/summation/simultaneous/GEVP/model-averaging strategies
- preserve resampling samples through every derived observable

Key planned checks:

- fit-window sensitivity
- prior sensitivity (where Bayesian fits are used)
- excited-state contamination diagnostics

### 5.2 Renormalization

Development focus:

- support constant, ratio, hybrid, and self-renormalization paths
- enforce complete user input for scheme-dependent execution
- propagate errors consistently under correlated operations

Self-renormalization detail to preserve:

- fit `ln M(z,a)` decomposition terms
- match perturbative-region behavior to estimate nonperturbative slope terms
- construct `M_R(z)` and extract `Z_R(z,a)` for reuse

### 5.3 Fourier transform

Development focus:

- enforce symmetric/antisymmetric z-space conventions
- jointly handle real/imaginary matrix element components
- support asymptotic extrapolation ansatz variants by method family (GI/CG)
- scan `z_min`, `z_max`, and smoothing choices for systematic stability

### 5.4 Perturbative matching

Development focus:

- deterministic matching interfaces (kernel/scheme/scale/user-configured)
- stable convolution path and endpoint handling
- explicit error propagation through matching transforms

### 5.5 Extrapolation (`physical_limit`)

Development focus:

- combined continuum/chiral/finite-volume fits
- optional higher-order term inclusion where data supports it
- prior-aware and model-averaged uncertainty decomposition

## 6) Runtime, Prompt, and Schema Plan

### Runtime components (planned)

- controller/orchestrator loop
- model adapter layer
- tool registry and action dispatcher
- state persistence (`state.json`-style lifecycle) with resumability
- structured run logs (`logs.jsonl`-style traceability)

### Prompt system (planned)

- `prompts/system.md`: role, boundaries, output protocol
- `prompts/controller.md`: orchestration logic and action format
- `prompts/skills/*.md`: per-stage input checks and output schemas

### Data and state schema plan (planned)

Priority object families:

- resampling containers (`mean`, `samples`, covariance metadata)
- correlator and fit specs/results
- matrix element and renormalization specs
- momentum-space and extrapolation datasets
- workflow state and report payloads

## 7) Testing Strategy by Maturity

### Unit tests

- resampling, covariance, fitting, per-stage core math, schema validation

### Integration tests

- staged mock pipelines:
  - correlator -> matrix element
  - renormalization -> coordinate-space matrix element
  - Fourier -> quasi distribution
  - matching -> matched distribution
  - extrapolation -> physical point

### Agent-protocol tests

- valid structured actions
- required-input request behavior
- no undefined tool calls
- reproducible action traces under deterministic settings

### Regression tests

- tracked synthetic benchmarks for representative workflows
- tolerance-based drift checks on known outputs

## 8) Implementation Status Tracking

Use this status legend in task updates and PR descriptions:

- Implemented: available in mainline and covered by tests/smokes
- In progress: partially merged or validated only on local/synthetic runs
- Planned: design agreed but implementation not started

## 9) Open Technical Decisions

The following questions should be resolved early and recorded when decisions are made:

- first production priority (`g_A` workflow or quasi-PDF-first path)
- canonical data format baseline (for example HDF5-first policy)
- model provider support scope in v1 runtime
- prompt packaging approach (plain markdown vs package resources)
- report target format priority (Markdown/LaTeX/notebook)
- synthetic-first vs early real-data integration balance

## 10) Local Development Environment

Create a local environment with development and analysis dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,analysis]'
```

If Matplotlib cannot write to the default config directory:

```bash
export MPLCONFIGDIR=/tmp/.mpl
```

Useful commands:

```bash
.venv/bin/pytest -q
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/workflow_smoke_manifest.json
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/pion_2pt_manifest.json
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/proton_cg_qpdf_manifest.json
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/pion_cg_qtmdpdf_manifest.json
```

Pion CG Collins-Soper kernel workflow (using local bare-quasi cache):

```bash
.venv/bin/python scripts/prepare_cs_kernel_data.py
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/pion_cg_cs_kernel_manifest.json \
  --resume-from examples/outputs/pion_cg_cs_kernel/run_prepared \
  --start-stage renormalization
```

Notes:

- The txt files in `examples/data/pion_cg_cs_kernel/` are gitignored (unpublished data).
- Restore unpublished local datasets from trusted backups when moving to a new machine.
- If the console script is not installed, use `python scripts/run_manifest.py ...`.
