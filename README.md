# lamet-agent

`lamet-agent` is a CLI-first LaMET analysis workflow scaffold.

It exists to make LQCD/LaMET analysis runs reproducible and easier to extend by combining:

- structured manifests for inputs and metadata
- deterministic stage execution
- consistent outputs (reports, summaries, plots, and stage artifacts)

## What You Can Do

- validate a workflow manifest before running
- inspect the resolved stage workflow
- execute staged analysis pipelines from tracked examples
- resume a previous run from a chosen stage when outputs already exist

The default stage pipeline is:

1. `correlator_analysis`
2. `renormalization`
3. `fourier_transform`
4. `perturbative_matching`
5. `physical_limit`

For custom goals, use `goal: "custom"` with an explicit `workflow.stages` list.

## Project Status

The repository is transitioning from a stage-only CLI scaffold to the agent runtime described in `TODO.md`.

- Current scaffold interface: `validate`, `workflow`, `run` (legacy path)
- TODO target interface: `main.py` + orchestrator + config/state-driven execution

## Quickstart

Create a local environment first:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
```

Install dependencies according to your current task:

```bash
python -m pip install -e '.[dev,analysis]'
```

If you are only editing docs/plans, installation can be skipped.

## Runtime Entry Points

### Current scaffold commands (legacy interface)

Validate a manifest:

```bash
lamet-agent validate examples/workflow_smoke_manifest.json
```

Inspect the resolved workflow:

```bash
lamet-agent workflow examples/workflow_smoke_manifest.json
```

Run a workflow:

```bash
lamet-agent run examples/workflow_smoke_manifest.json
```

Resume a previous run from a stage:

```bash
lamet-agent run examples/pion_cg_qtmdpdf_manifest.json \
  --resume-from examples/outputs/pion_cg_qtmdpdf/run_YYYYMMDDTHHMMSSZ \
  --start-stage fourier_transform
```

`--resume-from` and `--start-stage` must be used together.

If Matplotlib cache writes fail in your environment:

```bash
MPLCONFIGDIR=/tmp/matplotlib lamet-agent run examples/workflow_smoke_manifest.json
```

If you are running directly from repository scripts:

```bash
python scripts/run_manifest.py run examples/workflow_smoke_manifest.json
```

These commands reflect the current scaffold and may evolve as the TODO runtime milestones are merged.

### TODO-aligned target runtime (planned)

Planned entry points in `TODO.md` include:

- `main.py` as the runtime CLI entry
- `orchestrator.py` for controller loop
- `llm_client.py` for model provider integration
- config-driven runs (for example `config.yaml`)
- resumable state-driven execution (for example `state.json`)

Refer to `DEVELOPMENT.md` for implementation phase details and milestone status.

## Example Manifests (Current Scaffold)

- `examples/workflow_smoke_manifest.json`: small tracked full-pipeline smoke workflow
- `examples/pion_2pt_manifest.json`: pion two-point workflow
- `examples/proton_cg_qpdf_manifest.json`: proton CG qPDF workflow
- `examples/pion_cg_qtmdpdf_manifest.json`: pion CG qTMDPDF workflow
- `examples/pion_cg_cs_kernel_manifest.json`: pion CG Collins-Soper kernel workflow

See `examples/data/` and `data/` for referenced inputs. Some large or unpublished datasets are intentionally gitignored.

## Documentation Map

- `DEVELOPMENT.md`: developer plan, milestones, and implementation details
- `AGENTS.md`: durable operating rules for coding agents
- `docs/analysis_model.md`: manifest contract and analysis taxonomy
- `PLAN.md`: high-level product and physics workflow intent
- `TODO.md`: engineering backlog and milestone breakdown

## License

License is not yet finalized in this repository (no `LICENSE` file committed yet).
