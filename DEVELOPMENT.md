# Development Guide

This file is developer-facing. The user-facing overview lives in
[README.md](/home/jinchen/git/anl/lamet-agent/README.md).

## Development Environment

Create a local environment with development and analysis dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,analysis]'
```

If Matplotlib cannot write to the default config directory:

```bash
export MPLCONFIGDIR=/tmp/.mpl
```

## Useful Commands

Run the test suite:

```bash
.venv/bin/pytest -q
```

Run the tracked workflow smoke example:

```bash
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/workflow_smoke_manifest.json
```

Run the pion two-point example:

```bash
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/pion_2pt_manifest.json
```

Run the proton CG qPDF example:

```bash
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/proton_cg_qpdf_manifest.json
```

If you are working from the repository without the installed console script:

```bash
python scripts/run_manifest.py run examples/workflow_smoke_manifest.json
```

## Repository Structure

- `src/lamet_agent/cli.py`: CLI entry points
- `src/lamet_agent/schemas.py`: manifest models and validation
- `src/lamet_agent/workflows.py`: workflow execution
- `src/lamet_agent/stages/`: stage implementations and registry
- `src/lamet_agent/extensions/`: reusable analysis helpers
- `src/lamet_agent/loaders.py`: built-in correlator loaders
- `src/lamet_agent/plotting.py`: shared plotting helpers
- `examples/`: curated workflows and input data
- `docs/analysis_model.md`: manifest and example taxonomy
- `tests/`: focused contract, stage, and smoke tests

## Notes

- Keep reusable logic in the package, not in scripts.
- `correlator_analysis` now groups inputs by `(setup_id, momentum, smearing)`.
- Stage parameters should describe algorithms and fit controls, not lattice setup metadata.
- `physical_limit` currently writes per-family outputs and records available setup/momentum/`b` axes for future continuum, chiral, and boost extrapolations.
