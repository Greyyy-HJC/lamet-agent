# Development Guide

This file is for repository developers. The main
[README.md](/home/jinchen/git/anl/lamet-agent/README.md) is kept user-facing and
focused on installation and workflow execution.

## Development Environment

Create a local environment and install the package with development and analysis
dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,analysis]'
```

If Matplotlib cannot write to the default config directory in your environment,
set:

```bash
export MPLCONFIGDIR=/tmp/.mpl
```

## Useful Commands

Run the full test suite:

```bash
.venv/bin/pytest -q
```

Run only the two-point workflow tests:

```bash
MPLCONFIGDIR=/tmp/.mpl .venv/bin/pytest -q tests/test_two_point_analysis.py
```

Run the demo manifests:

```bash
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/demo_manifest.json
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/two_point_analysis_manifest.json
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/bare_qpdf_manifest.json
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/qpdf_ft_manifest.json
```

`examples/demo_manifest.json` is now the lightweight tracked qPDF-FT toy demo,
not the older scalar FFT example.

If you are working from the repository without the installed console script:

```bash
python scripts/run_manifest.py run examples/demo_manifest.json
```

## Repository Structure

- `src/lamet_agent/cli.py`: CLI entry points
- `src/lamet_agent/constants.py`: shared lattice/QCD constants and running-coupling helpers
- `src/lamet_agent/schemas.py`: manifest models and validation
- `src/lamet_agent/workflows.py`: workflow execution
- `src/lamet_agent/stages/`: stage implementations and registry
- `src/lamet_agent/extensions/`: reusable analysis helpers
- `src/lamet_agent/loaders.py`: built-in correlator loaders
- `src/lamet_agent/plotting.py`: shared export helpers
- `examples/`: demo manifests and sample data
- `tests/`: unit and end-to-end smoke tests
- `incoming/analysis_steps/`: local-only intake area for legacy or draft code;
  ignored by git and not meant for synchronization

## Workflow Notes

The default rule-based pipeline is:

1. `correlator_analysis`
2. `renormalization`
3. `fourier_transform`
4. `perturbative_matching`
5. `physical_limit`

Custom workflows can override the stage list in the manifest.

For three-point analysis, manifests can now:

- expand one correlator entry into many datasets via `correlators[].expand`
- configure independent `ratio` and `fh` fit windows
- override `fit_tsep` and `tau_cut` separately for `real` and `imag`
- control sample-wise multiprocessing with `three_point.sample_fit_workers`

For qPDF Fourier transforms, the workflow now keeps final sample-wise
`z`-dependent bare matrix elements through:

1. `correlator_analysis`
2. `renormalization` as an identity pass-through for qPDF families
3. `fourier_transform` for sample-wise extrapolation and FT

The retained sample arrays are stored both in the runtime stage payload and as
compact `.npz` artifacts under the stage directory.

`fourier_transform.sample_transform_workers` controls multiprocessing for the
sample-wise extrapolation step. The FT itself is then applied in one batched
matrix pass over all samples.

## Plotting Rules

Project plotting helpers should follow these conventions:

- simple plots can use `default_plot()` from `lamet_agent.extensions.plot_presets`
- save plots as `.pdf` by default
- call `tight_layout()` before saving
- save Matplotlib figures with transparent backgrounds

These rules are centralized in
`src/lamet_agent/plotting.py` where possible.

## Manifest And Output Defaults

- Default plot format: `pdf`
- Default numeric export format: `csv`
- Each run writes to `<output-root>/run_YYYYMMDDTHHMMSSZ/`

Typical run output contains:

- `report.md`
- `report.json`
- `summaries/<stage-name>.md`
- `stages/<stage-name>/...`

## Code Migration

Draft or legacy analysis code may live in local `incoming/analysis_steps/`
while developing. After review and cleanup, reusable logic should move into
`src/lamet_agent/`.

Current stable extension modules include:

- `lamet_agent.extensions.statistics`
- `lamet_agent.extensions.plot_presets`
- `lamet_agent.extensions.two_point`
- `lamet_agent.extensions.three_point`

Keep scripts thin. Reusable logic belongs in the package.

## Development-Only Internal Notes

The repository still contains authentication and backend scaffolding under:

- `src/lamet_agent/auth/`
- `src/lamet_agent/backends/`

These are internal implementation areas and are intentionally omitted from the
main user README.
