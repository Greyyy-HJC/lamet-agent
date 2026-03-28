# lamet-agent

`lamet-agent` is a CLI-first LaMET workflow scaffold. It validates a JSON
manifest, loads correlator data, runs staged analysis steps, and writes plots,
tables, and run summaries to disk.

## What It Does

The default pipeline is:

1. `correlator_analysis`
2. `renormalization`
3. `fourier_transform`
4. `perturbative_matching`
5. `physical_limit`

The project also supports custom workflows where you choose the stage list
explicitly in the manifest.

## Quickstart

Create a local environment and install the package:

```bash
python -m venv .venv && . .venv/bin/activate && python -m pip install -U pip && python -m pip install -e .
```

If you want tests:

```bash
python -m pip install -e .[dev]
```

If you want the analysis helpers used by the two-point workflow, including
`gvar`, `lsqfit`, and HDF5 support:

```bash
python -m pip install -e .[analysis]
```

If you want both:

```bash
python -m pip install -e '.[dev,analysis]'
```

## Main Commands

Validate a manifest:

```bash
lamet-agent validate examples/demo_manifest.json
```

Inspect the resolved workflow:

```bash
lamet-agent workflow examples/demo_manifest.json
```

Run the default demo:

```bash
lamet-agent run examples/demo_manifest.json
```

Run the two-point correlator analysis demo:

```bash
lamet-agent run examples/two_point_analysis_manifest.json
```

Run the bare-qPDF correlator analysis example:

```bash
lamet-agent run examples/bare_qpdf_manifest.json
```

If you are running from the repository without installing the console script,
use:

```bash
python scripts/run_manifest.py run examples/demo_manifest.json
```

## Two-Point Example

The two-point example manifest is:

- [examples/two_point_analysis_manifest.json](/home/jinchen/git/anl/lamet-agent/examples/two_point_analysis_manifest.json)

The raw demo dataset is:

- [examples/data/two_point_raw_demo.csv](/home/jinchen/git/anl/lamet-agent/examples/data/two_point_raw_demo.csv)
- [examples/data/two_point_raw_demo.md](/home/jinchen/git/anl/lamet-agent/examples/data/two_point_raw_demo.md)

Run it with:

```bash
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/two_point_analysis_manifest.json
```

This manifest runs only the `correlator_analysis` stage on a raw two-point
input. The current implementation does:

- resampling from raw configurations
- effective-mass construction
- configurable `n`-state two-point fitting
- plot and table export

Each run creates a timestamped output directory under:

- [examples/outputs/two_point_demo](/home/jinchen/git/anl/lamet-agent/examples/outputs/two_point_demo)

## Bare qPDF Example

The bare-qPDF example manifest is:

- [examples/bare_qpdf_manifest.json](/home/jinchen/git/anl/lamet-agent/examples/bare_qpdf_manifest.json)

The packaged raw data live under:

- [examples/data/bare_qpdf](/home/jinchen/git/anl/lamet-agent/examples/data/bare_qpdf)

Run it with:

```bash
MPLCONFIGDIR=/tmp/.mpl .venv/bin/lamet-agent run examples/bare_qpdf_manifest.json
```

This custom workflow runs `correlator_analysis` on one raw two-point dataset
and a family of raw three-point datasets for the packaged `p=(4,4,0)` bare-qPDF
example, then writes:

- ratio and FH data plots
- data+fit comparison plots
- `lsqfit` text summaries
- a final bare-qPDF versus `z` plot

The bundled `bare_qpdf` manifest also demonstrates two features intended for
real analyses:

- correlator-family expansion through `correlators[].expand`, so one manifest
  entry can generate many `three_point` inputs across `b` and `z`
- independent three-point fit windows for `ratio` and `fh`, with optional
  `real` and `imag` part overrides

## Output Layout

Each workflow run writes a new directory of the form:

```text
<output-root>/run_YYYYMMDDTHHMMSSZ/
```

Inside that directory you will find:

- `report.md`
- `report.json`
- `summaries/<stage-name>.md`
- `stages/<stage-name>/...`

Default plot format is `pdf`. Default data format is `csv`.

Matplotlib-backed plots are saved with:

- `transparent=True`
- `tight_layout()`

Simple plots use the shared plotting presets in
`lamet_agent.extensions.plot_presets`.

## Manifest Overview

A workflow manifest is a JSON file with these top-level fields:

- `goal`: `parton_distribution_function`, `distribution_amplitude`, or `custom`
- `correlators`: input datasets with `kind`, `path`, `file_format`, `label`,
  and optional `metadata`
- `metadata`: run-level metadata; currently at least `ensemble` and
  `conventions` are required
- `kernel`: inline Python source plus `callable_name`
- `workflow`: optional stage list and per-stage parameters
- `outputs`: output directory and requested plot/data formats

Supported correlator `file_format` values are currently `csv`, `npz`, and
`txt`.

Correlator entries may also define an optional `expand` object. This expands
one manifest entry into a family of correlators by formatting `path`, `label`,
and `metadata` fields with values such as `z`, `b`, or momentum components.

For `goal = "custom"`, you must provide `workflow.stages`.

For three-point correlator analysis, the stage parameters support:

- shared `ratio` and `fh` fit-window settings
- part-specific `real` / `imag` overrides for `fit_tsep` and `tau_cut`
- explicit `sample_fit_workers` for sample-wise parallel fitting

## Kernel Contract

The inline hard-kernel callable must accept:

```python
def my_kernel(axis, values, metadata):
    ...
```

It should return an array with the same shape as `values`.

## Repository Layout

- `src/lamet_agent/`: package code
- `examples/`: example manifests and demo data
- `tests/`: unit and smoke tests
- `incoming/analysis_steps/`: local draft analysis code during development;
  this directory is intentionally ignored by git and not part of the shared
  repository history

For repository internals, local development workflow, testing, and code
organization notes, see
[DEVELOPMENT.md](/home/jinchen/git/anl/lamet-agent/DEVELOPMENT.md).

## Migrated Analysis Helpers

Reusable helpers currently exposed from `lamet_agent.extensions` include:

- `lamet_agent.extensions.statistics`
- `lamet_agent.extensions.plot_presets`
- `lamet_agent.extensions.two_point`
- `lamet_agent.extensions.three_point`

These modules are the stable place for reusable analysis logic. Keep raw drafts
under `incoming/analysis_steps/` until they are cleaned up and integrated.
