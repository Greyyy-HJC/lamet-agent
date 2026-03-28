# lamet-agent

`lamet-agent` is a CLI-first LaMET workflow scaffold. It validates a JSON
manifest, loads correlator data, runs staged analysis steps, and writes plots,
tables, and summaries to disk.

## Project Model

The repository is organized in three layers:

1. reusable helper functions under `src/lamet_agent/`
2. analysis stages under `src/lamet_agent/stages/`
3. complete example workflows under `examples/`

The stage pipeline remains:

1. `correlator_analysis`
2. `renormalization`
3. `fourier_transform`
4. `perturbative_matching`
5. `physical_limit`

More detail on the manifest contract and example taxonomy lives in
[docs/analysis_model.md](/home/jinchen/git/anl/lamet-agent/docs/analysis_model.md).

## Quickstart

Create a local environment and install the package:

```bash
python -m venv .venv && . .venv/bin/activate && python -m pip install -U pip && python -m pip install -e .
```

If you want tests:

```bash
python -m pip install -e .[dev]
```

If you want the analysis helpers used by the physics-oriented stages:

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
lamet-agent validate examples/workflow_smoke_manifest.json
```

Inspect the resolved workflow:

```bash
lamet-agent workflow examples/workflow_smoke_manifest.json
```

Run the tracked full-pipeline smoke example:

```bash
lamet-agent run examples/workflow_smoke_manifest.json
```

Run the pion two-point example:

```bash
lamet-agent run examples/pion_2pt_manifest.json
```

Run the proton CG qPDF example:

```bash
lamet-agent run examples/proton_cg_qpdf_manifest.json
```

If you are running from the repository without installing the console script:

```bash
python scripts/run_manifest.py run examples/workflow_smoke_manifest.json
```

## Example Set

- [examples/workflow_smoke_manifest.json](/home/jinchen/git/anl/lamet-agent/examples/workflow_smoke_manifest.json)
  - small tracked full-pipeline smoke workflow
  - backed by [examples/data/workflow_smoke](/home/jinchen/git/anl/lamet-agent/examples/data/workflow_smoke)
- [examples/pion_2pt_manifest.json](/home/jinchen/git/anl/lamet-agent/examples/pion_2pt_manifest.json)
  - pion two-point-only workflow
  - backed by [examples/data/pion_2pt](/home/jinchen/git/anl/lamet-agent/examples/data/pion_2pt)
- [examples/proton_cg_qpdf_manifest.json](/home/jinchen/git/anl/lamet-agent/examples/proton_cg_qpdf_manifest.json)
  - canonical proton CG qPDF workflow
  - backed by [examples/data/proton_cg_qpdf](/home/jinchen/git/anl/lamet-agent/examples/data/proton_cg_qpdf)
- [examples/data/pion_cg_qtmdpdf](/home/jinchen/git/anl/lamet-agent/examples/data/pion_cg_qtmdpdf)
  - tracked representative pion CG qTMDPDF data slice for future examples

## Manifest Overview

A workflow manifest is a JSON file with these top-level fields:

- `goal`: `parton_distribution_function`, `distribution_amplitude`, or `custom`
- `correlators`: list of input correlator specs
- `metadata`: structured workflow metadata
- `kernel`: inline hard-kernel definition
- `workflow`: optional stage overrides and stage parameters
- `outputs`: output preferences

Required `metadata` fields:

- `purpose`: `smoke` or `physics`
- `analysis.gauge`: `cg` or `gi`
- `analysis.hadron`: `pion` or `proton`
- `analysis.channel`: `qpdf` or `qda`
- `conventions`
- `setups`

Each `setups.<setup_id>` entry must define:

- `lattice_action`
- `n_f`
- `lattice_spacing_fm`
- `spatial_extent`
- `temporal_extent`
- `pion_mass_valence_gev`
- `pion_mass_sea_gev`

Each correlator metadata block must define:

- `setup_id`
- `momentum`
- `smearing`
- for `three_point`: `displacement.b`, `displacement.z`, `operator.gamma`, `operator.flavor`

The effective observable is derived from the channel and `b`:

- `qpdf` with `b=0` -> `qpdf`
- `qpdf` with `b!=0` -> `qtmdpdf`
- `qda` with `b=0` -> `qda`
- `qda` with `b!=0` -> `qtmdwf`

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
