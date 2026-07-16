# lamet-agent

`lamet-agent` is a Python-first scaffold for a LaMET/LQCD analysis agent.

## Core Idea

The manifest defines global source pools and per-stage job lists. Job ids form a
DAG: correlator jobs group raw datasets, and later jobs consume upstream job ids
through role-named inputs such as `target` and `denominator`.

Expected agent behavior:

- Automatically run the full LaMET analysis workflow from correlators and kernels.
- Emit intermediate stage outputs as NetCDF (`.nc`) files so users can track
  progress and understand the analysis path.
- Produce final physics distribution functions (for example DA, PDF, and TMDs),
  including plots in PDF format and final result arrays in `.npy` files.

Ordered five-stage workflow:

1. `correlator_analysis` -> `stages/correlator/`
2. `renormalization` -> `stages/renorm/`
3. `fourier_transform` -> `stages/fourier/`
4. `perturbative_matching` -> `stages/matching/`
5. `extrapolation` -> `stages/extrapolation/`

The current job-DAG migration covers correlator analysis and hybrid-ratio
renormalization. Fourier, matching, and extrapolation remain the next migration step.

## Minimal Structure

```text
.
├── examples/
│   ├── fake_data/
│   │   └── generate_fake_data.py
│   ├── sample_manifest.jsonc
│   └── cg_pion_pdf_manifest.json
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

## Intermediate Data (NetCDF)

Stage-to-stage artifacts are **`EnsembleData` NetCDF files** written under the
manifest's `artifacts_directory` as `<stage>/<job_id>.nc`. Each file stores one resampled
array plus its lattice metadata:

- **Leading dimension** `resample`: bootstrap, jackknife, or raw sample index (length 1
  for `resample='gvar'`).
- **Physical dimensions** and coordinates: for example `z` for coordinate-space matrix
  elements, or `x` after Fourier transform.
- **Attributes**: reserved `ensemble` / `resample` metadata for `EnsembleInfo` and
  resampling mode, plus any stage-specific attrs on the underlying xarray object.

Typical artifact chain (paths are relative to `artifacts/` unless noted):

| Stage | Example artifact |
| --- | --- |
| `correlator_analysis` | `correlator_analysis/ca_p5.nc` |
| `renormalization` | `renormalization/rn_p5.nc` |
| `fourier_transform` | `fourier_results/fourier_result.nc`, `fourier_results/fourier_fit_info.nc` |
| `perturbative_matching` | `matching_results/quasi_pdf.nc` |

Within one run, downstream inputs resolve job ids to in-memory primary outputs.
`inputs.artifacts` provides equivalent source nodes for partial workflows.

### Write and read (Python)

Install the analysis extras (includes `xarray` and `netCDF4`):

```bash
pip install -e ".[dev,analysis]"
```

Use the typed helpers in `core/data.py`:

```python
from lamet_agent.core.data import EnsembleData

data.to_netcdf("artifacts/fourier_results/fourier_result.nc")
reload = EnsembleData.from_netcdf("artifacts/fourier_results/fourier_result.nc")
```

Complex arrays round-trip natively (`auto_complex=True`); you do not need to split real
and imaginary parts before saving.

### Inspect or read without lamet-agent

NetCDF is self-describing. Inspect with `ncdump -h file.nc`, Panoply, or xarray:

```python
import json
import xarray as xr
from lamet_agent.core.data import EnsembleInfo

da = xr.load_dataarray("fourier_result.nc", auto_complex=True)
ensemble = EnsembleInfo(**json.loads(da.attrs["ensemble"]))
resample = da.attrs["resample"]
values = da.values  # shape (n_sample, *physical_dims)
physical_dims = [d for d in da.dims if d != "resample"]
coords = {d: da.coords[d].values for d in physical_dims}
```

The first dimension is always named `resample`; remaining dims and coordinate variables
match the physical layout documented in each stage report.

## Manifest Example

`examples/sample_manifest.jsonc` is the annotated reference manifest. It is written
as **JSONC** (JSON with `//` comments) so that every field can document its allowed
options inline (for example `target_observable` is `"pdf"` or `"da"`, and `gfix` is
`"CG"` or `"GI"`). It is organized into three top-level blocks:

- `metadata`: run-level settings (`run_id`, `root_directory`, `artifacts_directory`,
  `target_observable`, `resample_mode`, `sample_error_mode`, `random_seed`,
  optional `workers`, ordered `stages` to run).
  `random_seed` is required and seeds every jackknife/bootstrap resampling step
  in the run (a job/stage no longer sets its own `seed`). When `resample_mode`
  is `"bs"`, `bs_samples` is required and must be set explicitly (there is no
  default bootstrap sample count). `sample_error_mode` controls how samples are
  averaged and how sample-by-sample fits receive errors; it defaults to
  `"covariance"`. `bin_size` is optional and bins configurations before
  resampling when set (default: no binning). `workers` is an optional positive
  integer controlling sample-fit processes in the correlator and Fourier
  stages; it defaults to `1`, which keeps execution serial.
- `inputs`: the `correlators` (each with its operator labels, `volume`,
  `lattice_spacing_fm`, momentum list, and for `3pt` the `bz_direction`, `tsep`, `bT`, and `bz`
  lists), external `artifacts`, and `kernels`.
- `stages`: `defaults` plus a `jobs` list. A job's `params` shallow-merge over
  defaults, and later jobs reference earlier job ids through role-named `inputs`.

Use it as a template and save runnable manifests as plain `.json`. The loader also
accepts JSONC for annotated authoring templates.

Stage `defaults` and job `params` use closed, stage-specific parameter contracts.
`lamet-agent validate` rejects unknown top-level and nested keys instead of
silently dropping them when tool arguments are prepared. Typographical errors
include the closest supported key when one is available. Runner-owned settings
such as `workers`, `random_seed`, and `sample_error_mode` belong under
`metadata`; derived quantities such as `momentum_gev` must not be written as
stage parameters. Full workflows derive them from their upstream correlators,
while partial workflows declare `momentum`, `volume`, and
`lattice_spacing_fm` on `inputs.artifacts[]`.

## Manifest Parameter Semantics

Some manifest parameters change both the statistical treatment and the runtime
substantially. This section records behavior that is not obvious from the field
name alone.

### `correlator_analysis.defaults.model_average`

This boolean controls how `fit_bare_matrix_grid` uses fit-function candidates.
It does not control whether tuning scans the candidates: `tune_bare_matrix` always
tests the configured `pt2_windows`, `pt3_tau_cuts`, `nstate`, `prior_width`, and
`fit_strategy` candidates on sample-average data at LLM-supplied `tune_z_values`
first. The tool returns cross-z feasibility summaries and
`recommended_robust_index`; the agent must pass explicit `tune_z_values` when
calling `tune_bare_matrix`.

- `false` (recommended production default): use one tuned data window and one
  sample-average-selected fit-function setting for every `z` and every resampled
  sample. The agent should provide the selected `pt2_window` and `pt3_window` from
  a candidate with `feasible_at_all_tune_z=true`; if it does not, the grid tool
  selects the best usable window on a single representative `tune_z`.
- `true`: still use one tuned data window, but scan `nstate` and `prior_width`
  fit-function candidates for each resampled sample and combine successful fits
  with `logGBF` weights. The default prior-width scan is `[0.5, 1.0, 2.0]`.

The correlator NetCDF artifact stores the weighted resampled bare matrix-element
samples as usual and records per-`z` uncertainty summaries in attrs:
`bare_re_stat_sdev` / `bare_im_stat_sdev` from the resampling spread and
`bare_re_sys_sdev` / `bare_im_sys_sdev` from the fit-function model spread. The
systematic arrays are zero for the single-model `model_average: false` path.

### `renormalization.defaults.normalization`

When `true` (default), the runner divides every bare `EnsembleData` input in the
job store by its lattice `z=0` value before any renormalization tool runs. Scheme
tools such as `apply_ratio_scheme_renormalization` then apply only the declared
ratio/hybrid prescription. Set `false` to skip this preprocessing and pass raw
bare matrix elements directly into the scheme.

For example, two `nstate` values and three `prior_width` values produce up to six
fit-function models inside the fixed data window. The manifest value is
authoritative and cannot be overridden by an LLM tool call.

### Per-job hybrid-ratio `zs_fm`

The hybrid switch distance belongs to the data-processing job, not to a global
kernel declaration. Set it as `stages.renormalization.defaults.zs_fm` or
`stages.renormalization.jobs[].params.zs_fm`, and independently as
`stages.perturbative_matching.defaults.zs_fm` or
`stages.perturbative_matching.jobs[].params.zs_fm`. Job values override stage
defaults, so different data chains may use different switch distances.

Do not place `zs_fm` under `inputs.kernels[].kernel_parameters` or under
renormalization `scheme_parameters`; manifest validation rejects both legacy
locations. For a complete in-manifest chain, the review stage follows
`matching.quasi -> fourier.input -> renormalization job` and reports whether the
hybrid matching and hybrid-ratio renormalization values agree. Partial runs that
start from an external artifact are reported as not verifiable rather than as a
match or mismatch.

### `metadata.random_seed`, `metadata.bs_samples`, `metadata.sample_error_mode`, `metadata.bin_size`, `metadata.workers`

These fields are the single source of resampling and sample-parallelism
configuration for the whole run; stage/job params cannot override them.

- `random_seed` (required): seeds every jackknife/bootstrap resampling call in
  `core/resampling.py`. `prepare_tool_args` injects it as the `seed` argument
  for every correlator tool call.
- `bs_samples` (required when `resample_mode` is `"bs"`; ignored for
  `"jk"`, where resampling has no sample-count parameter): sets the bootstrap
  sample count (the tool-level `n_boot` argument). There is no default; the
  manifest must set this value explicitly for bootstrap runs.
- `sample_error_mode` (optional, default: `"covariance"`): controls how
  bootstrap/jackknife samples are converted to `gvar` averages and how the same
  ensemble errors are attached to individual sample-by-sample fits. `"mean"`
  uses mean centers with diagonal standard deviations, `"median"` uses
  bootstrap medians with half the 16-84 percentile width and is invalid with
  jackknife, and `"covariance"` uses mean centers with the full covariance
  matrix.
- `bin_size` (optional, default: no binning): when set, configurations are
  averaged into bins of this size before jackknife/bootstrap resampling.
- `workers` (optional, default: `1`): maximum number of worker processes used
  for independent sample fits in `correlator_analysis` and `fourier_transform`.
  Sample-average tuning, stage/job execution, correlator `z` scans, Fourier
  extrapolation, and Fourier summation remain serial. Active sample batches are
  capped by the number of samples.

Each worker process may otherwise inherit native BLAS threading. For multi-core
runs, avoid oversubscription by setting the relevant library thread counts when
launching the CLI, for example:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  lamet-agent run manifest.json
```

## Self-Renormalization

Self-renormalization (`scheme: "self_renormalization"`) fits a coordinate-space
factor $z_R(z,a)$ from a zero-momentum **reference**, then applies

$$
H_{\mathrm{ren}}(z) = \frac{H_{\mathrm{bare}}(z)}{z_R(z,a)\,Z_{\overline{\mathrm{MS}}}(z)}
$$

to each target sample. The stage always splits into **one fit job** plus one or
more **apply jobs**. See `examples/temp_self_renorm_manifest.json` and
`runs/ds_self_renorm/` for a runnable PDF→DA smoke test.

### Workflow

```text
inputs.artifacts (bare reference + bare targets)
        │
        ▼
┌───────────────────────┐
│ fit job {reference}   │  fit_self_renormalization_factor
│ params.d required     │  → store['zR'] / <job_id>.nc
│ params.m0_gev optional│  → fit diagnostics (ln|M|, mR, f1, …)
└───────────┬───────────┘
            │ zR job id
            ▼
┌───────────────────────┐
│ apply job {target,zR} │  optional params.d / m0_gev remap zR
│ per lattice / momentum│  → H/(zR ZMSbar) NetCDF + ME plot
└───────────────────────┘  → zmsbar_compare; last apply may emit
                             stage-level discrete_effect_re/im
```

Typical agent tool order:

1. **Fit job** (`inputs` exactly `{ "reference": "<bare_ref_id>" }`):
   `fit_self_renormalization_factor` → `plot_self_renormalization_diagnostics` → finish.
2. **Apply job** (`inputs` exactly `{ "target": "<bare_id>", "zR": "<fit_job_id>" }`):
   `apply_self_renormalization` → `plot_self_renormalization_diagnostics` →
   `plot_renormalized_matrix_element` → finish.

Same-operator use (zero-momentum PDF → finite-$P_z$ PDF): fit with the PDF `d`,
omit `m0_gev` so $m_0$ is fitted, and leave apply jobs without `d`/`m0_gev`
overrides. Cross-operator use (PDF reference → DA targets): fit with PDF `d`
(and usually omit `m0_gev`); on each apply job set DA `d` and `m0_gev` so
upstream $z_R$ is remapped before division.

### Manifest shape

Declare a renormalization kernel with `scheme: "self_renormalization"` and
`kernel_id` `ZMSbar_pdf` or `ZMSbar_da`. Bare inputs are either upstream
correlator job ids or `inputs.artifacts` with `stage: "correlator_analysis"`.
Self-renorm knobs are **flat job `params`** (and stage `defaults`). Hybrid-ratio
`zs_fm` is also a flat stage/job parameter; only supporting values such as
`m0_gev` and `delta_m_gev` remain under `scheme_parameters`.

```json
{
  "inputs": {
    "artifacts": [
      { "id": "bare_pdf_reference", "stage": "correlator_analysis", "path": "…", "momentum": "PX0PY0PZ0", "volume": "S96T192", "lattice_spacing_fm": 0.0574, "hadron": "pion", "gfix": "CG" },
      { "id": "bare_da_a06", "stage": "correlator_analysis", "path": "…", "momentum": "PX0PY0PZ6", "volume": "S96T192", "lattice_spacing_fm": 0.0574, "hadron": "pion", "gfix": "CG" }
    ],
    "kernels": [
      {
        "stage": "renormalization",
        "kernel_id": "ZMSbar_da",
        "kernel_path": "src/lamet_agent/kernels.py",
        "scheme": "self_renormalization",
        "kernel_parameters": { "mu": 2.0 }
      }
    ]
  },
  "stages": {
    "renormalization": {
      "defaults": {
        "normalization": false,
        "scheme": "self_renormalization",
        "mu": 2.0,
        "svdcut": 1e-12
      },
      "jobs": [
        {
          "id": "rn_zR_fit",
          "inputs": { "reference": "bare_pdf_reference" },
          "params": { "d": -0.08183 }
        },
        {
          "id": "rn_da_a06",
          "inputs": { "target": "bare_da_a06", "zR": "rn_zR_fit" },
          "params": { "d": 0.19, "m0_gev": -0.094 }
        }
      ]
    }
  }
}
```

### Parameters

| Parameter | Where | Required? | Meaning |
|-----------|--------|-----------|---------|
| `scheme` | stage defaults / job | yes (`"self_renormalization"`) | Selects the self-renorm tool path instead of hybrid ratio. |
| `normalization` | stage defaults / job | no (default `true`) | If `true`, divide bare inputs by lattice $z=0$ before tools. Set `false` when inputs are already $z=0$-normalized (`normalized_at_z0` attr). |
| `d` | **fit** job `params` | **yes** | Fixed continuum/discretization coefficient in the $g(z)$ fit and in the initial $z_R$ construction. Never fitted. Use the reference-operator value (e.g. PDF $d_{\mathrm{pdf}}$). |
| `m0_gev` | **fit** job `params` | no | If set, freeze $m_0$ (GeV) when building $z_R$. If omitted, fit $m_0$ from the first three $g(z)$ points against $\log Z_{\overline{\mathrm{MS}}}^{\mathrm{PDF}}(z)$. |
| `d` | **apply** job `params` | no | If set (alone or with `m0_gev`), remap upstream $z_R$ from the fit-job $(d,m_0)$ onto this operator’s $d$ before $H/(z_R Z_{\overline{\mathrm{MS}}})$. Typical DA value: $0.19$. |
| `m0_gev` | **apply** job `params` | no | Target-operator $m_0$ for the same remap. If only one of `d` / `m0_gev` is set, the other is taken from upstream $z_R$ attrs. |
| `mu` | defaults, job, or `kernel_parameters` | no (tool default `2.0`) | Renormalization scale (GeV) for $Z_{\overline{\mathrm{MS}}}$ and related logs. |
| `svdcut` | defaults / fit job | no (default `1e-12`) | SVD cut for the correlated $g(z)$ (and optional $m_0$) fits. |
| `kernel_id` | job or unique `inputs.kernels` entry | yes if multiple kernels | `ZMSbar_pdf` or `ZMSbar_da`; choose the conversion factor for the **apply** target. Fit diagnostics compare $m_R$ to `ZMSbar_pdf` regardless. |
| `lqcd`, `k`, `cf`, `b0` | rarely overridden | no | Advanced fit constants (defaults match the usual self-renorm ansatz). |

Job roles:

| Role | Job type | Points to |
|------|----------|-----------|
| `reference` | fit | Bare zero-momentum `EnsembleData` (often multi-$a$ on `(a,z)`). |
| `target` | apply | Bare matrix element to renormalize. |
| `zR` | apply | Fit job id whose NetCDF / store output holds $z_R$. |

### Outputs

- Fit job: `<artifacts>/renormalization/<fit_job_id>.nc` ($z_R$), plus fit panels
  (`*_fit_lnM_vs_inv_a`, `*_fit_mR_zmsbar`, `*_fit_m_over_zR`, `*_fit_f1`).
- Apply job: `<artifacts>/renormalization/<apply_job_id>.nc` (renormalized ME),
  ME plot, `*_zmsbar_compare`; the last apply job with sibling NetCDFs present
  can also write stage-level `discrete_effect_re` / `discrete_effect_im`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis]"
```

Validate and run manifest:

```bash
lamet-agent validate examples/cg_pion_pdf_manifest.json
lamet-agent run examples/cg_pion_pdf_manifest.json
```

Interactively plan a draft manifest before running it:

```bash
lamet-agent plan draft_manifest.jsonc --backend api --model deepseek/deepseek-chat
lamet-agent plan draft_manifest.jsonc --backend codex
```

`plan` accepts incomplete JSON/JSONC manifests and runs an LLM-controlled
planning loop. The configured backend chooses planning actions such as checking
the manifest, inspecting HDF5 inputs, asking terminal questions, applying
validated JSON Patch edits to the in-memory candidate, and proposing the next
writable plan. Python guardrails apply every manifest mutation to a candidate
copy first, then validate schema, DAG, stage-local contracts, and quick/full
manifest generation before the proposal is shown. On accept it writes
`<artifacts_directory>/plan_manifests/<stem>.quick.json` and
`<artifacts_directory>/plan_manifests/<stem>.full.json`; the original draft is
never overwritten.

The terminal summary is organized as Missing parameters, Inconsistent settings,
Suggested modifications, and Data conversions. The quick manifest uses jackknife
plus mean errors, sets `model_average: false`, and conservatively shrinks
configured scan lists for a low-cost smoke run. The full manifest uses covariance
errors, sets `model_average: true`, and can add recommended search expansions.
If you choose revise, the revision text is routed back through the planning LLM
instead of a fixed phrase matcher, so natural-language requests such as adding a
`renormalization` stage are handled as validated manifest patches.

The first planning release only prepares files listed in
`inputs.correlators[*].data_path`. If a correlator HDF5 file does not already use
the standard reader layout, plan mode can write converted files under
`<artifacts_directory>/plan_data/` and update the generated manifests to point at
those files. `inputs.artifacts` are not converted by plan mode.

Artifact placement and stage order come from the manifest. The complete first-phase
CG pion PDF check is available in `runs/ds_pdf_complete/run.sh`:

```bash
cd runs/ds_pdf_complete
./run.sh
```

`root_directory` resolves relative to the manifest file when it is not absolute.
Correlator, artifact, kernel, and artifact-output paths resolve from that root.
`metadata.stages` is the sole ordered list of stages to execute; partial runs use a
manifest with a shorter list and source nodes under `inputs.artifacts`.

`examples/cg_pion_pdf_manifest.json` runs the current P0/P5 workflow through
correlator analysis, hybrid-ratio renormalization, Fourier transformation, and
perturbative matching. `examples/partial_cg_pion_pdf_manifest.json` starts from
the saved `rn_p5` renormalization artifact and runs only Fourier and matching.
External partial-run sources declare the discrete kinematic triple `momentum`,
`volume`, and `lattice_spacing_fm`; the framework derives `momentum_gev` from
those values just as it does for correlator inputs. `hadron` and `gfix` may also
be supplied as provenance.

## Standard Correlator HDF5 Format

Each standard correlator file contains one ensemble and one correlator type. A
file may combine any number of compatible momentum settings, and a 3pt file may
also combine multiple source-sink separations. The corresponding manifest entry
uses `correlator_type: "2pt"` or `"3pt"`; the name reserves room for future
correlator types, but 4pt data are not currently accepted.

The operator fields are free strings:

- `source_operator` and `sink_operator` are required for both types.
- `current_operator` is required for 3pt data.
- `bz_direction` is required for 3pt data and must be one of `X`, `Y`, `Z`,
  `XY`, `XZ`, `YZ`, or `XYZ`. It records the spatial direction or canonical
  direction set represented by the `bz` separation grid.
- Gamma structures use labels such as `g5`. Append `_nonlocal` when locality is
  part of the distinction, for example `gT_nonlocal`. This also allows a local
  PDF 2pt input and a nonlocal DA 2pt input to remain distinguishable.

Datasets use these paths and axis orders:

- 2pt: `<source_operator>/<sink_operator>/<momentum>`, shape `(Lt, n_cfg)`.
- 3pt: `<source_operator>/<sink_operator>/<current_operator>/<momentum>/tsep<tsep>/bT<bT>/bz<bz>`,
  shape `(tsep + 1, n_cfg)`.

There is no `source_sink`, `bz_direction`, or `eta` path layer. The manifest is
authoritative for `bz_direction`; an HDF5 root attr with the same name is
optional provenance. Files with different `bz_direction` settings remain
separate because their standard dataset paths would otherwise collide. `Lt`
must equal the temporal extent encoded in the manifest `volume`. For example,
`volume: "S48T64"` means 48 sites in each spatial direction and 64 time slices.

`bz` lists nonlocal-current separations along `bz_direction`, conventionally
the longitudinal direction relative to momentum. `bT` lists separations in
the transverse directions. Both are integer lattice-site separations; the
current correlator fitter supports exactly one `bT` value per 3pt entry.

A minimal shared-input declaration is:

```json
{
  "inputs": {
    "correlators": [
      {
        "correlator_id": "ensemble_2pt",
        "correlator_type": "2pt",
        "data_path": "data/ensemble_2pt.h5",
        "ensemble": "HISQa060_X",
        "hadron": "pion",
        "gfix": "CG",
        "source_operator": "g5",
        "sink_operator": "g5",
        "volume": "S48T64",
        "lattice_spacing_fm": 0.0574,
        "momentum": ["PX0PY0PZ0", "PX5PY0PZ0"]
      },
      {
        "correlator_id": "ensemble_3pt",
        "correlator_type": "3pt",
        "data_path": "data/ensemble_free_3pt.h5",
        "ensemble": "HISQa060_X",
        "hadron": "pion",
        "gfix": "CG",
        "source_operator": "g5",
        "sink_operator": "g5",
        "current_operator": "gT_nonlocal",
        "bz_direction": "X",
        "volume": "S48T64",
        "lattice_spacing_fm": 0.0574,
        "momentum": ["PX0PY0PZ0", "PX5PY0PZ0"],
        "tsep": [8, 10, 12],
        "bT": [0],
        "bz": [0, 1, 2]
      }
    ]
  }
}
```

The matching HDF5 tree includes, for example,
`g5/g5/PX5PY0PZ0` and
`g5/g5/gT_nonlocal/PX5PY0PZ0/tsep10/bT0/bz2`. A file containing only one
momentum or one `tsep` uses the same layout and a one-element manifest list.
Each correlator-analysis job still selects exactly one momentum through scalar
`job.params.momentum`; NonBreit jobs instead select scalar
`initial_momentum` and `final_momentum`.

Momentum labels have the exact form `PXnPYnPZn`, where every component is a
signed integer. Their physical magnitude is derived without an intermediate
rounding step:

\[
p\,[\mathrm{GeV}] =
\frac{2\pi\hbar c}{L_s\,a\,[\mathrm{fm}]}
\sqrt{n_x^2+n_y^2+n_z^2},
\qquad \hbar c = 0.1973269804\ \mathrm{GeV\,fm}.
\]

Here `L_s` comes from `volume`, and `a` is `lattice_spacing_fm`.

If a source dataset is an unambiguous transpose of the expected shape, plan mode
may transpose it during conversion and records provenance attributes on the output
dataset.

Valid stage IDs:

| Stage ID | Package |
| --- | --- |
| `correlator_analysis` | correlator |
| `renormalization` | renorm |
| `fourier_transform` | fourier |
| `perturbative_matching` | matching |
| `extrapolation` | extrapolation |

Print each agent cycle (prompt, model action, tool observation) while the run
executes:

```bash
lamet-agent run examples/cg_pion_pdf_manifest.json --backend api --model deepseek/deepseek-chat --verbose
```

Choose the LLM integration with `--backend` (`mock`, `external`, `api`, or `codex`).
`codex` uses the Codex Python SDK and the current Codex login, so install the optional
extra first:

```bash
python -m pip install -e ".[codex]"
lamet-agent run examples/cg_pion_pdf_manifest.json --backend codex --verbose
```

The `api` backend reads the API key from `--api-key-file` (default `api.key`) or the
provider environment variable (`DEEPSEEK_API_KEY` / `OPENAI_API_KEY`). Pass
`--model provider/model_id` (shorthand `provider` uses that provider's default model).
Override the HTTP endpoint with `--base-url` when needed:

```bash
lamet-agent run examples/cg_pion_pdf_manifest.json --backend api --model openai/gpt-4o-mini --verbose
lamet-agent run examples/cg_pion_pdf_manifest.json --backend api --model openai/gpt-4o
```

Replay a deterministic JSONL action transcript (tests and regression):

```bash
lamet-agent run examples/cg_pion_pdf_manifest.json --backend external --actions-path actions.jsonl
```

Run the agent loop without a real LLM (dev/test smoke only):

```bash
lamet-agent run examples/cg_pion_pdf_manifest.json --backend mock
```

## File Responsibilities

- `src/lamet_agent/manifest.py`
  - Defines the `metadata`, source `inputs`, and stage-job schema.
  - Validates ids, ordered job references, and root-relative paths.
- `src/lamet_agent/core/stages.py`
  - Maps stage IDs to concrete stage packages.
- `src/lamet_agent/core/data.py`
  - Defines typed data containers (`EnsembleInfo`, `EnsembleData`) for resampled
    lattice data.
  - Serializes stage artifacts with `EnsembleData.to_netcdf` /
    `EnsembleData.from_netcdf` (NETCDF4, complex-aware).
  - Provides common data operations (resampling, coordinate transforms, and
    cross-stage arithmetic/alignment helpers).
- `src/lamet_agent/core/prompting.py`
  - Stores `SYSTEM_PROMPT` and shared output-format hint.
  - Builds static context once per job; incremental tool observations are
    appended as separate user turns in multi-turn LLM sessions.
- `src/lamet_agent/core/llm.py`
  - Pluggable `LlmSession` backends: `mock`, `external` (JSONL transcript), `codex`
    (Codex Python SDK), and `api` (OpenAI-compatible HTTP via `PROVIDERS`).
  - `parse_api_model()` splits `provider/model_id` CLI specs; `PROVIDERS` holds each
    provider's base URL, default model, and API-key env var; shared HTTP lives in
    `_post_chat_completion` (add new OpenAI-compatible providers to `PROVIDERS`).
- `src/lamet_agent/core/tools.py`
  - Resolves a stage's `STAGE_TOOLS` registry for the agent loop.
  - `prepare_tool_args()` / `filter_tool_kwargs()` normalize LLM tool calls
    (manifest paths, plot `save_path` under `artifacts/`).
  - `resolve_plot_save_path()` keeps plots under the manifest's stage artifact directory.
- `src/lamet_agent/manifest_params.py`
  - Loads each stage's lightweight `MANIFEST_PARAM_SCHEMA` and recursively
    rejects unknown `defaults` / `params` keys before DAG execution.
- `src/lamet_agent/stage_registry.py`
  - Owns the stage-id to package mapping re-exported by `core/stages.py`.
- `src/lamet_agent/core/trace.py`
  - Optional ReAct-style stdout trace (`--verbose`).
  - Default (non-verbose) runs print a LaMET Agent ASCII banner and one line per
    job (`Stage: … | Job: …`) before stage tool progress output.
- `src/lamet_agent/core/banner.py`
  - GRID-style startup banner and job header formatting for quiet CLI runs.
- `src/lamet_agent/core/plotting.py`
  - Self-contained publication-style plotting (default plot, 2pt fit-on-data).
- `src/lamet_agent/agent.py`
  - `run_agent()` executes `metadata.stages`, runs each declared job with an
    isolated store, and registers `store["output"]` under the job id.
- `src/lamet_agent/cli.py`
  - Exposes `validate` and `run` commands.
  - `run` requires `--backend` (`mock`/`external`/`api`/`codex`), accepts
    `--model provider/model_id` (for `api`), `--verbose` / `-v` (ReAct-style trace
    to stdout), `--actions-path` (for `external`), and `--api-key-file`/`--base-url`
    (for `api`), plus `--report_language en|ch` to select the single report language
    written for each stage.
- `src/lamet_agent/kernels.py`
  - Built-in kernel function examples for smoke tests.
- `src/lamet_agent/stages/*`
  - Each stage owns `params.py`, `prompts.py`, `skills.py`, `functions.py`, and,
    when it writes a report, `reporting.py`.
  - `params.py` declares the user-authored manifest keys, including allowed
    nested keys; tool-only and runner-derived arguments do not belong there.
  - `prompts.py` contains the stage instruction text and action protocol.
  - `skills.py` performs stage-local checks plus `STAGE_SKILL` strategy text and
    a `tool_catalog()`.
  - `functions.py` holds the stage tools and a `STAGE_TOOLS` registry.
  - `reporting.py` controls the per-stage report that is generated after the stage
    finishes, so users can track the analysis progress and inspect intermediate
    results.
  - `stages/correlator/` is the first worked example and exposes four agentic
    tools (requires the `analysis` optional dependencies):
    `inspect_correlator_scale` (choose a `correlator_rescale`), `tune_ground_state`
    (2pt-only window scan + model average), `tune_bare_matrix` (scan bare-matrix fit
    windows on sample-average data for one representative z), and
    `fit_bare_matrix_grid` (apply one shared tuned window to every z and every
    resampled sample, then export a bare-matrix NetCDF artifact, fit-on-data PDFs,
    and split logs). The agent tunes once on sample-average data, then applies the
    same data window everywhere; `model_average=true` BMA-combines fit-function
    candidates within that fixed window.
- `examples/fake_data/generate_fake_data.py`
  - Generates fake correlator-style datasets used for local testing.
- `examples/sample_manifest.jsonc`
  - Annotated reference manifest (JSONC). Copy it, drop the `//` comments, and save
    as `.json` to author a real run.
- `examples/cg_pion_pdf_manifest.json`
  - Runnable P0/P5 correlator and hybrid-ratio renormalization manifest.
- `examples/temp_self_renorm_manifest.json`
  - Renorm-only self-renormalization smoke (PDF reference → DA mom=6 targets);
    see [Self-Renormalization](#self-renormalization). Prepare/run helpers live
    under `runs/ds_self_renorm/`.

## Agent Workflow

1. CLI receives a manifest path and runtime options (`--backend`, `--verbose`).
2. `manifest.py` validates source ids, job ids, ordered dependencies, and paths.
3. `agent.py` executes the ordered `metadata.stages` list.
4. For each stage job:
   - `core/tools.validate_stage_inputs()` surfaces missing inputs as
     `input_issues`.
   - `core/prompting.build_stage_static_prompt()` assembles static context once
     (system prompt, job inputs, effective params, tool catalog).
   - `core/llm.make_llm_session()` provides a pluggable `LlmSession` that drives a
     multi-turn loop (up to `max_tool_steps`, default 40): the model emits one
     JSON action per cycle; on `call_tool`, `core/tools.prepare_tool_args()` and
     `resolve_stage_tools()` run the tool and return an observation as the next
     user turn; terminal tools place the primary data in `store["output"]`.
  - After the stage finishes, the stage's `reporting.py` emits one report in the
    selected language so users can track analysis progress and inspect that stage's
    intermediate results.
5. Session backends: `mock` (deterministic scaffold), `external` (JSONL
   transcript replay via `--actions-path`), `codex` (Codex Python SDK), or `api`
   (OpenAI-compatible chat-completions providers in `core/llm.py` via
   `--model provider/model_id`).
6. The run ends with a compact JSON summary on stdout (`run_id`, `status`,
   `summary`, manifest paths, etc.). By default, stdout first shows a LaMET Agent
   banner and one line per job (`Stage: … | Job: …`) before stage tool progress
   bars; use `--verbose` for per-cycle ReAct-style logging instead. Programmatic
   callers using `run_agent()` still receive `actions` and `stage_results` in the
   return dict.

## Current Status

- `validate` already enforces schema + kernel import checks.
- `run` executes the stage loop and collects structured actions.
- Real provider API wiring lives in `core/llm.py` (DeepSeek today).
