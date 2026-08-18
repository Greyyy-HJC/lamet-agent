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

The renormalization stage supports pointwise ratio, hybrid-ratio, and
hybrid-self-renormalization workflows within the job DAG.

## Minimal Structure

```text
.
├── examples/
│   ├── fake_data/
│   │   └── generate_fake_data.py
│   ├── sample_manifest.jsonc
│   └── pion_pdf_cg_manifest.json
├── lamet_agent/
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent.py
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
│       │   ├── prompts.md
│       │   ├── validation.py
│       │   └── functions.py
│       ├── renorm/
│       ├── fourier/
│       ├── matching/
│       └── extrapolation/
├── lamet_literature/
│   ├── README.md
│   ├── arxiv.py
│   ├── classify_arxiv.py
│   └── arxiv.json
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
- `stages`: `defaults` plus a `jobs` list. A job's `params` recursively merge
  over defaults; nested mappings merge by key, while lists and scalars are
  replaced by the job value. Later jobs reference earlier job ids through
  role-named `inputs`.

Use it as a template and save runnable manifests as plain `.json`. The loader also
accepts JSONC for annotated authoring templates.

Stage `defaults` and job `params` use closed, stage-specific parameter contracts.
`lamet-agent validate` rejects unknown top-level and nested keys instead of
silently dropping them when tool arguments are prepared. Typographical errors
include the closest supported key when one is available. Stage keys under
`stages` that are missing from `metadata.stages` are also rejected: add them to
the run list or delete the unused block. Matching jobs whose `lc_x_ls` is denser
than the quasi grid print a boxed warning on `validate` and `run`; `validate`
then fails, while `run` continues and kernel construction still rejects that
density at runtime.
Runner-owned settings such as `workers`, `random_seed`, and `sample_error_mode`
belong under `metadata`; derived quantities such as `momentum_gev` must not be
written as stage parameters. Full workflows derive them from their upstream
correlators, while partial workflows read `momentum`, `volume`, and
`lattice_spacing_fm` from standard `EnsembleData` NetCDF attrs. Legacy artifacts
missing those attrs may declare the complete triple on `inputs.artifacts[]` as a
fallback.

Every stage owns exactly one typed, executable parameter contract in its
`validation.py`. The contract contains the complete parameter tree, physical
explanations, cross-parameter/context constraints, structured diagnostics,
planning notes, and any deterministic draft normalization. The central
`STAGE_PARAM_CONTRACTS` mapping contains lazy import routes only.

Final validation and incomplete-draft planning both construct a resolved job
context and call the same `STAGE_PARAM_CONTRACT.evaluate()`. Required parameters
such as Fourier `y_grid` are therefore declared once by
`ParameterSpec.required`; planning only converts the returned issue into a
question. `ConstraintSpec` similarly owns one executable check together with
its physical reason and suggested repair. For example, a named Fourier `sector`
cannot be combined with manual `part`, `output_scale`, or `im_flip_for_ft`: the
sector already fixes the complex channel, normalization, and negative-$x$
convention.

There are intentionally two manifest-validation layers, but only one authority
for each concern. `AnalysisManifest` in `manifest.py` defines the global
`metadata`/`inputs`/`stages` envelope, source records, paths, ids, and DAG.
The selected stage contract defines that stage's `defaults` and job `params`
subtree and semantic behavior. Nested field maps inside a contract are parts of
that one tree, not independent schemas. The small JSON schemas used by the run
and planning LLM loops describe agent action-response protocols; they do not
validate manifests and are deliberately separate.

For a human-readable view generated directly from that authority, run:

```bash
lamet-agent describe-stage fourier_transform
lamet-agent describe-stage renormalization
```

The output includes accepted shapes, defaults, units, option-specific behavior,
physical explanations, input-role meanings, and cross-parameter rules. The same
contract data is injected into both the planning and execution LLM prompts.

## Cross-Stage Manifest Semantics

This section keeps only conventions that span the global input schema or more
than one stage. Individual stage parameter behavior is generated from
`validation.py` by `describe-stage` and is not intended to be duplicated here.

### Ratio renormalization

Renormalization uses two independent stage parameters. `scheme` is the physical
scheme (`ratio`, `hybrid`, or `msbar`), while `strategy` selects how it is
implemented (`external_denominator` or `self_renormalization`). Perturbative matching owns only
`scheme`; its value must match the scheme token in the selected `kernel_id`.
Kernel declarations under `inputs.kernels` no longer carry a `scheme` field.

Set `scheme: "ratio"` and `strategy: "external_denominator"` on a renormalization stage or job
to divide the target and denominator pointwise on the complete coordinate grid
for every resampled sample:

$$
h_s^R(z) = \frac{h_s^{\mathrm{target}}(z)}{h_s^{\mathrm{denominator}}(z)}.
$$

`external_denominator`-strategy jobs use the same `{target, denominator}` input roles as hybrid
jobs, but do not require `zs_fm` and do not apply a fixed denominator or a
long-distance exponential correction. Hybrid-only settings (`zs_fm`,
`scheme_parameters.m0_gev`, and `scheme_parameters.delta_m_gev`) are ignored if
they remain in shared defaults. The `normalization` preprocessing described
above still applies; set it to `false` for a direct ratio of raw bare inputs.

Both ratio and hybrid jobs using the `external_denominator` strategy consume lattice-unit `z` coordinates and
require a positive finite `lattice_spacing_fm` on the target data. Their
terminal `EnsembleData`, `store["matrix_element"]`, and NetCDF artifact convert
the coordinate to signed physical distance as
$z_{\mathrm{fm}}=(z/a)a_{\mathrm{fm}}$ and record `coord_unit: "fm"` plus
`input_coord_unit: "lattice"`. Hybrid-ratio branch selection and its
long-distance exponent continue to use $|z_{\mathrm{fm}}|$.

### `inputs.correlators[].polarization` and Fourier sectors

Every example 3pt correlator declares `polarization` explicitly as
`unpolarized`, `helicity`, or `transversity`; the schema default remains
`unpolarized`. Correlator and
renormalization NetCDF outputs preserve it together with `current_operator`, so
Fourier jobs infer the short observable, such as `pion_quark_quasi_pdf`, from
`target_observable`, `parton`, and upstream `hadron`, while inheriting
`polarization` separately from the target 3pt. Partial PDF/GPD inputs without
that provenance must supply both `observable` and `polarization` explicitly.

Quark PDF/GPD jobs support `sea`, `valence`, `singlet`, and `full`. Helicity
interchanges the real/imaginary channels used by `valence` and `singlet` and
uses $\Delta q_{\rm ext}(-x)=+\Delta\bar q(x)$; unpolarized and transversity use
$q_{\rm ext}(-x)=-\bar q(x)$. Gluon jobs use `full` only and do not inherit
quark/antiquark sector semantics. The current gluon tail backend supports the
unpolarized gluon PDF only; gluon helicity, transversity, and GPD operators can
carry `polarization` metadata but are not silently mapped onto that
backend. DA behavior is unchanged.

### Per-job hybrid `zs_fm`

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

## Self-Renormalization Strategy

The self-renormalization strategy (`strategy: "self_renormalization"`) fits the
zero-momentum **reference** over the full coordinate range and uses short-distance
$\overline{\mathrm{MS}}$ matching to fix the finite renormalization:

$$
g(z)-\ln Z_{\overline{\mathrm{MS}}}^{\mathrm{PDF}}(z;\mu)\simeq m_0z+b,
\qquad
z_R(z,a)=\exp[\ln M_{\mathrm{fit}}(z,a)-g(z)+m_0z].
$$

The resulting factor is defined only on the reference grid and is applied as

$$
H_{\mathrm{ren}}(z) = \frac{H_{\mathrm{bare}}(z)}{z_R(z,a)\,Z_{\overline{\mathrm{MS}}}(z)}
$$

for every retained target sample. By default, apply detects missing
long-distance points, infers $f_1(z)$ from the
available $z_R$, fits its long-distance tail quadratically, and rebuilds only
the missing $z_R$ points. No endpoint is frozen, and no explicit extension
length or fit boundary is required. `scheme_parameters.z_coverage_policy: "strict"` can require
complete coverage instead, while `intersection` explicitly keeps only the
target/$z_R$ overlap.

Targets with `coord_unit: "lattice"` are converted inside this scheme as
$z_{\mathrm{fm}}=|z/a|a_{\mathrm{fm}}$; `fm` targets and legacy targets without
the attribute are already physical coordinates. After conversion, `z=0` is
excluded from coverage checks and from the evaluation of $z_R$ and
$Z_{\overline{\mathrm{MS}}}$, then its already-normalized target samples are
passed through unchanged and merged back into the output. Hybrid-self outputs
therefore retain the complete coordinate grid, including $H^R(0)=1$, in fm.

With `scheme: "ratio"`, there is no explicit $z_s$ switch and the apply formula
above is used over the full nonzero coordinate range. With `scheme: "msbar"`,
the apply formula is instead $H_{\mathrm{bare}}/z_R$.

With `scheme: "hybrid"`, apply jobs additionally require a `denominator` input
and flat `zs_fm`. They use the pointwise target/denominator ratio for
$|z|\le z_s$ and

$$
H^R_s(z)=\frac{H^{\mathrm{target}}_s(z)}{z_R(z,a)Z_{T,s}},
\qquad
Z_{T,s}=\frac{H^{\mathrm{denominator}}_s(z_s^{\mathrm{grid}})}
{z_R(z_s^{\mathrm{grid}},a)}
$$

for $|z|>z_s$. $Z_{T,s}$ is constant in $z$ but is constructed per resample,
which keeps the two branches continuous at the nearest switch-grid point and
propagates denominator uncertainty.
The stage always splits into **one fit job** plus one or more **apply jobs**.
See `examples/temp_self_renorm_manifest.json` and
`runs/ds_self_renorm/` for a runnable PDF→DA smoke test.

### Workflow

```text
inputs.artifacts (bare reference + bare targets)
        │
        ▼
┌───────────────────────┐
│ fit job {reference}   │  fit_self_renormalization_factor
│ scheme params d req.  │  → store['zR'] / <job_id>.nc
│ reference m0 fitted   │  → fit diagnostics (ln|M|, mR, f1, …)
└───────────┬───────────┘
            │ zR job id
            ▼
┌───────────────────────┐
│ apply job {target,zR} │  optional scheme d / m0 remap zR
│ per lattice / momentum│  → H/(zR ZMSbar) NetCDF + ME plot
└───────────────────────┘  → zmsbar_compare; last apply may emit
                             stage-level momentum-resolved discrete-effect plots
```

Typical agent tool order:

1. **Fit job** (`inputs` exactly `{ "reference": "<bare_ref_id>" }`):
   `fit_self_renormalization_factor` → `plot_self_renormalization_diagnostics` → finish.
2. **Apply job** (`ratio`/`msbar` inputs are `{target, zR}`; `hybrid` also requires `denominator`):
   `apply_self_renormalization` → `plot_self_renormalization_diagnostics` →
   `plot_renormalized_matrix_element` → finish.

Same-operator use (zero-momentum PDF → finite-$P_z$ PDF): fit with the PDF
`scheme_parameters.d` ($m_0$ of the reference operator is fitted), and leave
apply jobs without `d`/`m0_gev`
overrides. Cross-operator use (PDF reference → DA targets): fit with PDF `d`
(and do not set `m0_gev` on the fit job); on each apply job set DA `d` and
`scheme_parameters.m0_gev` so the target operator can use a different finite renormalization and
upstream $z_R$ is remapped before division.

### Manifest shape

Declare a renormalization kernel with `kernel_id` `ZMSbar_pdf` or `ZMSbar_da`.
Bare inputs are either upstream
correlator job ids or `inputs.artifacts` with `stage: "correlator_analysis"`.
Self-renormalization-specific knobs are grouped under
`scheme_parameters`; `kernel_id`, `mu`, and the cross-scheme `normalization`
setting remain outside that object. Hybrid `zs_fm` remains a flat
stage/job parameter.

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
        "kernel_path": "lamet_agent/kernels.py",
        "kernel_parameters": { "mu": 2.0 }
      }
    ]
  },
  "stages": {
    "renormalization": {
      "defaults": {
        "normalization": false,
        "scheme": "ratio",
        "strategy": "self_renormalization",
        "mu": 2.0,
        "scheme_parameters": { "LambdaQCD_gev": 0.1 }
      },
      "jobs": [
        {
          "id": "rn_zR_fit",
          "inputs": { "reference": "bare_pdf_reference" },
          "params": {
            "scheme_parameters": { "d": -0.08183 }
          }
        },
        {
          "id": "rn_da_a06",
          "inputs": { "target": "bare_da_a06", "zR": "rn_zR_fit" },
          "params": {
            "scheme_parameters": {
              "d": 0.19,
              "m0_gev": -0.094
            }
          }
        }
      ]
    }
  }
}
```

### Parameters

Run `lamet-agent describe-stage renormalization` for the generated parameter
reference, including fit/apply distinctions, choice behavior, defaults, units,
coverage policy, linked-chain constraints, and removed-parameter migrations.

Stage defaults and job params recursively merge. Put shared values such as the
required `LambdaQCD_gev` in defaults; job-level `scheme_parameters` can then
override only operator-specific values such as `d` or `m0_gev`.

Job roles:

| Role | Job type | Points to |
|------|----------|-----------|
| `reference` | fit | Bare zero-momentum `EnsembleData`, often multi-$a$ on `(a,z)`, from one discretization family. All spacings share $g(z)$ and one $f_1(z)$. The obsolete `discretization_groups` metadata is rejected. |
| `target` | apply | Bare matrix element to renormalize. |
| `zR` | apply | Fit job id whose NetCDF / store output holds $z_R$. |

### Outputs

- Fit job: `<artifacts>/renormalization/<fit_job_id>.nc` ($z_R$ on exactly the
  reference grid), plus fit panels
  (`*_fit_lnM_vs_inv_a`, `*_fit_mR_zmsbar`, `*_fit_m_over_zR`, `*_fit_f1`).
- Apply job: `<artifacts>/renormalization/<apply_job_id>.nc` (renormalized ME),
  ME plot, `*_zmsbar_compare`; the last apply job with sibling NetCDFs present
  writes one stage-level `discrete_effect_<momentum>_re/im` pair per momentum,
  with only the corresponding lattice spacings overlaid in each figure.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis]"
```

Validate and run manifest:

```bash
lamet-agent describe-stage correlator_analysis
lamet-agent validate examples/pion_pdf_cg_manifest.json
lamet-agent run examples/pion_pdf_cg_manifest.json
```

`describe-stage` prints the stage-owned parameter and physics reference without
requiring a manifest.

`validate` now runs stage-local validation after schema, DAG, and path checks.
Its JSON result includes structured `issues` with stable codes, manifest paths,
immediate causes, physical explanations, and suggested fixes. Any issue makes
the command exit nonzero. The same diagnostics are passed to the planning LLM
and to the per-job agent context.

Interactively plan a draft manifest before running it:

```bash
lamet-agent plan draft_manifest.jsonc --backend api --model deepseek/deepseek-chat --api-key-file api.key
lamet-agent plan draft_manifest.jsonc --backend codex --model CODEX_MODEL_ID
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

Plan questions are limited to fields that are missing, invalid, or genuinely
ambiguous. A valid manifest proceeds directly to candidate generation: its
`metadata.stages` list is treated as the complete requested execution scope,
and already valid optional stage parameters are not presented for confirmation.

`run` still performs strict manifest validation before starting any stage. If
that validation fails with the `api`, `codex`, or `mock` backend, it prints a
framed fallback notice followed by the validation error and automatically
enters the same interactive planning loop,
using the run command's backend, model, API key file, and base URL settings.
Accepting the fallback plan writes the quick and full manifests and then ends
the command; run one of those generated manifests explicitly to execute it.
The `external` backend cannot drive interactive planning, so its validation
failures continue to exit as CLI errors.

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
For CLI `validate` and `run`, the resolved value must be this `lamet-agent`
checkout's project root. Correlator data, external artifact, and kernel paths
resolve from that root and must name existing files; `artifacts_directory` and
other output paths may be created later and therefore need not exist during
validation. When one of these path checks fails, `run` with a planning-capable
backend enters plan mode without running workflow stages, confirms the project
root first, and then asks for each invalid input path in manifest order. The
`external` backend reports the path error directly because it cannot plan.

`metadata.stages` is the sole ordered list of stages to execute; partial runs use a
manifest with a shorter list and source nodes under `inputs.artifacts`.
Every key under `stages` must appear in that list; leftover stage blocks fail
`validate`, and `run` falls back to plan mode so the unused stage can be included
or removed.

`examples/pion_pdf_cg_manifest.json` runs the current P0/P5 workflow through
correlator analysis, hybrid-ratio renormalization, Fourier transformation, and
perturbative matching. For a standard `EnsembleData` NetCDF source, an
`inputs.artifacts[]` entry only
needs `id`, `stage`, and `path`: the runner reads the discrete kinematic triple
`momentum`, `volume`, and `lattice_spacing_fm`, plus provenance such as `hadron`,
`gfix`, and `bz_direction`, from the data-variable attrs without loading the
array. The framework derives `momentum_gev` from the resolved discrete values.
Legacy files may use a complete manifest kinematic triple as a fallback. When a
supported field is present in both places, the values must agree or validation
fails before stage execution.

## Standard Correlator HDF5 Format

Each standard correlator file contains one ensemble and one correlator type. A
file may combine any number of compatible momentum settings, and a 3pt file may
also combine multiple source-sink separations. The corresponding manifest entry
uses `correlator_type: "2pt"` or `"3pt"`; the name reserves room for future
correlator types, but 4pt data are not currently accepted.

The operator fields are free strings:

- `source_operator` and `sink_operator` are required for both types.
- `current_operator` is required for 3pt data.
- `polarization` records the 3pt operator family. Write it explicitly as
  `unpolarized`, `helicity`, or `transversity`; the schema default is
  `unpolarized`.
- `bz_direction` is required for 3pt data and must be one of `X`, `Y`, `Z`,
  `XY`, `XZ`, `YZ`, or `XYZ`. It records the spatial direction or canonical
  direction set represented by the `bz` separation grid.
- Gamma structures use labels such as `g5`. Append `_nonlocal` when locality is
  part of the distinction, for example `gT_nonlocal`. This also allows a local
  PDF 2pt input and a nonlocal DA 2pt input to remain distinguishable.

Datasets use these paths and axis orders:

- ordinary 2pt: `<source_operator>/<sink_operator>/<momentum>`, shape `(Lt, n_cfg)`.
- qDA 2pt: `<source_operator>/<sink_operator>/<momentum>/bT<bT>/bz<bz>`,
  shape `(Lt, n_cfg)`; `bT` and `bz` come from correlator metadata rather than
  being appended to either operator name. The reader also accepts the legacy
  `<source>/<nonlocal_sink>_bT<bT>_bz<bz>/<momentum>` layout.
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
lamet-agent run examples/pion_pdf_cg_manifest.json --backend api --model deepseek/deepseek-chat --api-key-file api.key --verbose
```

Choose the LLM integration with `--backend` (`mock`, `external`, `api`, or `codex`).
`codex` uses the Codex Python SDK and the current Codex login, so install the optional
extra first:

```bash
python -m pip install -e ".[codex]"
lamet-agent run examples/pion_pdf_cg_manifest.json --backend codex --model CODEX_MODEL_ID --verbose
```

For the `codex` backend, `--model` accepts a Codex model ID and passes it to the
SDK. Omit `--model` to use the current Codex SDK default. Startup prints a boxed
`codex` summary (`model` and `auth=Codex login`) before the LaMET Agent banner.

The `api` backend reads the API key from `--api-key-file` or, if that flag is
omitted, the provider environment variable (`DEEPSEEK_API_KEY` / `OPENAI_API_KEY`).
The two sources are not mixed: a missing or empty key file does not fall back to
the environment. Startup prints a boxed `api` summary (provider, model, base URL,
and key source — file path or env var name, never the key itself), then a blank
line before the LaMET Agent banner. Pass `--model provider/model_id`
(shorthand `provider` uses that provider's default model). Override the HTTP
endpoint with `--base-url` when needed:

```bash
lamet-agent run examples/pion_pdf_cg_manifest.json --backend api --model openai/gpt-4o-mini --api-key-file api.key --verbose
lamet-agent run examples/pion_pdf_cg_manifest.json --backend api --model openai/gpt-4o
```

The second command works only when `OPENAI_API_KEY` is set in the environment.

Replay a deterministic JSONL action transcript (tests and regression):

```bash
lamet-agent run examples/pion_pdf_cg_manifest.json --backend external --actions-path actions.jsonl
```

Run the agent loop without a real LLM (dev/test smoke only):

```bash
lamet-agent run examples/pion_pdf_cg_manifest.json --backend mock
```

## File Responsibilities

- `lamet_agent/manifest.py`
  - Defines the `metadata`, source `inputs`, and stage-job schema.
  - Validates ids, ordered job references, and root-relative paths.
- `lamet_agent/core/stages.py`
  - Maps stage IDs to concrete stage packages.
- `lamet_agent/core/data.py`
  - Defines typed data containers (`EnsembleInfo`, `EnsembleData`) for resampled
    lattice data.
  - Serializes stage artifacts with `EnsembleData.to_netcdf` /
    `EnsembleData.from_netcdf` (NETCDF4, complex-aware).
  - Provides common data operations (resampling, coordinate transforms, and
    cross-stage arithmetic/alignment helpers).
- `lamet_agent/core/prompting.py`
  - Stores `SYSTEM_PROMPT` and shared output-format hint.
  - Builds static context once per job, including the authoritative stage
    contract; incremental tool observations are appended as separate user turns
    in multi-turn LLM sessions.
- `lamet_agent/core/llm.py`
  - Pluggable `LlmSession` backends: `mock`, `external` (JSONL transcript), `codex`
    (Codex Python SDK), and `api` (OpenAI-compatible HTTP via `PROVIDERS`).
  - `parse_api_model()` splits `provider/model_id` CLI specs; `PROVIDERS` holds each
    provider's base URL, default model, and API-key env var; shared HTTP lives in
    `_post_chat_completion` (add new OpenAI-compatible providers to `PROVIDERS`).
- `lamet_agent/core/tools.py`
  - Resolves a stage's `STAGE_TOOLS` registry for the agent loop.
  - `prepare_tool_args()` / `filter_tool_kwargs()` normalize LLM tool calls
    (manifest paths, plot `save_path` under `artifacts/`).
  - `resolve_plot_save_path()` keeps plots under the manifest's stage artifact directory.
- `lamet_agent/manifest_params.py`
  - Defines reusable parameter specs, constraint descriptions, structured stage
    diagnostics, lazy contract routing, human-facing contract rendering, and
    recursive `defaults` / `params` validation before DAG execution.
- `lamet_agent/core/trace.py`
  - Optional ReAct-style stdout trace (`--verbose`).
  - Default (non-verbose) runs print a LaMET Agent ASCII banner and one line per
    job (`Stage: … | Job: …`) before stage tool progress output.
- `lamet_agent/core/banner.py`
  - GRID-style startup banner and job header formatting for quiet CLI runs.
- `lamet_agent/core/plotting.py`
  - Self-contained publication-style plotting (default plot, 2pt fit-on-data).
- `lamet_agent/agent.py`
  - `run_agent()` executes `metadata.stages`, runs each declared job with an
    isolated store, and registers `store["output"]` under the job id.
- `lamet_agent/__main__.py`
  - Exposes the `describe-stage`, `plan`, `validate`, and `run` commands and backs both
    `python -m lamet_agent` and the `lamet-agent` console script.
  - `run` requires `--backend` (`mock`/`external`/`api`/`codex`), accepts
    `--model model_id` (for `codex`) or `--model provider/model_id` (for `api`),
    `--verbose` / `-v` (ReAct-style trace to stdout), `--actions-path` (for
    `external`), `--api-key-file` or the provider env var plus `--base-url`
    (for `api`), and `--report_language en|ch` to select the single report
    language written for each stage.
- `lamet_agent/kernels.py`
  - Built-in kernel function examples for smoke tests.
- `lamet_literature/arxiv.py`
  - Downloads the arXiv HTML selected by the local INSPIREHEP export.
- `lamet_literature/classify_arxiv.py`
  - Uses the local OpenAI-compatible model to tag downloaded papers by LaMET
    physics topic, workflow relevance, systematics, and lattice setup, writing
    `lamet_literature/arxiv.json` for review retrieval.
- `lamet_agent/stages/*`
  - Each stage owns `prompts.md`, `validation.py`, `functions.py`, and, when it
    writes a report, `reporting.py`.
  - `prompts.md` contains the stage instruction, strategy guidance, and tool catalog.
  - `validation.py` owns the stage's one executable parameter contract,
    physical explanations, semantic input checks, planning guidance, and the
    resolved validation context used by both plan and validate.
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
- `examples/pion_pdf_cg_manifest.json`
  - Runnable P0/P5 correlator and hybrid-ratio renormalization manifest.
- `examples/pion_pdf_gi_manifest.json`
  - Runnable P0/P4 GI pion PDF workflow.
- `examples/temp_pdf_gi_manifest.json`
  - Partial GI pion PDF resume from renormalized `rn_p4.nc` through Fourier and matching.
- `examples/pion_da_gi_manifest.json`
  - Full GI pion DA workflow from qDA correlator analysis through matching and review.
- `examples/kaon_da_gi_manifest.json`
  - Full GI kaon DA workflow from qDA correlator analysis through matching and review;
    API run helpers live under `runs/ds_pion_da_gi/` and `runs/ds_kaon_da_gi/`
    (including `plot_agent_data_compare.py` for agent vs reference overlays).
- `examples/temp_self_renorm_manifest.json`
  - Renorm-only hybrid-self-renormalization smoke (PDF reference → DA mom=6 targets);
    see [Hybrid Self-Renormalization](#hybrid-self-renormalization). Prepare/run helpers live
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
   `--model provider/model_id`). The `codex` backend accepts an optional Codex
   model ID through the same `--model` option.
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
