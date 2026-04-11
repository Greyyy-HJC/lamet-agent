# Analysis Model

`lamet-agent` is organized in three layers:

1. Reusable helper functions in `src/lamet_agent/extensions/` and related package modules
2. Stage implementations in `src/lamet_agent/stages/`
3. Curated manifests and input datasets under `examples/`

## Manifest Metadata v1

Workflow manifests keep the top-level `goal` for planner compatibility, but the physics meaning is defined by `metadata.analysis`.

Required top-level metadata:

- `metadata.purpose`: `smoke` or `physics`
- `metadata.analysis.gauge`: `cg` or `gi`
- `metadata.analysis.hadron`: `pion` or `proton`
- `metadata.analysis.channel`: `qpdf` or `qda`
- `metadata.conventions`: short free-form description
- `metadata.setups`: mapping from `setup_id` to lattice-setup metadata

Required setup fields:

- `lattice_action`
- `n_f`
- `lattice_spacing_fm`
- `spatial_extent`
- `temporal_extent`
- `pion_mass_valence_gev`
- `pion_mass_sea_gev`

Required correlator metadata:

- all correlators: `setup_id`, `momentum`, `smearing`
- three-point correlators: `displacement.b`, `displacement.z`, `operator.gamma`, `operator.flavor`

## Observable Naming

The effective observable is derived from `metadata.analysis.channel` and the transverse separation `b`.

- `qpdf` with `b=0` -> `qpdf`
- `qpdf` with `b!=0` -> `qtmdpdf`
- `qda` with `b=0` -> `qda`
- `qda` with `b!=0` -> `qtmdwf`

This naming is carried in family metadata through the stage payloads so downstream stages do not need to re-derive it.

## Stage Responsibilities

- `correlator_analysis`: match 2pt and 3pt inputs by `(setup_id, momentum, smearing)`, run 2pt/3pt analysis, and emit coordinate-space matrix-element families
- `renormalization`: apply scheme-specific renormalization to coordinate-space families.  Supported schemes:
  - `identity` (default): pass families through unchanged
  - `cg_ratio`: CG ratio-scheme for qTMDWF/qTMDPDF — divide by the b=0 reference at the same momentum, then divide by the p=0 reference at the same b.  Both normalization factors use the mean real-part value at z=0
- `fourier_transform`: run family-wise asymptotic extrapolation and transform coordinate-space families into `x` space; qPDF/qDA and qTMDPDF/qTMDWF share the same hadron-specific asymptotic form (reference: Eq. 2.8 of arXiv:2601.12189 for GI gauge; CG gauge adds a power-law suppression) and each nonzero-`b` family is extrapolated independently.  The `imaginary_zeroing` parameter sets all imaginary samples to zero before extrapolation when the physics requires it (e.g. pion DA)
- `evaluation`: extract derived observables that require cross-family (cross-momentum) aggregation.  Dispatches on `method`:
  - `cs_kernel_momentum_ratio`: compute the CS kernel from the logarithmic ratio of x-space quasi distributions at different momenta and optionally reduce to a constant-in-x value per b.  Pairwise systematic bands are estimated from pair-to-pair spread
  - `cs_kernel_momentum_fit`: fit all available momenta simultaneously at each (x, b) point using the perturbative CG hard-matching kernel `H(x, P^z; mu)` (NLL RGR).  The model is `quasi(P^z) = A * H(x, P^z) * (P^z/P^z_ref)^gamma` where `gamma(b)` is the CS kernel extracted directly from the fit.  Pz correlations across gauge configurations are preserved through a joint jackknife covariance.  If `literature_data_path` (a gvar `.gv` file) is supplied, a comparison plot is produced alongside the standard outputs.
  Future methods (form factor, intrinsic soft function) can be added via the same dispatch
- `perturbative_matching`: apply the user kernel family by family
- `physical_limit`: write per-family final outputs and record which setup, momentum, and `b` axes are available for future continuum/chiral/boost extrapolations

## Example Taxonomy

- `workflow_smoke_manifest.json`: tiny tracked full-pipeline smoke test
- `pion_2pt_manifest.json`: minimal pion two-point analysis example
- `proton_cg_qpdf_manifest.json`: canonical proton CG qPDF example using local raw inputs
- `pion_cg_qtmdpdf_manifest.json`: pion CG qTMDPDF example matching the ratio-fit scope of `mp_zdep_samp.py`
- `pion_cg_cs_kernel_manifest.json`: pion CG Collins-Soper kernel end-to-end workflow — from 2pt + QDA correlators at multiple momenta through `cg_ratio` renormalization, imaginary-zeroed Fourier transform, to CS kernel extraction via the `cs_kernel_momentum_fit` method.  Produces an x-dependent CS kernel per b, reduces to a constant via correlated constant fit, and generates a comparison plot against literature results.

## Data Directory

`data/` at the project root holds reference data tracked by git:

- `data/cs_kernel_literature_updated.gv`: compiled CS kernel results from phenomenology and other lattice groups (gvar binary format), used by the literature comparison plot in `cs_kernel_momentum_fit`

Large raw correlator files are gitignored:

- `data/pion_cg_cs_kernel/*.txt`: local-only raw correlator inputs for the CS kernel example

`examples/data/pion_cg_qtmdpdf/` is also local-only and excluded from git.