"""Stage-local helpers and skill guidance for perturbative matching."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest

# Reuse the kernel registry from functions.py so the valid kernel_ids always stay
# in sync with the kernels that actually exist -- no drift between "an id named in
# the prompt" and "an id actually registered".
from lamet_agent.stages.matching.functions import KERNEL_REGISTRY


# --- domain/strategy guidance shown to the LLM ------------------------------
STAGE_SKILL = """
Perturbative-matching skill: turn a finite-momentum quasi-PDF into the
light-cone PDF with an NLO matching kernel, applied sample by sample so the
full resampling axis (and its correlations) carries through.

Inputs:
- The quasi-PDF is the Fourier stage's output, read from disk (each stage starts
  with a fresh store). load_quasi_pdf accepts either the Fourier EnsembleData
  NetCDF artifact (default; grid from coords['x'], quasi-PDF from the requested
  component samples) or a legacy/simple npz with x_ls/quasi_samples.
- component selects the real ('re', default) or imaginary ('im') channel. The
  unpolarized quasi-PDF lives in the real part.

Kernel choice:
- Each operator has its own kernel, registered under a logical kernel_id in
  KERNEL_REGISTRY following the CG_<operator>_PDF_<scheme> convention (e.g.
  CG_gt_PDF_msbar for the unpolarized gamma^t PDF in Coulomb gauge, MSbar scheme,
  reference arXiv:2602.11283 Eq. (2.14)). Call list_kernels to see them all, then
  pick the operator and scheme (msbar / ratio / hybrid) you need.
- pz_gev is the hadron momentum P_z in GeV and must match the Fourier stage; mu
  is the MSbar renormalization scale in GeV (default 2.0).
- Schemes are selected by the kernel_id suffix: "<op>_msbar" is MSbar, "<op>_ratio"
  is the ratio scheme, "<op>_hybrid" is the hybrid scheme. Hybrid kernels also need
  the Wilson-line length zs_fm (z_s in fm) from metadata.matching; together with
  pz_gev it sets zspz = zs_fm * pz_gev / GEV_FM (arXiv:2602.11283 Eq. (2.20)).
  MSbar, ratio and gluon kernels do not use zs_fm.

Strategy:
- The matching is the matrix product lightcone_i = K @ quasi_i applied to every
  resampling sample i independently; the matched samples are stored as an
  EnsembleData and the uncertainty/covariance is rebuilt from them. K is (nx, ny)
  on the x grid.
- The x grid must avoid x=0 because the kernel uses xi=x/y; a symmetric grid
  with an even number of points avoids the singular point. If load_quasi_pdf
  reports a grid that hits 0, regenerate the quasi-PDF on a zero-avoiding grid.
- Sanity check: NLO matching is a perturbative correction, so the light-cone PDF
  should stay close to the quasi-PDF and nearly preserve the norm (integral of
  f(x) dx). A large departure or wild oscillation signals a problem (e.g. a bad
  grid or pz_gev).
- Finish with plot_matched_pdf, which draws quasi vs light-cone as error bands,
  then report_matching_result to write the English and Chinese Markdown reports.
""".strip()


# --- tool catalog: name -> one-line description; core/prompting.py renders it
#     under "Available tools" -------------------------------------------------
TOOL_CATALOG = {
    "list_kernels": "list_kernels() -> the registered kernel_ids you may pass to build_matching_kernel.",
    "load_quasi_pdf": "load_quasi_pdf(path, component='re'|'im') -> read the quasi-PDF and its x grid (stores x_ls, quasi_ed); auto-detects the Fourier EnsembleData NetCDF artifact or a legacy/simple x_ls/quasi_samples npz.",
    "build_matching_kernel": "build_matching_kernel(kernel_id, pz_gev, mu=2.0, zs_fm=None, grid='x_ls') -> the (nx, ny) NLO matching matrix for the chosen operator; the scheme follows the kernel_id suffix (_msbar, _ratio, _hybrid); hybrid kernels require zs_fm (z_s in fm) and form zspz = zs_fm * pz_gev / GEV_FM; x grid must avoid x=0.",
    "apply_matching": "apply_matching(kernel='kernel_matrix', quasi='quasi_ed') -> lightcone_ed: applies the kernel to every sample (lightcone_i = kernel @ quasi_i) and stores the matched PDF as an EnsembleData, rebuilding its statistics from the samples.",
    "plot_matched_pdf": "plot_matched_pdf(save_path=None) -> quasi vs light-cone f(x) comparison as error bands, written to artifacts/matched_pdf.pdf.",
    "report_matching_result": "report_matching_result(save_path=None) -> write an English report at artifacts/report_matching.md and a Chinese companion at artifacts/report_matching_CN.md with the chosen kernel/operator/scheme, pz_gev/mu (and zspz for hybrid), the matching convolution formula, scheme explanation, norm-preservation and quasi-vs-light-cone deviation diagnostics, the embedded comparison plot, and artifact paths.",
}


def tool_catalog() -> str:
    """Return a human-readable catalog of available matching-stage tools."""
    return "\n".join(f"- {name}: {desc}" for name, desc in TOOL_CATALOG.items())


# --- help text: when input validation fails, tell the user/agent what is
#     missing and how to fill it ----------------------------------------------
KERNEL_HELP = (
    "metadata.matching.kernel_id (or a manifest kernels[].kernel_id) is required "
    "and must be one of the registered kernels: "
    f"{sorted(KERNEL_REGISTRY)}. It selects the operator's NLO matching kernel."
)

PZ_HELP = (
    "metadata.matching.pz_gev is required. It is the hadron momentum P_z in GeV "
    "and must match the value used in the Fourier stage; it sets the kernel's "
    "log(4 y^2 P_z^2 / mu^2) terms."
)

ZS_FM_HELP = (
    "metadata.matching.zs_fm is required for hybrid kernels. It is the Wilson-line "
    "length z_s in fm; with pz_gev it forms the dimensionless zspz = zs_fm * pz_gev "
    "/ GEV_FM in the hybrid correction."
)

QUASI_HELP = (
    "metadata.matching.quasi_input is not set; defaulting to the Fourier-stage "
    "artifact artifacts/fourier_result.nc. Set it explicitly to read a different "
    "quasi-PDF artifact."
)


def _selected_kernel_id(manifest: AnalysisManifest) -> str | None:
    """Pick the kernel_id this stage will use: prefer metadata.matching.kernel_id,
    otherwise fall back to the first kernel_id declared in manifest.kernels."""
    matching = manifest.metadata.get("matching", {})
    if isinstance(matching, dict) and matching.get("kernel_id"):
        return str(matching["kernel_id"])
    if manifest.kernels:
        return manifest.kernels[0].kernel_id
    return None


def validate_stage_inputs(manifest: AnalysisManifest) -> list[str]:
    """Return stage-local issues only.

    Only checks the inputs this (matching) stage needs: whether a registered
    kernel was selected, and whether pz_gev was given. A missing quasi_input is
    just a soft note (it defaults to reading the Fourier artifact).
    """
    matching = manifest.metadata.get("matching", {})
    if not isinstance(matching, dict):
        return ["metadata.matching must be an object containing matching-stage options."]

    issues: list[str] = []

    # 1) Kernel: one must be selected, and it must be registered in KERNEL_REGISTRY.
    kernel_id = _selected_kernel_id(manifest)
    if kernel_id is None:
        issues.append(KERNEL_HELP)
    elif kernel_id not in KERNEL_REGISTRY:
        issues.append(f"kernel_id={kernel_id!r} is not registered. {KERNEL_HELP}")

    # 2) pz_gev: required to build the kernel; flag if missing.
    if "pz_gev" not in matching:
        issues.append(PZ_HELP)

    # 3) zs_fm: only hybrid kernels need the Wilson-line length to form zspz.
    if kernel_id is not None and kernel_id.endswith("_hybrid") and "zs_fm" not in matching:
        issues.append(ZS_FM_HELP)

    # 4) quasi input: not an error if absent; the note says it defaults to the
    #    Fourier-stage artifact.
    if not matching.get("quasi_input"):
        issues.append(QUASI_HELP)

    if issues:
        issues.insert(0, "Matching stage needs more metadata before running tools.")
    return issues
