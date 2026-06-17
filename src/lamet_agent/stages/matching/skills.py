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
light-cone PDF with an NLO matching kernel, carrying gvar errors through.

Inputs:
- The quasi-PDF is the Fourier stage's output, read from disk (each stage starts
  with a fresh store). load_quasi_pdf accepts either the Fourier EnsembleData
  npz (default; grid from coords['x'], quasi-PDF = ft_<component>_mean with
  error sqrt(stat^2 + sys^2)) or a simple npz with x_ls/quasi_mean/quasi_sdev.
- component selects the real ('re', default) or imaginary ('im') channel. The
  unpolarized quasi-PDF lives in the real part.

Kernel choice:
- Each operator has its own kernel, registered under a logical kernel_id in
  KERNEL_REGISTRY. Call list_kernels to see them, then pick the one matching the
  operator (unpolarized_gT for the unpolarized gamma^t PDF in Coulomb gauge,
  reference arXiv:2602.11283 Eq. (2.14)).
- pz_gev is the hadron momentum P_z in GeV and must match the Fourier stage; mu
  is the MSbar renormalization scale in GeV (default 2.0).

Strategy:
- The matching is the matrix product lightcone = K @ quasi; with gvar arrays the
  quasi-PDF uncertainty propagates automatically. K is (nx, ny) on the x grid.
- The x grid must avoid x=0 because the kernel uses xi=x/y; a symmetric grid
  with an even number of points avoids the singular point. If load_quasi_pdf
  reports a grid that hits 0, regenerate the quasi-PDF on a zero-avoiding grid.
- Sanity check: NLO matching is a perturbative correction, so the light-cone PDF
  should stay close to the quasi-PDF and nearly preserve the norm (integral of
  f(x) dx). A large departure or wild oscillation signals a problem (e.g. a bad
  grid or pz_gev).
- Finish with plot_matched_pdf, which draws quasi vs light-cone as error bands.
""".strip()


# --- tool catalog: name -> one-line description; core/prompting.py renders it
#     under "Available tools" -------------------------------------------------
TOOL_CATALOG = {
    "list_kernels": "list_kernels() -> the registered kernel_ids you may pass to build_matching_kernel.",
    "load_quasi_pdf": "load_quasi_pdf(path, component='re'|'im') -> read the quasi-PDF and its x grid (stores x_ls, quasi_gv); auto-detects the Fourier EnsembleData npz (real part, sqrt(stat^2+sys^2) error) or a simple x_ls/quasi_mean/quasi_sdev npz.",
    "build_matching_kernel": "build_matching_kernel(kernel_id, pz_gev, mu=2.0, grid='x_ls') -> the (nx, ny) NLO matching matrix for the chosen operator; x grid must avoid x=0.",
    "apply_matching": "apply_matching(kernel='kernel_matrix', quasi='quasi_gv') -> lightcone_gv = kernel @ quasi with gvar error propagation.",
    "plot_matched_pdf": "plot_matched_pdf(save_path=None) -> quasi vs light-cone f(x) comparison as error bands, written to artifacts/matched_pdf.pdf.",
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

QUASI_HELP = (
    "metadata.matching.quasi_input is not set; defaulting to the Fourier-stage "
    "artifact artifacts/quasi_pdf.npz. Set it explicitly to read a different "
    "quasi-PDF npz."
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

    # 3) quasi input: not an error if absent; the note says it defaults to the
    #    Fourier-stage artifact.
    if not matching.get("quasi_input"):
        issues.append(QUASI_HELP)

    if issues:
        issues.insert(0, "Matching stage needs more metadata before running tools.")
    return issues
