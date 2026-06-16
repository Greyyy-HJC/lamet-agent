"""Stage-local helpers for renormalization."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


STAGE_SKILL = """
Ratio/hybrid renormalization takes complex bare matrix-element samples on a z
grid and outputs complex renormalized samples on the same grid. For Eq. 15, use
the P=0 matrix element as h(z,0,a), the target boosted matrix element as
h(z,P,a), N=h(0,0,a)/h(0,P,a), and switch from h(z,0,a) to h(zs,0,a) in the
denominator for |z| >= zs. Keep all samples for downstream error propagation.
""".strip()


TOOL_CATALOG = {
    "load_bare_matrix_element_grid": "load_bare_matrix_element_grid(...) -> load bare real/imag txt samples as EnsembleData(z).",
    "apply_ratio_scheme_renormalization": "apply_ratio_scheme_renormalization(...) -> apply Eq. 15 sample-by-sample and write a Fourier-compatible NPZ.",
    "plot_renormalized_matrix_element": "plot_renormalized_matrix_element(...) -> plot renormalized matrix elements to PDF.",
}


def tool_catalog() -> str:
    """Return a human-readable catalog of available renormalization-stage tools."""
    return "\n".join(f"- {name}: {desc}" for name, desc in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest) -> list[str]:
    """Return stage-local issues only."""
    issues = []
    renorm = manifest.metadata.get("renormalization", {})
    if renorm and not isinstance(renorm, dict):
        return ["metadata.renormalization must be an object when provided."]
    if isinstance(renorm, dict):
        if "denominator_report_json" not in renorm and "target_report_json" in renorm:
            issues.append(
                "metadata.renormalization.denominator_report_json is required when renormalization is run from "
                "manifest-provided target_report_json instead of an upstream correlator stage."
            )
        if any(key in renorm for key in {"zs", "delta_m", "m0"}) and "zs" in renorm:
            try:
                float(renorm["zs"])
            except (TypeError, ValueError):
                issues.append("metadata.renormalization.zs must be numeric.")
    return issues
