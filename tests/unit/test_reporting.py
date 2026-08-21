"""Unit tests for shared stage-report formatting and artifact paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from lamet_agent.core.reporting import (
    format_report_list,
    format_report_value,
    markdown_artifact_paths,
    resolve_report_target,
)
from lamet_agent.stages.correlator.reporting import _z_fit_table, build_correlator_stage_report_markdown
from lamet_agent.stages.renorm.reporting import build_renorm_stage_report_markdown
from lamet_agent.stages.fourier import reporting as fourier_reporting
from lamet_agent.stages.fourier.reporting import write_fourier_stage_report
from lamet_agent.stages.matching import reporting as matching_reporting


def test_report_formatters_handle_scalars_and_list_previews() -> None:
    assert format_report_value(None) == "not set"
    assert format_report_value(1.23456, digits=4) == "1.235"
    assert format_report_value("label") == "label"
    assert format_report_list([]) == "[]"
    assert format_report_list(range(10), max_items=3) == "[0, 1, 2, ...]"


def test_correlator_report_marks_unestimated_systematics() -> None:
    table = _z_fit_table(
        {
            "z_fits": [
                {
                    "z": 0,
                    "n_failed_samples": 0,
                    "real_sys_sdev": None,
                    "imag_sys_sdev": None,
                }
            ]
        }
    )
    assert "not estimated" in table[-1]


def test_resolve_report_target_selects_one_language_path(tmp_path: Path) -> None:
    path = tmp_path / "stage_report.md"
    assert resolve_report_target(path, "en") == (path, "en")
    assert resolve_report_target(path, "ch") == (tmp_path / "stage_report_CN.md", "ch")
    with pytest.raises(ValueError, match="report_language"):
        resolve_report_target(path, "fr")


def test_markdown_artifact_paths_relativizes_selected_scalar_and_list_paths(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    artifact = tmp_path / "artifacts" / "result.nc"
    plot = tmp_path / "artifacts" / "plot.svg"
    output = markdown_artifact_paths(
        {
            "artifact": artifact,
            "plots": [plot, "relative.svg", None],
            "relative": "already-relative.pdf",
            "untouched": artifact,
        },
        base_dir=report_dir,
        path_keys=("artifact", "relative"),
        list_path_keys=("plots",),
    )
    assert output["artifact"] == "../artifacts/result.nc"
    assert output["plots"] == ["../artifacts/plot.svg", "relative.svg"]
    assert output["relative"] == "already-relative.pdf"
    assert output["untouched"] == artifact


def test_correlator_stage_report_shows_overlay_and_omits_dispersion_from_job_outputs(tmp_path: Path) -> None:
    text = build_correlator_stage_report_markdown(
        jobs=[
            {
                "job_id": "ca_p4",
                "result": {
                    "fit_scope": "3pt_ratio",
                    "fit_strategy": "joint",
                    "auto_window_scan": {
                        "pt2": {
                            "source": "automatic",
                            "stable_tmax": 12,
                            "fallback_reason": None,
                            "pt2_windows": [{"tmin": 3, "tmax": 12}],
                        },
                        "pt3": {
                            "source": "automatic",
                            "pt3_windows": [{"tsep_ls": [8, 10], "tau_cut": 2}],
                        },
                    },
                },
                "artifacts": {
                    "bare_artifact": "ca_p4.nc",
                    "summary_plot": "ca_p4.pdf",
                    "summary_plot_image": "ca_p4.svg",
                    "tuning_log": "fit_logs/tuning.log",
                    "sample_log": "fit_logs/sample.log",
                    "E0_artifact": "dispersion_relation.nc",
                    "dispersion_relation_plot": "dispersion_relation.pdf",
                    "dispersion_relation_image": "dispersion_relation.svg",
                    "matrix_overlay_re_image_ca_HISQa060_X_re": "ca_HISQa060_X_re.svg",
                    "matrix_overlay_im_image_ca_HISQa060_X_im": "ca_HISQa060_X_im.svg",
                },
            }
        ],
        base_dir=tmp_path,
    )

    assert "## HISQa060_X ensemble overview" in text
    assert "![HISQa060_X ensemble overview](ca_HISQa060_X_re.svg)" in text
    assert text.index("ca_HISQa060_X_re.svg") < text.index("ca_HISQa060_X_im.svg")
    assert "### Automatic Window Scan" in text
    assert '"tau_cut":2' not in text
    assert "Generated candidates" not in text
    assert "### Artifacts" in text
    per_job = text.split("### Artifacts", 1)[1].split("### Diagnostic SVGs", 1)[0]
    assert "dispersion_relation" not in per_job
    assert "ca_p4.pdf" not in per_job
    assert "ca_p4.svg" not in per_job
    assert "fit_logs/tuning.log" in per_job
    assert "![Bare matrix element summary](ca_p4.svg)" in text


def test_correlator_qda_report_uses_spectral_ratio_without_3pt_diagnostics(tmp_path: Path) -> None:
    text = build_correlator_stage_report_markdown(
        jobs=[
            {
                "job_id": "ca_qda",
                "result": {"fit_scope": "qda_ratio", "fit_strategy": "chained"},
                "artifacts": {
                    "bare_artifact": "ca_qda.nc",
                    "summary_plot": "ca_qda.pdf",
                    "summary_plot_image": "ca_qda.svg",
                    "sample0_fit_plots": [
                        "fit_logs/qda_bz1_qda_ratio_re.pdf",
                        "fit_logs/qda_bz1_qda_ratio_re.svg",
                    ],
                },
            }
        ],
        base_dir=tmp_path,
    )
    assert 'fit_scope="qda_ratio"' in text
    assert "O_{00}^{(a)}/z_0" in text
    assert "ordinary local-local 2pt input" in text
    assert "O_{00}^{(a)}/z'_0" in text
    assert "automatically normalized" in text
    assert "bz=0" not in text
    assert "analysis_mode" not in text
    assert "### Diagnostic SVGs" in text
    assert "fit_logs/qda_bz1_qda_ratio_re.svg" in text


def test_correlator_stage_report_embeds_sample_quality_svgs(tmp_path: Path) -> None:
    text = build_correlator_stage_report_markdown(
        jobs=[
            {
                "job_id": "ca_p0",
                "result": {
                    "fit_scope": "3pt_ratio",
                    "fit_strategy": "joint",
                    "sample_fit_Q": [0.9, 0.01],
                    "sample_fit_chi2_dof": [0.5, 2.0],
                },
                "artifacts": {
                    "bare_artifact": "ca_p0.nc",
                    "sample_fit_quality_Q_image": "sample_fit_quality_Q.svg",
                    "sample_fit_quality_chi2_image": "sample_fit_quality_chi2.svg",
                    "sample_fit_quality_Q_plot": "sample_fit_quality_Q.pdf",
                    "sample_fit_quality_chi2_plot": "sample_fit_quality_chi2.pdf",
                },
            }
        ],
        base_dir=tmp_path,
    )
    assert "## Sample Fit Quality" in text
    assert "![CDF of per-sample $Q$](sample_fit_quality_Q.svg)" in text
    assert r"![Histogram of per-sample $\chi^2/\mathrm{dof}$](sample_fit_quality_chi2.svg)" in text
    assert "numerically failed" not in text
    assert r"q_{\rm min}" not in text
    assert text.index("## Sample Fit Quality") < text.index("## `ca_p0`")
    per_job = text.split("### Artifacts", 1)[1].split("### Diagnostic SVGs", 1)[0]
    assert "sample_fit_quality" not in per_job


def test_renorm_stage_report_shows_overlay_after_method(tmp_path: Path) -> None:
    text = build_renorm_stage_report_markdown(
        jobs=[
            {
                "job_id": "rn_p4",
                "result": {"scheme": "hybrid", "strategy": "external_denominator", "zs_fm": 0.17},
                "artifacts": {
                    "renormalized_artifact": "rn_p4.nc",
                    "renormalized_plot": "rn_p4.pdf",
                    "matrix_overlay_re_image_rn_HISQa060_X_re": "rn_HISQa060_X_re.svg",
                    "matrix_overlay_im_image_rn_HISQa060_X_im": "rn_HISQa060_X_im.svg",
                },
            }
        ],
        base_dir=tmp_path,
    )

    assert text.index("## Method") < text.index("## HISQa060_X ensemble overview") < text.index("## `rn_p4`")
    assert text.index("rn_HISQa060_X_re.svg") < text.index("rn_HISQa060_X_im.svg")


def test_fourier_stage_report_lists_overlay_last_with_ensemble_description(tmp_path: Path, monkeypatch) -> None:
    translations = []

    def translate(markdown, **kwargs):
        translations.append((markdown, kwargs))
        return markdown

    monkeypatch.setattr(fourier_reporting, "translate_markdown_report", translate)
    path = tmp_path / "ft_report.md"
    write_fourier_stage_report(
        jobs=[
            {
                "job_id": "ft_p4",
                "result": {
                    "momentum_gev": 1.8,
                    "final_momentum_gev": 2.2,
                    "observable": "pion_quark_quasi_pdf",
                    "gfix": "GI",
                    "order": 2,
                    "z_ext_max": 2.0,
                },
                "artifacts": {
                    "fourier_artifact": "ft_p4.nc",
                    "fourier_plot": "ft_p4_xdep.pdf",
                    "fourier_overlay_ft_HISQa060_X_xdep": "ft_HISQa060_X_xdep.pdf",
                    "fourier_overlay_image_ft_HISQa060_X_xdep": "ft_HISQa060_X_xdep.svg",
                },
            },
            {
                "job_id": "ft_p5",
                "result": {"momentum_gev": 2.25, "observable": "pion_quark_quasi_pdf"},
                "artifacts": {"fourier_artifact": "ft_p5.nc", "fourier_plot": "ft_p5_xdep.pdf"},
            },
        ],
        path=path,
        report_language="ch",
    )
    assert path.exists()
    text = path.with_name("ft_report_CN.md").read_text(encoding="utf-8")
    output = text.split("## Output Artifacts", 1)[1].split("## Reading", 1)[0]

    assert "## HISQa060_X ensemble overview" in text
    assert output.rfind("ft_HISQa060_X_xdep.svg") > output.rfind("ft_p5_xdep.pdf")
    assert "Fourier overlay for ensemble HISQa060_X" in output
    assert r"$P_i^z=1.80,\ P_f^z=2.20$" in text
    assert len(translations) == 1


@pytest.mark.parametrize(
    ("polarization", "family", "decomposition", "negative_x_relation"),
    [
        ("unpolarized", "vector family $H,E$", "$H/E$ decomposition", "$q_{\\rm ext}(-x)=-\\bar q(x)$"),
        ("helicity", "axial family $\\widetilde H,\\widetilde E$", "$\\widetilde H/\\widetilde E$ decomposition", "$\\Delta q_{\\rm ext}(-x)=+\\Delta\\bar q(x)$"),
        ("transversity", "tensor family $H_T,E_T,\\widetilde H_T,\\widetilde E_T$", "tensor-GPD decomposition", "$q_{\\rm ext}(-x)=-\\bar q(x)$"),
    ],
)
def test_fourier_gpd_report_records_operator_family_and_projection_limits(
    polarization: str, family: str, decomposition: str, negative_x_relation: str
) -> None:
    text = fourier_reporting.build_fourier_report_markdown(
        result={
            "target_observable": "gpd",
            "observable": "nucleon_quark_quasi_gpd",
            "polarization": polarization,
            "current_operator": "operator",
            "parton": "quark",
            "hadron": "nucleon",
            "momentum_gev": 1.0,
            "final_momentum_gev": 2.0,
            "sector": "sea",
            "component": "both",
            "bilocal_anchor": "barpsi_at_0",
            "hermitian_partner_id": "rn_reverse",
            "gpd_completion_mode": "paired_flow",
            "delta_momentum_gev": 1.0,
            "phase_momentum_source": "signed_discrete_momentum",
            "gfix": "GI",
            "order": "LA",
            "y_grid": [-0.5, 0.0, 0.5],
        }
    )

    assert family in text
    assert decomposition in text
    assert negative_x_relation in text
    assert "projected quasi-GPD matrix element" in text
    assert "negative-$x$ DGLAP region" in text
    assert "ERBL region" in text
    assert "not a pure sea density" in text
    assert "Polarization" in text
    assert "Current operator" in text
    assert "Parton" in text
    assert "Hadron" in text
    assert "`barpsi_at_0`" in text
    assert "`paired_flow`; partner `rn_reverse`" in text
    assert r"\Delta P_z=1" in text
    assert r"h_{fi}^{\rm mid}(-z)=h_{if}^{\rm mid}(z)^*" in text
    assert r"\lambda=\bar P^z z_{\rm GeV^{-1}}" in text


def test_da_fourier_stage_report_documents_symmetry_projection(tmp_path: Path) -> None:
    path = tmp_path / "ft_report.md"
    write_fourier_stage_report(
        jobs=[
            {
                "job_id": "ft_da",
                "result": {
                    "target_observable": "da",
                    "observable": "meson_quasi_da",
                    "momentum_gev": 2.0,
                    "gfix": "GI",
                    "order": "NLA",
                    "component": "both",
                    "symmetry_guarantee": True,
                },
                "artifacts": {},
            }
        ],
        path=path,
    )

    text = path.read_text(encoding="utf-8")
    assert "before range selection and large-distance fitting" in text
    assert "e^{+izP_z/2}" in text
    assert "discards $\\operatorname{Im}h_{+}$" in text
    assert "e^{-izP_z/2}" in text
    assert "e^{+ix\\lambda}" in text
    assert "Polarization" not in text
    assert "Current operator" not in text


def test_matching_stage_report_lists_overlay_last_with_ensemble_description(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(matching_reporting, "_llm_kernel_formula", lambda *args, **kwargs: ("formula", False))
    monkeypatch.setattr(matching_reporting, "translate_markdown_report", lambda markdown, **kwargs: markdown)
    path = tmp_path / "matching_report.md"
    matching_reporting.write_matching_stage_report(
        jobs=[
            {
                "job_id": "mt_p4",
                "result": {
                    "kernel_id": "dummy_kernel",
                    "momentum_gev": 1.8,
                    "x_grid": [0.0, 1.0],
                    "quasi_mean": [1.0, 1.0],
                    "lightcone_mean": [1.0, 1.0],
                },
                "artifacts": {
                    "lightcone_artifact": "mt_p4.nc",
                    "matched_plot": "mt_p4.pdf",
                    "matching_overlay_mt_HISQa060_X": "mt_HISQa060_X.pdf",
                    "matching_overlay_image_mt_HISQa060_X": "mt_HISQa060_X.svg",
                },
            },
            {
                "job_id": "mt_p5",
                "result": {
                    "kernel_id": "dummy_kernel",
                    "momentum_gev": 2.25,
                    "x_grid": [0.0, 1.0],
                    "quasi_mean": [1.0, 1.0],
                    "lightcone_mean": [1.0, 1.0],
                },
                "artifacts": {"lightcone_artifact": "mt_p5.nc", "matched_plot": "mt_p5.pdf"},
            },
        ],
        path=path,
        report_language="ch",
        llm=matching_reporting.FormulaLlm(backend="cli", provider="codex"),
    )
    assert path.exists()
    text = path.with_name("matching_report_CN.md").read_text(encoding="utf-8")
    output = text.split("## Output Artifacts", 1)[1]

    assert output.rfind("mt_HISQa060_X.pdf") > output.rfind("mt_p5.pdf")
    assert output.rfind("mt_HISQa060_X.svg") > output.rfind("mt_HISQa060_X.pdf")
