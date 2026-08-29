from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.stages._reporting import StageReportRecord

_TEST_ENSEMBLE = EnsembleInfo("test", "a06", 0.06, 0.06, 64, 128, 0.13)


def _data(*, attrs=None, values=None) -> EnsembleData:
    values = values or [[0.8, 1.0], [0.9, 1.1]]
    return EnsembleData(_TEST_ENSEMBLE, "bootstrap", values, ["x"], {"x": [-0.2, 0.2]}, attrs=attrs or {})


def _correlator_lsqfit_params() -> dict:
    return {
        "analysis_method": "lsqfit",
        "component": "re",
        "nstate": [2],
        "fit_scope": ["3pt_ratio"],
        "fit_strategy": ["joint"],
        "fitting_form": "Breit",
        "pt2_windows": [{"tmin": 3, "tmax": 8}],
        "pt3_windows": [{"tsep_ls": [8], "tau_cut": 2}],
        "svdcut": 1e-6,
        "q_min": 0.05,
    }


def _correlator_dispersion_record(stage: Path, job_id: str, ensemble, momentum, energy_samples) -> StageReportRecord:
    output = EnsembleData(
        ensemble,
        "bootstrap",
        [[0.8, 0.7], [0.82, 0.72], [0.79, 0.69], [0.81, 0.71]],
        ["z"],
        {"z": [0, 1]},
        attrs={
            "momentum_gev": momentum,
            "sample_error_mode": "covariance",
            "resample_id": "shared",
        },
    )
    summary = {
        "result": "bare_matrix_element",
        "decisions": {"candidate_id": "matrix_001", "method": "joint"},
        "diagnostics": {
            "Q": 0.8,
            "chi2_dof": 0.9,
            "candidates": [],
            "sample_fit_quality": {
                "Q": [0.9, 0.4, 0.01],
                "chi2_dof": [0.5, 1.0, 2.0],
                "n_successful": 3,
                "n_failed": 0,
            },
            "dispersion_energy": {"z": 0.0, "energy_unit": "lattice", "E0_samples": energy_samples},
        },
        "artifacts": ["output.nc"],
    }
    return _record(stage, job_id, params=_correlator_lsqfit_params(), output=output, summary=summary)


def _record(stage: Path, job_id: str, *, params, output, summary, inputs=None) -> StageReportRecord:
    directory = stage / job_id
    directory.mkdir(parents=True)
    for artifact in summary["artifacts"]:
        path = directory / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return StageReportRecord(job_id, params, inputs or {}, output, summary, directory)


def test_stage_report_hook_receives_all_completed_jobs(tmp_path: Path) -> None:
    from lamet_agent.agent import _write_stage_report

    stage_root = tmp_path / "stages"
    stage_source = stage_root / "toy" / "reporting.py"
    stage_source.parent.mkdir(parents=True)
    stage_source.write_text(
        "from pathlib import Path\n"
        "def write_stage_report(*, records, artifact_directory):\n"
        "    if len(records) != 2:\n"
        "        raise ValueError('report hook ran before every job completed')\n"
        "    path = artifact_directory / 'report.md'\n"
        "    path.write_text(','.join(record.job_id for record in records), encoding='utf-8')\n"
        "    return path\n",
        encoding="utf-8",
    )
    artifact_directory = tmp_path / "runs" / "01_toy"
    output = _data()
    summary = {"result": "toy", "artifacts": []}
    records = [
        _record(artifact_directory, "one", params={}, output=output, summary=summary),
        _record(artifact_directory, "two", params={}, output=output, summary=summary),
    ]

    path = _write_stage_report("toy", records, stage_root=stage_root)

    assert path is not None
    assert path.read_text(encoding="utf-8") == "one,two"


def test_correlator_stage_report_contains_method_candidates_and_artifacts(tmp_path: Path) -> None:
    from lamet_agent.stages.correlator_analysis.reporting import write_stage_report

    stage = tmp_path / "01_correlator_analysis"
    output = EnsembleData(
        _TEST_ENSEMBLE,
        "bootstrap",
        [[0.8, 1.0], [0.9, 1.1]],
        ["z"],
        {"z": [0, 1]},
        attrs={"momentum_gev": 2.0},
    )
    params = {
        "analysis_method": "lsqfit",
        "component": "re",
        "nstate": [2],
        "fit_scope": ["3pt_ratio"],
        "fit_strategy": ["joint"],
        "fitting_form": "Breit",
        "pt2_windows": [{"tmin": 3, "tmax": 8}],
        "pt3_windows": [{"tsep_ls": [8], "tau_cut": 2}],
        "svdcut": 1e-6,
        "q_min": 0.05,
    }
    summary = {
        "result": "bare_matrix_element",
        "decisions": {"candidate_id": "matrix_001", "method": "joint"},
        "diagnostics": {
            "Q": 0.8,
            "chi2_dof": 0.9,
            "candidates": [
                {
                    "candidate_id": "matrix_001",
                    "method": "joint",
                    "window": {"tmin": 3, "tmax": 8},
                    "nstate": 2,
                    "Q": 0.8,
                    "chi2_dof": 0.9,
                    "quality_passed": True,
                    "numerical_failure": False,
                }
            ],
        },
        "artifacts": ["output.nc"],
    }
    path = write_stage_report(
        records=(_record(stage, "ca", params=params, output=output, summary=summary),), artifact_directory=stage
    )
    text = path.read_text(encoding="utf-8")
    assert "Correlator Analysis Stage Report" in text
    assert "matrix_001" in text
    assert "Selection Policy" in text
    assert "Per-tuning-z Fit Summary" in text
    assert "Field Definitions" in text
    assert "fallback_no_q_passing" in text
    assert "[output.nc](ca/output.nc)" in text


def test_correlator_fit_artifacts_write_logs_and_pdf_only(tmp_path: Path) -> None:
    from lamet_agent.stages.correlator_analysis._diagnostics import write_fit_artifacts

    def plot_payload(z_value: int) -> dict[str, object]:
        plots = []
        for component in ("re", "im"):
            plots.append(
                {
                    "kind": "pt3_ratio",
                    "component": component,
                    "series": [
                        {
                            "label": r"$t_{\mathrm{sep}}=8$",
                            "x": [2.0, 3.0],
                            "y": [0.8, 0.82],
                            "yerr": [0.03, 0.03],
                            "fit_x": [1.5, 3.5],
                            "fit_mean": [0.81, 0.81],
                            "fit_sdev": [0.02, 0.02],
                        }
                    ],
                    "plateau_mean": 0.81,
                    "plateau_sdev": 0.02,
                }
            )
        return {"z": z_value, "plots": plots}

    fits = []
    for z_value in (0, 1):
        fits.append(
            {
                "z": z_value,
                "Q": 0.8,
                "chi2": 4.0,
                "dof": 8.0,
                "chi2_dof": 0.5,
                "logGBF": 10.0,
                "n_failed_samples": 0,
                "sample_diagnostics": [
                    {"sample": 0, "Q": 0.8, "chi2": 4.0, "dof": 8.0, "chi2_dof": 0.5, "logGBF": 10.0}
                ],
                "sample0_plot": plot_payload(z_value),
                "E0_samples": [0.2, 0.21],
            }
        )
    selected = {
        "id": "matrix_001",
        "method": "joint",
        "fit_scope": "3pt_ratio",
        "window": {"tmin": 3, "tmax": 8, "tau_min": 2},
        "nstate": 1,
        "prior_width": 1.0,
    }
    result = write_fit_artifacts(
        job_id="ca_p0",
        selected=selected,
        candidates=[selected],
        preflight_fit={"fits": fits},
        application_fit={"fits": fits, "sample_failures": []},
        application_rejections=[],
        artifact_directory=tmp_path,
        component="re",
        q_min=0.05,
    )
    assert len(result.artifacts) == 4
    assert len(list((tmp_path / "fit_logs" / "plots").glob("*.pdf"))) == 2
    assert not list((tmp_path / "fit_logs" / "plots").glob("*.svg"))
    assert "Good sample=0" in (tmp_path / "fit_logs" / "ca_p0_joint_3pt_ratio_samples.log").read_text()
    assert result.sample_fit_quality["Q"] == [0.8, 0.8]
    assert result.dispersion_energy["z"] == 0.0
    assert "sample_diagnostics" not in result.application_fit["fits"][0]
    assert "sample0_plot" not in result.application_fit["fits"][0]


def test_correlator_sample0_plot_restores_legacy_physical_labels(monkeypatch, tmp_path: Path) -> None:
    import lamet_agent.stages.correlator_analysis._diagnostics as diagnostics

    configured = {}
    monkeypatch.setattr(diagnostics, "start_plot", lambda: None)
    monkeypatch.setattr(diagnostics, "errorline", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnostics, "errorband", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnostics, "configure_plot", lambda **kwargs: configured.update(kwargs))
    monkeypatch.setattr(diagnostics, "save_figure", lambda *args: None)
    payload = {
        "kind": "pt3_ratio",
        "component": "re",
        "z": 0.0,
        "series": [
            {
                "label": r"$t_{\mathrm{sep}}=8\,a$",
                "x": [-2.0, -1.0, 0.0, 1.0, 2.0],
                "y": [0.80, 0.81, 0.82, 0.81, 0.80],
                "yerr": [0.02] * 5,
                "fit_x": [-2.5, 2.5],
                "fit_mean": [0.81, 0.81],
                "fit_sdev": [0.01, 0.01],
            }
        ],
        "plateau_mean": 0.81,
        "plateau_sdev": 0.01,
    }

    diagnostics._write_sample0_plot(tmp_path / "sample0.pdf", payload, job_id="ca_p0")

    assert configured["xlabel"] == diagnostics.TAU_CENTER_LABEL
    assert configured["ylabel"] == diagnostics.RATIO_REAL_LABEL
    low, high = configured["ylim"]
    data_min = 0.80 - 0.02
    data_max = 0.82 + 0.02
    assert low < data_min < data_max < high
    assert (data_max - data_min) / (high - low) > 0.45
    assert (high - data_max) > (data_min - low)
    assert diagnostics._sample0_plot_labels("fh", "im") == (
        diagnostics.TSEP_LABEL,
        diagnostics.FH_IMAG_LABEL,
    )
    assert diagnostics._sample0_plot_labels("qda_ratio", "re") == (
        diagnostics.QDA_TIME_LABEL,
        diagnostics.QDA_RATIO_REAL_LABEL,
    )


def test_correlator_sample0_plot_draws_bands_only_on_fit_region(monkeypatch, tmp_path: Path) -> None:
    import lamet_agent.stages.correlator_analysis._diagnostics as diagnostics

    bands = []
    monkeypatch.setattr(diagnostics, "start_plot", lambda: None)
    monkeypatch.setattr(diagnostics, "errorline", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        diagnostics, "errorband", lambda x, values, **kwargs: bands.append((list(x), kwargs.get("label")))
    )
    monkeypatch.setattr(diagnostics, "configure_plot", lambda **kwargs: None)
    monkeypatch.setattr(diagnostics, "save_figure", lambda *args: None)
    payload = {
        "kind": "qda_ratio",
        "component": "im",
        "z": 5.0,
        "series": [
            {
                "label": "qDA ratio",
                "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "y": [0.1] * 8,
                "yerr": [0.02] * 8,
                "fit_x": [3.5, 6.5],
                "fit_mean": [0.81, 0.81],
                "fit_sdev": [0.01, 0.01],
            }
        ],
        "plateau_mean": 0.81,
        "plateau_sdev": 0.01,
    }

    diagnostics._write_sample0_plot(tmp_path / "sample0.pdf", payload, job_id="ca_a06m130_pz6")

    assert bands == [([3.5, 6.5], None), ([3.5, 6.5], "Sample-0 fit matrix element")]


def test_correlator_stage_report_writes_quality_and_physical_dispersion_plots(tmp_path: Path) -> None:
    from lamet_agent.plotting import COLOR_CYCLE
    from lamet_agent.stages.correlator_analysis.reporting import write_stage_report

    stage = tmp_path / "01_correlator_analysis"
    ensemble = EnsembleInfo("HISQ", "HISQa060_X", 0.06, 0.06, 48, 64, 0.3)
    records = [
        _correlator_dispersion_record(stage, "ca_p0", ensemble, 0.0, [0.090, 0.092, 0.089, 0.091]),
        _correlator_dispersion_record(stage, "ca_p5", ensemble, 2.15, [0.690, 0.700, 0.680, 0.695]),
    ]
    report = write_stage_report(records=tuple(records), artifact_directory=stage)
    text = report.read_text(encoding="utf-8")
    assert "## Sample Fit Quality" in text
    assert "sample_fit_quality_Q" not in text
    for stem in ("sample_fit_quality_chi2", "dispersion_relation"):
        assert (stage / "plots" / f"{stem}.pdf").is_file()
        assert (stage / "plots" / f"{stem}.svg").is_file()
    chi2_svg = (stage / "plots" / "sample_fit_quality_chi2.svg").read_text(encoding="utf-8")
    assert r"$\chi^2/\mathrm{d.o.f.}$" in chi2_svg
    assert "Per-sample fit" not in chi2_svg
    assert COLOR_CYCLE[3].lower() in chi2_svg.lower()
    assert "fill" in chi2_svg.lower()
    dispersion_svg = (stage / "plots" / "dispersion_relation.svg").read_text(encoding="utf-8")
    assert "Dispersion relation" in dispersion_svg
    assert "E^2=p^2" in dispersion_svg
    assert "mathrm{GeV}" in dispersion_svg
    assert "FillBetween" not in dispersion_svg
    assert "2 momenta and the dispersion model has 3 parameters" in text
    overview_svg = (stage / "plots" / "correlator_overview.svg").read_text(encoding="utf-8")
    assert "bare matrix element" in overview_svg
    assert "bare_matrix_element" not in overview_svg


def test_correlator_dispersion_fit_requires_more_momenta_than_parameters(tmp_path: Path, monkeypatch) -> None:
    import gvar as gv

    import lamet_agent.plotting as plotting
    from lamet_agent.stages.correlator_analysis.reporting import write_stage_report

    band_widths: list[float] = []
    original_errorband = plotting.errorband

    def capture_errorband(x, values, **kwargs):
        band_widths.append(float(np.max(gv.sdev(values))))
        return original_errorband(x, values, **kwargs)

    monkeypatch.setattr(plotting, "errorband", capture_errorband)
    stage = tmp_path / "01_correlator_analysis"
    ensemble = EnsembleInfo("HISQ", "HISQa060_X", 0.06, 0.06, 48, 64, 0.3)
    records = [
        _correlator_dispersion_record(stage, job_id, ensemble, momentum, samples)
        for job_id, momentum, samples in (
            ("ca_p0", 0.0, [0.090, 0.092, 0.089, 0.091]),
            ("ca_p2", 0.86, [0.280, 0.285, 0.275, 0.282]),
            ("ca_p4", 1.72, [0.530, 0.540, 0.520, 0.535]),
            ("ca_p5", 2.15, [0.690, 0.700, 0.680, 0.695]),
        )
    ]
    report = write_stage_report(records=tuple(records), artifact_directory=stage)
    text = report.read_text(encoding="utf-8")
    svg = (stage / "plots" / "dispersion_relation.svg").read_text(encoding="utf-8")
    assert "FillBetween" in svg
    assert "fit band was omitted" not in text
    assert band_widths
    assert all(width < 1.0 for width in band_widths)


def test_renormalization_stage_report_contains_scheme_formula(tmp_path: Path) -> None:
    from lamet_agent.stages.renormalization.reporting import write_stage_report

    stage = tmp_path / "02_renormalization"
    output = EnsembleData(
        _TEST_ENSEMBLE,
        "bootstrap",
        [[1.0, 0.8], [1.0, 0.9]],
        ["z"],
        {"z": [0.0, 0.1]},
        attrs={"coord_unit": "fm"},
    )
    params = {
        "strategy": "external_denominator",
        "type": "apply",
        "normalization": True,
        "scheme": "hybrid",
        "zs_fm": 0.1,
        "m0_gev": 0.0,
        "delta_m_gev": 0.2,
    }
    summary = {
        "result": "renormalized_matrix_element",
        "decisions": {},
        "diagnostics": {"dims": ["z"]},
        "artifacts": ["output.nc"],
    }
    path = write_stage_report(
        records=(_record(stage, "rn", params=params, output=output, summary=summary),), artifact_directory=stage
    )
    text = path.read_text(encoding="utf-8")
    assert "external_denominator" in text
    assert "h_s^R(z)" in text
    assert "Coverage and Statistical Semantics" in text
    assert "Field Definitions" in text


def test_fourier_stage_report_contains_tail_and_selection(tmp_path: Path) -> None:
    from lamet_agent.stages.fourier_transform.reporting import write_stage_report

    stage = tmp_path / "03_fourier_transform"
    output = EnsembleData(
        _TEST_ENSEMBLE,
        "bootstrap",
        [[0.8 + 0.1j, 1.0 + 0.2j], [0.9 + 0.12j, 1.1 + 0.18j]],
        ["x"],
        {"x": [-0.2, 0.2]},
        attrs={
            "momentum_gev": 1.72,
            "gfix": "GI",
            "parton": "quark",
            "sector": "full",
            "component": "both",
            "output_scale": 1.0,
            "target_observable": "da",
            "selected_range": json.dumps([0.2, 0.4]),
            "phase_transfer_da": "true",
            "sample_error_mode": "covariance",
        },
    )
    params = {
        "quasi_y_ls": [-0.2, 0.2],
        "zmin_fm": [0.2],
        "zmax_fm": [0.4],
        "zmax_ext_fm": 1.0,
        "smooth": "linear",
        "tail_window_step_offset": 0,
        "scheme_scan": {
            "sector": "full",
            "order": ["LA", "NLA"],
            "posterior_prior_error_scale": [2.0],
            "model_average": True,
        },
    }
    diagnostics = {
        "selected_range_label": "zmin_0p2_zmax_0p4",
        "selected_fit_model_labels": ["gi_nla_NLA"],
        "fit_model_weights": [1.0],
        "selected_Q": 0.8,
        "selected_chi2_dof": 0.9,
        "range_candidate_count": 1,
        "model_candidate_count": 2,
    }
    summary = {
        "result": "quasi_distribution",
        "decisions": {},
        "diagnostics": diagnostics,
        "artifacts": ["output.nc", "diagnostics/candidates.json", "diagnostics/ranges.json"],
    }
    record = _record(stage, "ft", params=params, output=output, summary=summary)
    (record.artifact_directory / "diagnostics" / "candidates.json").write_text(
        json.dumps(
            [
                {
                    "label": "gi_nla_NLA",
                    "model_id": "gi_nla",
                    "z_min_fm": 0.2,
                    "z_max_fm": 0.4,
                    "order": "NLA",
                    "prior_width": 2.0,
                    "parameter_mean": {"A2": 1.0, "Lambda": 0.3},
                    "parameter_sdev": {"A2": 0.1, "Lambda": 0.05},
                    "Q": 0.8,
                    "chi2_dof": 0.9,
                    "model_weight": 1.0,
                    "selected": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    (record.artifact_directory / "diagnostics" / "ranges.json").write_text(
        json.dumps(
            [
                {
                    "model_id": "gi_nla",
                    "z_min_fm": 0.2,
                    "z_max_fm": 0.4,
                    "fit_success": True,
                    "Q": 0.8,
                    "chi2_dof": 0.9,
                    "logGBF": 2.0,
                    "selected": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    path = write_stage_report(records=(record,), artifact_directory=stage)
    text = path.read_text(encoding="utf-8")
    assert "Large-Distance Extrapolation" in text
    assert "gi_nla_NLA" in text
    assert "Range and Fit-model Candidates" in text
    assert "Projection and Field Definitions" in text
    assert "Tail posterior parameters" in text
    assert "Range-selection fits" in text
    real_svg = (stage / "plots" / "fourier_overview_real.svg").read_text(encoding="utf-8")
    imag_svg = (stage / "plots" / "fourier_overview_imag.svg").read_text(encoding="utf-8")
    assert "FillBetweenPolyCollection" in real_svg
    assert "FillBetweenPolyCollection" in imag_svg
    assert r"$\mathrm{Re}\,\tilde q(x)$" in real_svg
    assert r"$\mathrm{Im}\,\tilde q(x)$" in imag_svg
    assert r"$x$" in real_svg
    assert r"$P_z=1.72\,\mathrm{GeV}$" in real_svg
    assert r"$P_z=1.72\,\mathrm{GeV}$" in imag_svg
    assert r"$a=0.06\,\mathrm{fm}$" in real_svg
    assert r"$a=0.06\,\mathrm{fm}$" in imag_svg
    assert "quasi distribution" not in real_svg
    assert "ft" not in real_svg


def test_overlay_groups_are_even_and_capped() -> None:
    from lamet_agent.stages._reporting import overlay_groups

    assert overlay_groups(list(range(6))) == [list(range(6))]
    assert overlay_groups(list(range(7))) == [list(range(4)), list(range(4, 7))]
    assert overlay_groups(list(range(9))) == [list(range(5)), list(range(5, 9))]
    assert overlay_groups(list(range(20))) == [
        list(range(5)),
        list(range(5, 10)),
        list(range(10, 15)),
        list(range(15, 20)),
    ]
    assert overlay_groups(list(range(27))) == [
        list(range(6)),
        list(range(6, 12)),
        list(range(12, 17)),
        list(range(17, 22)),
        list(range(22, 27)),
    ]
    assert all(len(group) <= 6 for group in overlay_groups(list(range(27))))


def test_stage_overlay_splits_crowded_series_into_even_figures(tmp_path: Path) -> None:
    from lamet_agent.stages._reporting import stage_overlay_lines

    records = tuple(
        StageReportRecord(
            f"job_{index}",
            {},
            {},
            _data(attrs={"momentum_gev": 1.0 + 0.05 * index, "sample_error_mode": "covariance"}),
            {"result": "overlay", "artifacts": ["output.nc"]},
            tmp_path / f"job_{index}",
        )
        for index in range(9)
    )

    lines = stage_overlay_lines(
        records,
        tmp_path,
        coordinate="x",
        stem="renormalization_overview",
        xlabel=r"$x$",
        ylabel="overlay",
        band=True,
    )
    text = "".join(lines)
    first = (tmp_path / "plots" / "renormalization_overview_1.svg").read_text(encoding="utf-8")
    second = (tmp_path / "plots" / "renormalization_overview_2.svg").read_text(encoding="utf-8")

    assert "renormalization_overview_1.svg" in text
    assert "renormalization_overview_2.svg" in text
    assert "renormalization_overview_1.pdf" in text
    assert not (tmp_path / "plots" / "renormalization_overview.svg").exists()
    assert "job_0" in first and "job_4" in first and "job_5" not in first
    assert "job_5" in second and "job_8" in second and "job_0" not in second
    assert r"$x$" in first
    assert "overlay" in first


def test_matching_stage_report_embeds_shipped_kernel_document(tmp_path: Path) -> None:
    from lamet_agent.stages.perturbative_matching.reporting import write_stage_report

    stage = tmp_path / "04_perturbative_matching"
    quasi = _data(attrs={"momentum_gev": 2.0, "output_scale": 1.0, "component": "both"})
    quasi = EnsembleData(
        None,
        "bootstrap",
        [np.asarray(sample) + 0.2j * np.asarray(sample) for sample in quasi.values],
        ["x"],
        {"x": quasi.coords["x"]},
        attrs={"momentum_gev": 2.0, "output_scale": 1.0, "component": "both"},
    )
    output = _data(attrs={"momentum_gev": 2.0, "output_scale": 1.0}, values=[[0.7, 0.9], [0.8, 1.0]])
    params = {"kernel_id": "da_gi_gzg5_ratio_nlo", "scheme": "ratio", "mu": 2.0, "kernel_parameters": {}}
    summary = {
        "result": "matched_distribution",
        "decisions": {},
        "diagnostics": {"matrix_shape": [2, 2]},
        "artifacts": ["output.nc"],
    }
    path = write_stage_report(
        records=(_record(stage, "mt", params=params, inputs={"quasi": quasi}, output=output, summary=summary),),
        artifact_directory=stage,
    )
    text = path.read_text(encoding="utf-8")
    assert "da_gi_gzg5_ratio_nlo" in text
    assert "Matching" in text
    assert "Relative change" in text
    assert "Kernel-id and Field Definitions" in text
    assert "Matching Scheme" in text
    assert "Matching Formula and Literature Consistency Check" in text
    assert "V_{qq,p}" in text
    assert "No discrepancies found" in text
    overview_svg = (stage / "plots" / "matching_overview.svg").read_text(encoding="utf-8")
    assert "FillBetweenPolyCollection" in overview_svg
    assert r"$x$" in overview_svg


def test_extrapolation_stage_report_contains_model_and_budget(tmp_path: Path) -> None:
    from lamet_agent.stages.extrapolation.reporting import write_stage_report

    stage = tmp_path / "05_extrapolation"
    provenance = {
        "momentum_gev": 2.0,
        "kernel_id": "GI_gzg5_DA_ratio_NLO",
    }
    source = _data(attrs=provenance)
    output = _data(
        attrs={
            "extrapolation_terms": "a,inv_p2",
            "x_independent_terms": json.dumps(["a"]),
            "x_dependent_terms": json.dumps(["inv_p2"]),
        }
    )
    fit_params = {
        "x_independent_terms": ["a"],
        "x_dependent_terms": ["inv_p2"],
        "x_covariance": False,
        "priors": {"mean": 0.0, "sdev": 3.0},
        "posterior_prior_error_scale": 3.0,
        "pdep_gev": [1.5, 2.0, 2.5],
    }
    params = {"operation": "fit", **fit_params}
    momentum_dependence = {
        f"{momentum:g}": {
            "momentum_gev": momentum,
            "mean": [0.8 + 0.1 / momentum**2, 1.0 + 0.2 / momentum**2],
            "sdev": [0.05, 0.05],
        }
        for momentum in fit_params["pdep_gev"]
    }
    comparison = {
        "candidates": [
            {
                "Q": 0.8,
                "chi2_dof": 0.9,
                "n_failed_samples": 0,
                "parameter_mean": {"h0": [0.8, 1.0], "a": 0.2, "inv_p2": [0.1, 0.2]},
                "parameter_sdev": {"h0": [0.05, 0.05], "a": 0.03, "inv_p2": [0.02, 0.02]},
                "momentum_dependence": momentum_dependence,
            }
        ]
    }
    fit_summary = {
        "result": "physical_distribution",
        "decisions": {},
        "diagnostics": comparison,
        "artifacts": ["output.nc"],
    }
    budget_summary = {
        "result": "systematics_budget",
        "decisions": {"systematics_groups": {"main": 0}},
        "diagnostics": {"sources": ["lamet_scale"], "point_count": 2},
        "artifacts": ["output.nc", "diagnostics/systematics_budget.json"],
    }
    fit_record = _record(
        stage, "fit", params=params, inputs={"distributions": [source]}, output=output, summary=fit_summary
    )
    budget_record = _record(
        stage,
        "budget",
        params={
            "operation": "systematics_budget",
            "systematics_groups": {"main": 0},
            "systematics_prescription": "variant_envelope_quadrature",
        },
        output=output,
        summary=budget_summary,
    )
    (budget_record.artifact_directory / "diagnostics" / "systematics_budget.json").write_text(
        json.dumps({"lamet_scale": [0.1, 0.2], "total_systematic_error": [0.2, 0.3], "total_error": [0.3, 0.4]}),
        encoding="utf-8",
    )
    records = (fit_record, budget_record)
    path = write_stage_report(records=records, artifact_directory=stage)
    text = path.read_text(encoding="utf-8")
    assert "c_{a}" in text
    assert "Systematics Budget" in text
    assert "Input Coverage" in text
    assert "maximum absolute size" in text
    assert "Momentum Dependence" in text
    assert "pdep_gev" in text
    overview_svg = (stage / "plots" / "extrapolation_overview.svg").read_text(encoding="utf-8")
    assert "FillBetweenPolyCollection" in overview_svg
    assert "Errorbar" not in overview_svg
    assert r"$x$" in overview_svg
