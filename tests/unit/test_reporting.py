from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.stages._reporting import StageReportRecord


def _data(*, attrs=None, values=None) -> EnsembleData:
    values = values or [[0.8, 1.0], [0.9, 1.1]]
    return EnsembleData(None, "bootstrap", values, ["x"], {"x": [-0.2, 0.2]}, attrs=attrs or {})


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
        None, "bootstrap", [[0.8, 1.0], [0.9, 1.1]], ["z"], {"z": [0, 1]}, attrs={"momentum_gev": 2.0}
    )
    params = {
        "analysis_method": "lsqfit",
        "component": "re",
        "nstate": [2],
        "lsqfit": {
            "fit_scope": ["3pt_ratio"],
            "fit_strategy": ["joint"],
            "fitting_form": "Breit",
            "pt2_windows": [{"tmin": 3, "tmax": 8}],
            "pt3_windows": [{"tsep_ls": [8], "tau_cut": 2}],
            "svdcut": 1e-6,
            "q_min": 0.05,
        },
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
                    "window": {"t_min": 3, "t_max": 8},
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
    assert "[output.nc](ca/output.nc)" in text


def test_renormalization_stage_report_contains_scheme_formula(tmp_path: Path) -> None:
    from lamet_agent.stages.renormalization.reporting import write_stage_report

    stage = tmp_path / "02_renormalization"
    output = EnsembleData(
        None, "bootstrap", [[1.0, 0.8], [1.0, 0.9]], ["z"], {"z": [0.0, 0.1]}, attrs={"coord_unit": "fm"}
    )
    params = {
        "strategy": "external_denominator",
        "normalization": True,
        "external_denominator": {"scheme": "hybrid", "hybrid": {"zs_fm": 0.1, "m0_gev": 0.0, "delta_m_gev": 0.2}},
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
    output = _data(
        attrs={
            "momentum_gev": 2.0,
            "gfix": "GI",
            "parton": "quark",
            "sector": "full",
            "component": "both",
            "output_scale": 1.0,
            "target_observable": "da",
            "selected_range": json.dumps([0.2, 0.4]),
            "phase_transfer_da": "true",
        }
    )
    params = {
        "quasi_y_ls": [-0.2, 0.2],
        "zmin_fm": [0.2],
        "zmax_fm": [0.4],
        "zmax_ext_fm": 1.0,
        "smooth": "linear",
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
    params = {"kernel_id": "GI_gzg5_DA_ratio_NLO", "scheme": "ratio", "mu": 2.0, "kernel_parameters": {}}
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
    assert "GI_gzg5_DA_ratio_NLO" in text
    assert "Matching" in text
    assert "Relative change" in text
    assert "Kernel-id and Field Definitions" in text
    assert "Matching Scheme" in text
    assert "Matching Formula and Literature Consistency Check" in text
    assert "V_{qq,p}" in text
    assert "No discrepancies found" in text


def test_extrapolation_stage_report_contains_model_and_budget(tmp_path: Path) -> None:
    from lamet_agent.stages.extrapolation.reporting import write_stage_report

    stage = tmp_path / "05_extrapolation"
    provenance = {
        "ensemble": "a06",
        "lattice_spacing_fm": 0.06,
        "momentum_gev": 2.0,
        "m_pi": 0.13,
        "kernel_id": "GI_gzg5_DA_ratio_NLO",
    }
    source = _data(attrs=provenance)
    output = _data(attrs={"extrapolation_terms": "a,inv_p2", "x_dependence": json.dumps({"a": False, "inv_p2": True})})
    fit_params = {
        "required_terms": ["a", "inv_p2"],
        "allowed_terms": [],
        "x_dependence": {"a": False, "inv_p2": True},
        "priors": {"mean": 0.0, "sdev": 3.0},
        "posterior_prior_error_scale": 3.0,
        "pdep_gev": [1.5, 2.0, 2.5],
    }
    params = {"operation": "fit", "fit": fit_params}
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
            "systematics_budget": {
                "systematics_groups": {"main": 0},
                "systematics_prescription": "variant_envelope_quadrature",
            },
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
