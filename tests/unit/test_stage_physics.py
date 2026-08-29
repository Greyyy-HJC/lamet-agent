"""Focused numerical checks for stage-owned physics.

Purpose: exercise multi-state spectra, coordinate-aware matrix ratios, and the
self-renormalization factor fit. Inputs are deterministic toy arrays; outputs
are recovered physical parameters. Example: ``pytest tests/unit/test_stage_physics.py``.
"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import tokenize

import numpy as np
import pytest

from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.agent import ToolContext
from lamet_agent.parallel import FitNumericalError
from lamet_agent.kernels import list_kernel_ids, load_kernel, load_kernel_document, load_renormalization_kernel
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.stages.correlator_analysis.physics import fit_spectrum_samples, matrix_element_samples
from lamet_agent.stages.correlator_analysis.physics import fit_matrix_element_samples, matrix_element_prior
from lamet_agent.parallel.lanczos import (
    _analyze_threept,
    _analyze_twopt,
    _median_threept_matrix,
    _median_twopt_energies,
    analyze_prepared_lanczos,
    prepare_lanczos_data,
)
from lamet_agent.stages.extrapolation.physics import basis_terms, fit_candidate
from lamet_agent.parallel import fourier_transform
from lamet_agent.stages.fourier_transform.physics import fit_tail_parameters, scan_fourier_transform, tail_model_values
from lamet_agent.stages.perturbative_matching.physics import inspect_callable
from lamet_agent.stages.renormalization.physics import (
    fit_factor,
    load_data as load_renormalization_data,
    log_m,
)
from lamet_agent.stages.fourier_transform.physics import fourier_transform as stage_fourier_transform


def _ensemble(spacing: float, identifier: str = "test", *, L_s: int = 64, m_pi: float = 0.14) -> EnsembleInfo:
    return EnsembleInfo("test", identifier, spacing, spacing, L_s, 2 * L_s, m_pi)


def test_matrix_ratio_uses_declared_tsep_and_tau_coordinates() -> None:
    t = np.arange(1.0, 7.0)
    tsep = np.array([2.0, 3.0, 4.0])
    tau = np.array([1.0, 2.0, 3.0])
    z = np.array([0.0, 1.0])
    c2_values = [np.exp(-0.3 * t), 1.1 * np.exp(-0.3 * t)]
    c3_values = [
        np.stack([np.full((tau.size, z.size), 0.7 * np.exp(-0.3 * ts)) for ts in tsep]),
        np.stack([np.full((tau.size, z.size), 0.7 * 1.1 * np.exp(-0.3 * ts)) for ts in tsep]),
    ]
    c2 = EnsembleData(None, "bootstrap", c2_values, ["t"], {"t": t.tolist()}, attrs={"correlator_type": "two_point"})
    c3 = EnsembleData(
        None,
        "bootstrap",
        c3_values,
        ["tsep", "tau", "z"],
        {"tsep": tsep.tolist(), "tau": tau.tolist(), "z": z.tolist()},
        attrs={"correlator_type": "three_point"},
    )
    values, coordinates, _ = matrix_element_samples({"c2": c2, "c3": c3}, method="ratio", tmin=2, tmax=4, tau_min=1)
    assert coordinates == [0.0, 1.0]
    assert np.allclose(values, 0.7, atol=1e-12)


def test_matrix_ratio_rejects_a_missing_exact_two_point_denominator() -> None:
    c2 = EnsembleData(
        None, "bootstrap", [np.ones(2), np.ones(2)], ["t"], {"t": [1.0, 2.0]}, attrs={"correlator_type": "two_point"}
    )
    c3 = EnsembleData(
        None,
        "bootstrap",
        [np.ones((1, 1)), np.ones((1, 1))],
        ["tsep", "tau"],
        {"tsep": [3.0], "tau": [1.0]},
        attrs={"correlator_type": "three_point"},
    )
    with pytest.raises(ValueError, match="exactly one entry"):
        matrix_element_samples({"c2": c2, "c3": c3}, method="ratio", tmin=1, tmax=3, tau_min=1)


def _exact_lanczos_correlators(
    n_configurations: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energies = np.asarray([0.25, 0.7])
    overlaps_squared = np.asarray([1.0, 0.3])
    transfer_values = np.exp(-energies)
    times = np.arange(8)
    c2 = np.sum(overlaps_squared[:, None] * np.exp(-energies[:, None] * times), axis=0)
    current = np.asarray([[0.8, 0.1], [0.1, 0.4]])
    overlaps = np.sqrt(overlaps_squared)
    c3 = np.empty((3, 3), dtype=float)
    for sigma in range(3):
        for tau in range(3):
            c3[sigma, tau] = np.sum(
                overlaps[:, None]
                * transfer_values[:, None] ** sigma
                * current
                * overlaps[None, :]
                * transfer_values[None, :] ** tau
            )
    return (
        np.tile(c2, (n_configurations, 1)),
        np.tile(c3, (n_configurations, 1, 1)),
        current,
    )


def test_migrated_lanczos_recovers_exact_spectrum_and_matrix() -> None:
    c2, c3, current = _exact_lanczos_correlators()

    spectra = _analyze_twopt(c2, 6, seed=0, max_iterations=3)
    energies = _median_twopt_energies(spectra, max_states=2)
    matrices = _analyze_threept(c3, c2, c2, 6, seed=0, max_iterations=2)
    matrix = _median_threept_matrix(matrices, iteration=2, max_states=2)

    assert energies[-1] == pytest.approx([0.25, 0.7])
    assert matrix == pytest.approx(current)


def test_lanczos_uses_raw_nested_resampling_and_standard_tsep_conversion(
    tmp_path: Path,
) -> None:
    n_configurations = 6
    energies = np.asarray([0.25, 0.7])
    overlaps_squared = np.asarray([1.0, 0.3])
    overlaps = np.sqrt(overlaps_squared)
    current = np.asarray([[0.8 + 0.25j, 0.1 - 0.05j], [0.1 + 0.02j, 0.4 - 0.1j]])
    times = np.arange(14)
    c2_values = np.tile(
        np.sum(
            overlaps_squared[:, None] * np.exp(-energies[:, None] * times),
            axis=0,
        ),
        (n_configurations, 1),
    )
    tseps = [4, 6, 8, 12]
    taus = list(range(13))
    c3_values = np.zeros((n_configurations, len(tseps), len(taus), 1), dtype=complex)
    for tsep_index, tsep in enumerate(tseps):
        for tau in range(tsep + 1):
            sigma = tsep - tau
            value = np.sum(
                overlaps[:, None]
                * np.exp(-energies[:, None] * sigma)
                * current
                * overlaps[None, :]
                * np.exp(-energies[None, :] * tau)
            )
            c3_values[:, tsep_index, tau, 0] = value
    momentum = "[0, 0, 0]"
    c2 = EnsembleData(
        None,
        "raw",
        [sample for sample in c2_values],
        ["t"],
        {"t": times.tolist()},
        attrs={
            "correlator_type": "two_point",
            "source_momentum": momentum,
            "sink_momentum": momentum,
        },
        name="c2",
    )
    c3 = EnsembleData(
        None,
        "raw",
        [sample for sample in c3_values],
        ["tsep", "tau", "z"],
        {"tsep": tseps, "tau": taus, "z": [0]},
        attrs={
            "correlator_type": "three_point",
            "source_momentum": momentum,
            "sink_momentum": momentum,
        },
        name="c3",
    )

    prepared = prepare_lanczos_data({"c2": c2, "c3": c3}, scope="3pt_matrix")
    result = analyze_prepared_lanczos(
        prepared,
        components="both",
        max_states=2,
        resampling="jackknife",
        bootstrap_samples=None,
        bin_size=1,
        inner_samples=4,
        precision=0,
        seed=0,
        workers=1,
    )

    inspection = prepared["inspection"]
    assert inspection["lanczos_t0"] == 2
    assert inspection["lanczos_time_step"] == 2
    assert inspection["sampling_plan"]["selected_tseps"] == [4, 6, 8]
    assert inspection["point_usage"]["used_per_z"] == 4
    assert inspection["point_usage"]["discarded_per_z"] == 30
    assert result["values"][:, 0] == pytest.approx(np.full(n_configurations, current[0, 0]))

    from lamet_agent.stages.correlator_analysis._lanczos_inspection import (
        run as inspect_lanczos,
    )
    from lamet_agent.stages.correlator_analysis._lanczos import (
        run as run_lanczos,
    )

    params = {
        "analysis_method": "lanczos",
        "component": "both",
        "nstate": [2],
        "scope": "3pt_matrix",
        "inner_samples": 4,
        "precision": 0,
    }
    context = ToolContext(
        {
            "metadata": {
                "workers": 1,
                "random_seed": 0,
                "sample_error_mode": "covariance",
                "target_observable": "pdf",
                "resample_mode": "jackknife",
                "bin_size": 1,
            }
        },
        tmp_path / "manifest.json",
        "correlator_analysis",
        "lanczos",
        params,
        {},
        {},
        {"raw_correlators": {"c2": c2, "c3": c3}},
        tmp_path,
        np.random.default_rng(1),
    )
    with pytest.warns(UserWarning, match="30 points are discarded"):
        inspect_lanczos(context)
    observation = run_lanczos(context)

    assert context.output.values[:, 0] == pytest.approx(np.full(n_configurations, current[0, 0]))
    assert observation["summary"] == "published bare_matrix_element"
    assert (tmp_path / "output.nc").is_file()
    assert (tmp_path / "diagnostics" / "state_matrices.nc").is_file()
    assert (tmp_path / "plots" / "result.pdf").is_file()

    spectrum_dir = tmp_path / "spectrum"
    spectrum_dir.mkdir()
    spectrum_params = {
        **params,
        "component": "re",
        "scope": "2pt_spectrum",
        "inner_samples": 4,
        "precision": 0,
    }
    spectrum_context = ToolContext(
        context.manifest,
        tmp_path / "manifest.json",
        "correlator_analysis",
        "lanczos_spectrum",
        spectrum_params,
        {},
        {},
        {"raw_correlators": {"c2": c2}},
        spectrum_dir,
        np.random.default_rng(2),
    )
    inspect_lanczos(spectrum_context)
    spectrum_observation = run_lanczos(spectrum_context)

    assert spectrum_observation["summary"] == "published lanczos_energy"
    assert spectrum_context.output.dims == ["channel", "iteration", "state"]
    assert (spectrum_dir / "output.nc").is_file()


def test_qda_fit_divides_by_nonlocal_origin_and_fits_each_sample() -> None:
    rng = np.random.default_rng(17)
    times = np.arange(8.0)
    z = [0.0, 1.0]
    target = 0.72 + 0.18j
    samples = []
    for _ in range(24):
        denominator = np.exp(-0.25 * times) * (1.0 + rng.normal(0.0, 0.01))
        ratio = target + rng.normal(0.0, 0.003, times.size) + 1j * rng.normal(0.0, 0.003, times.size)
        samples.append(np.column_stack([denominator, denominator * ratio]))
    source = EnsembleData(
        _ensemble(0.1),
        "bootstrap",
        samples,
        ["t", "z"],
        {"t": times.tolist(), "z": z},
        attrs={"correlator_type": "qda"},
    )
    values, coordinates, diagnostics = matrix_element_samples(
        {"qda": source},
        method="qda",
        tmin=2,
        tmax=7,
        tau_min=None,
        lsqfit={
            "pt2_windows": [{"tmin": 2, "tmax": 7}],
            "svdcut": 1e-8,
            "posterior_prior_error_scale": 3.0,
            "q_min": 0.0,
        },
        workers=2,
    )
    assert coordinates == z
    assert np.all(values[:, 0] == 1.0)
    assert np.isclose(np.mean(values[:, 1]), target, atol=2e-3)
    assert diagnostics["min_Q"] >= 0.0
    assert all("E0" in fit and "E0_sdev" in fit for fit in diagnostics["fits"])
    production_fit = diagnostics["fits"][0]
    assert len(production_fit["sample_diagnostics"]) == source.n_sample
    assert len(production_fit["E0_samples"]) == source.n_sample
    assert production_fit["sample0_plot"]["plots"][0]["kind"] == "qda_ratio"
    assert production_fit["sample0_plot"]["plots"][0]["series"][0]["x"] == times.tolist()


def test_correlator_publish_requires_complete_scan_and_deterministic_best_candidate(tmp_path, monkeypatch) -> None:
    import lamet_agent.stages.correlator_analysis._publish as tool

    labels = []
    original = tool.configure_plot
    monkeypatch.setattr(tool, "configure_plot", lambda **kwargs: labels.append(kwargs) or original(**kwargs))

    attrs = {"observable": "matrix_element", "sample_error_mode": "one_sigma"}
    low = EnsembleData(
        None,
        "bootstrap",
        [np.array([0.9]), np.array([1.1])],
        ["z"],
        {"z": [0]},
        attrs=attrs,
        name="bare_matrix_element",
    )
    high = EnsembleData(
        None,
        "bootstrap",
        [np.array([1.0]), np.array([1.2])],
        ["z"],
        {"z": [0]},
        attrs=attrs,
        name="bare_matrix_element",
    )
    candidates = [
        {
            "id": "matrix_001",
            "method": "qda",
            "fit_strategy": "independent",
            "nstate": 1,
            "prior_width": 1.0,
            "observable": "matrix_element",
            "window": {"tmin": 2, "tmax": 5, "tau_min": None},
            "data": low,
            "Q": 0.2,
            "chi2_dof": 1.2,
            "min_Q": 0.2,
            "worst_chi2_dof": 1.2,
            "n_data": 6,
            "n_params": 5,
            "quality_passed": True,
            "feasible_at_all_tune_z": True,
        },
        {
            "id": "matrix_002",
            "method": "qda",
            "fit_strategy": "independent",
            "nstate": 1,
            "prior_width": 1.0,
            "observable": "matrix_element",
            "window": {"tmin": 3, "tmax": 6, "tau_min": None},
            "data": high,
            "Q": 0.8,
            "chi2_dof": 0.9,
            "min_Q": 0.8,
            "worst_chi2_dof": 0.9,
            "n_data": 6,
            "n_params": 5,
            "quality_passed": True,
            "feasible_at_all_tune_z": True,
        },
    ]
    params = {
        "observable": "matrix_element",
        "analysis_method": "lsqfit",
        "nstate": [1],
        "fit_scope": ["qda_ratio"],
        "fit_strategy": ["independent"],
        "prior_width": [1.0],
        "q_min": 0.05,
        "chi2_dof_tolerance": 0.25,
        "tune_z_values": [1],
        "pt2_windows": [{"tmin": 2, "tmax": 5}, {"tmin": 3, "tmax": 6}],
    }
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "one_sigma"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "qda",
        params,
        {},
        {},
        {"matrix_element_candidates": candidates},
        tmp_path,
        np.random.default_rng(2),
    )
    with pytest.raises(ValueError, match="deterministic best"):
        tool.run(context, candidate_id="matrix_001")
    tool.run(context, candidate_id="matrix_002")
    assert context.output is high
    assert (tmp_path / "diagnostics" / "candidates.json").is_file()
    assert labels[-1]["xlabel"] == r"$z~/~a$"
    assert labels[-1]["ylabel"] == "bare matrix element"


def test_correlator_window_selection_preserves_original_information_rule() -> None:
    from lamet_agent.stages.correlator_analysis._selection import select_data_window

    candidates = [
        {"id": "largest", "n_data": 24, "n_params": 10, "Q": 0.8, "chi2_dof": 0.70},
        {"id": "best_chi2", "n_data": 18, "n_params": 10, "Q": 0.9, "chi2_dof": 0.50},
        {"id": "outside_tolerance", "n_data": 25, "n_params": 10, "Q": 0.9, "chi2_dof": 0.76},
    ]
    selected, fallback = select_data_window(candidates, q_min=0.05, chi2_dof_tolerance=0.25)
    assert selected["id"] == "largest"
    assert fallback is False
    candidates[0]["numerical_failure"] = True
    selected, fallback = select_data_window(candidates, q_min=0.05, chi2_dof_tolerance=0.25)
    assert selected["id"] == "best_chi2"
    assert fallback is False


def test_matrix_element_prior_keeps_original_inactive_component_parameters() -> None:
    prior = matrix_element_prior(2, form="Breit", scope="3pt_ratio", components=("re",), width_scale=1.0)
    assert "O00_im" in prior and "O01_im" in prior and "O11_im" in prior
    assert float(prior["O00_im"].mean) == 1.0
    assert "log(E0)" in prior
    assert float(prior["E0"].mean) > 0.0


@pytest.mark.parametrize(
    ("typical_abs", "expected_scale"),
    [(3.64e-19, 1.0e15), (3.25e-22, 1.0e18)],
)
def test_correlator_rescale_is_a_data_driven_power_of_ten(typical_abs: float, expected_scale: float) -> None:
    from lamet_agent.stages.correlator_analysis._inspection import (
        _automatic_correlator_rescale,
    )

    data = EnsembleData(
        None,
        "bootstrap",
        [np.full(8, typical_abs), np.full(8, typical_abs)],
        ["t"],
        {"t": list(range(8))},
        attrs={"correlator_type": "two_point"},
    )
    result = _automatic_correlator_rescale({"two_point": data}, [{"tmin": 2, "tmax": 7}])
    assert result["correlator_rescale"] == expected_scale
    assert 1.0e-4 <= result["rescaled_typical_abs"] <= 1.0e-2


def test_inspect_correlators_does_not_write_raw_correlator_plots(tmp_path) -> None:
    from lamet_agent.stages.correlator_analysis._inspection import run

    data = EnsembleData(
        None,
        "bootstrap",
        [np.full(8, 1.0e-3), np.full(8, 1.1e-3)],
        ["t"],
        {"t": list(range(8))},
        attrs={"correlator_type": "two_point"},
    )
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "inspect",
        {},
        {},
        {},
        {"correlators": {"two_point": data}},
        tmp_path,
        np.random.default_rng(1),
    )
    observation = run(context)
    assert observation["artifacts"] == []
    assert isinstance(context.state["inspection"]["two_point"]["effective_mass"], str)
    assert not list((tmp_path / "plots").glob("correlator_*.pdf"))


def test_matrix_fit_tool_records_a_numerically_rejected_candidate(monkeypatch, tmp_path) -> None:
    import lamet_agent.stages.correlator_analysis._fit_matrix as tool

    three_point = EnsembleData(
        None,
        "bootstrap",
        [np.ones((1, 3, 1)), np.ones((1, 3, 1))],
        ["tsep", "tau", "z"],
        {"tsep": [8], "tau": [0, 1, 2], "z": [0]},
        attrs={"correlator_type": "three_point"},
    )
    settings = {
        "fitting_form": "Breit",
        "fit_scope": ["3pt_ratio"],
        "fit_strategy": ["joint"],
        "pt2_windows": [{"tmin": 3, "tmax": 8}, {"tmin": 4, "tmax": 8}],
        "pt3_windows": [{"tsep_ls": [8], "tau_cut": 2}],
        "svdcut": 1e-6,
        "posterior_prior_error_scale": 10.0,
        "q_min": 0.05,
        "chi2_dof_tolerance": 0.25,
    }
    params = {
        "observable": "matrix_element",
        "analysis_method": "lsqfit",
        "component": "re",
        "nstate": [2],
        "prior_width": [1.0],
        **settings,
    }
    context = ToolContext(
        {"metadata": {"workers": 2, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "matrix",
        params,
        {},
        {},
        {"correlators": {"c3": three_point}, "correlator_rescale": 1.0},
        tmp_path,
        np.random.default_rng(2),
    )

    received = []

    def fail_fit(*args, **kwargs):
        received.append(kwargs)
        if kwargs["tmin"] == 3:
            raise FitNumericalError("sample-average fit failed: ZeroDivisionError: float division")
        return None, {
            "tune_z": kwargs["tune_z"],
            "fit_scope": "3pt_ratio",
            "Q": 0.8,
            "chi2": 8.0,
            "dof": 10,
            "chi2_dof": 0.8,
            "logGBF": 2.0,
            "n_data": 12,
            "n_params": 5,
        }

    monkeypatch.setattr(tool, "fit_matrix_element_samples", fail_fit)
    observation = tool.run(context, tune_z_values=[0])
    rejected, accepted = context.state["matrix_element_candidates"]
    assert rejected["numerical_failure"] is True
    assert rejected["feasible_at_all_tune_z"] is False
    assert "ZeroDivisionError" in rejected["failure_reasons"]["0.0"]
    assert accepted["feasible_at_all_tune_z"] is True
    assert observation["metrics"]["recommended_candidate_id"] == "matrix_002"
    assert all(call["fit_samples"] is False for call in received)
    assert all(call["tune_z"] == 0 for call in received)


def test_matrix_fit_tool_scans_authored_grid_in_reference_order(monkeypatch, tmp_path) -> None:
    import lamet_agent.stages.correlator_analysis._fit_matrix as tool

    three_point = EnsembleData(
        None,
        "bootstrap",
        [np.ones((1, 4, 2)), np.ones((1, 4, 2))],
        ["tsep", "tau", "z"],
        {"tsep": [8], "tau": [0, 1, 2, 3], "z": [0, 1]},
        attrs={"correlator_type": "three_point"},
    )
    settings = {
        "fitting_form": "Breit",
        "fit_scope": ["3pt_ratio"],
        "fit_strategy": ["joint"],
        "pt2_windows": [{"tmin": 3, "tmax": 8}, {"tmin": 4, "tmax": 8}],
        "pt3_windows": [
            {"tsep_ls": [8], "tau_cut": 2},
            {"tsep_ls": [8], "tau_cut": 3},
        ],
        "prior_width": [1.0],
        "svdcut": 1e-6,
        "posterior_prior_error_scale": 10.0,
        "q_min": 0.05,
        "chi2_dof_tolerance": 0.25,
    }
    params = {
        "observable": "matrix_element",
        "analysis_method": "lsqfit",
        "component": "re",
        "nstate": [2],
        **settings,
    }
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "matrix",
        params,
        {},
        {},
        {
            "correlators": {"c3": three_point},
            "correlator_rescale": 1.0,
        },
        tmp_path,
        np.random.default_rng(2),
    )
    calls = []

    def tune(*args, **kwargs):
        calls.append((kwargs["tmin"], kwargs["tau_min"], kwargs["tune_z"]))
        return None, {
            "tune_z": kwargs["tune_z"],
            "fit_scope": "3pt_ratio",
            "Q": 0.8,
            "chi2": 8.0,
            "dof": 10,
            "chi2_dof": 0.8,
            "logGBF": 2.0,
            "n_data": 20 - kwargs["tmin"] - kwargs["tau_min"],
            "n_params": 5,
        }

    monkeypatch.setattr(tool, "fit_matrix_element_samples", tune)
    observation = tool.run(context, tune_z_values=[0, 1])

    assert calls == [
        (3, 2, 0.0),
        (3, 2, 1.0),
        (3, 3, 0.0),
        (3, 3, 1.0),
        (4, 2, 0.0),
        (4, 2, 1.0),
        (4, 3, 0.0),
        (4, 3, 1.0),
    ]
    assert observation["metrics"]["candidate_count"] == 4
    assert observation["metrics"]["recommended_candidate_id"] == "matrix_001"
    assert all(candidate["feasible_at_all_tune_z"] for candidate in context.state["matrix_element_candidates"])


def test_qda_fit_tool_tunes_every_window_before_full_application(monkeypatch, tmp_path) -> None:
    import lamet_agent.stages.correlator_analysis._fit_qda as tool

    source = EnsembleData(
        _ensemble(0.1),
        "bootstrap",
        [np.ones((8, 3)), np.ones((8, 3))],
        ["t", "z"],
        {"t": list(range(8)), "z": [0, 1, 2]},
        attrs={"correlator_type": "qda"},
    )
    settings = {
        "fit_scope": ["qda_ratio"],
        "fit_strategy": ["independent"],
        "pt2_windows": [{"tmin": 2, "tmax": 6}, {"tmin": 2, "tmax": 7}],
        "prior_width": [1.0],
        "posterior_prior_error_scale": 3.0,
        "svdcut": 1e-6,
        "q_min": 0.05,
        "chi2_dof_tolerance": 0.25,
    }
    params = {
        "observable": "matrix_element",
        "analysis_method": "lsqfit",
        "component": "both",
        "nstate": [1],
        **settings,
    }
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "qda",
        params,
        {},
        {},
        {"correlators": {"qda": source}},
        tmp_path,
        np.random.default_rng(2),
    )
    calls = []

    def tune(*args, **kwargs):
        calls.append((kwargs["tmin"], kwargs["tmax"], kwargs["tune_z"]))
        q_value = 0.6 if kwargs["tmax"] == 6 else 0.8
        return (
            None,
            [0, 1, 2],
            {
                "tune_z": kwargs["tune_z"],
                "Q": q_value,
                "chi2": 4.0,
                "dof": 8,
                "chi2_dof": 0.5,
                "logGBF": 2.0,
                "n_data": 2 * (kwargs["tmax"] - kwargs["tmin"]),
                "n_params": 5,
            },
        )

    monkeypatch.setattr(tool, "matrix_element_samples", tune)
    observation = tool.run(context, tune_z_values=[1, 2])

    assert calls == [(2, 6, 1.0), (2, 6, 2.0), (2, 7, 1.0), (2, 7, 2.0)]
    assert observation["metrics"]["candidate_count"] == 2
    assert observation["metrics"]["recommended_candidate_id"] == "matrix_002"
    assert all(candidate.get("data") is None for candidate in context.state["matrix_element_candidates"])


def test_publish_applies_only_the_selected_tuned_candidate_to_all_samples(monkeypatch, tmp_path) -> None:
    import lamet_agent.stages.correlator_analysis._publish as tool

    data = EnsembleData(
        None,
        "bootstrap",
        [np.array([0.9, 0.7]), np.array([1.1, 0.8])],
        ["z"],
        {"z": [0, 1]},
        attrs={"observable": "matrix_element"},
        name="bare_matrix_element",
    )
    candidate = {
        "id": "matrix_001",
        "method": "joint",
        "fit_scope": "3pt_ratio",
        "observable": "matrix_element",
        "window": {"tmin": 3, "tmax": 8, "tau_min": 2},
        "tsep_values": [8],
        "nstate": 2,
        "prior_width": 1.0,
        "correlator_rescale": 1.0,
        "quality_passed": True,
        "numerical_failure": False,
        "n_data": 8,
        "n_params": 4,
        "Q": 0.8,
        "chi2_dof": 0.9,
    }
    settings = {
        "fitting_form": "Breit",
        "fit_scope": ["3pt_ratio"],
        "fit_strategy": ["joint"],
        "pt2_windows": [{"tmin": 3, "tmax": 8}],
        "pt3_windows": [{"tsep_ls": [8], "tau_cut": 2}],
        "svdcut": 1e-6,
        "posterior_prior_error_scale": 10.0,
        "q_min": 0.05,
        "chi2_dof_tolerance": 0.25,
        "tune_z_values": [0],
    }
    params = {
        "observable": "matrix_element",
        "analysis_method": "lsqfit",
        "component": "re",
        "nstate": [2],
        "prior_width": [1.0],
        **settings,
    }
    context = ToolContext(
        {"metadata": {"workers": 2, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "matrix",
        params,
        {},
        {},
        {"correlators": {"placeholder": object()}, "matrix_element_candidates": [candidate]},
        tmp_path,
        np.random.default_rng(2),
    )
    calls = []

    def apply_fit(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("fit_samples") is False:
            return None, {"n_failed_samples": 0, "sample_failures": [], "fits": []}
        return data, {"n_failed_samples": 0, "sample_failures": [], "fits": []}

    monkeypatch.setattr(tool, "fit_matrix_element_samples", apply_fit)
    tool.run(context, candidate_id="matrix_001")
    assert len(calls) == 2
    assert calls[0]["tune_z"] is None
    assert calls[0]["fit_samples"] is False
    assert "tune_z" not in calls[1]
    assert "fit_samples" not in calls[1]
    assert context.output is data


def test_publish_fails_immediately_when_selected_candidate_fails_full_grid(monkeypatch, tmp_path) -> None:
    import lamet_agent.stages.correlator_analysis._publish as tool

    candidates = [
        {
            "id": "matrix_001",
            "method": "joint",
            "fit_scope": "3pt_ratio",
            "observable": "matrix_element",
            "window": {"tmin": 3, "tmax": 8, "tau_min": 2},
            "tsep_values": [8],
            "nstate": 2,
            "prior_width": 1.0,
            "correlator_rescale": 1.0,
            "quality_passed": True,
            "numerical_failure": False,
            "n_data": 9,
            "n_params": 4,
            "Q": 0.8,
            "chi2_dof": 0.9,
        },
        {
            "id": "matrix_002",
            "method": "joint",
            "fit_scope": "3pt_ratio",
            "observable": "matrix_element",
            "window": {"tmin": 4, "tmax": 8, "tau_min": 2},
            "tsep_values": [8],
            "nstate": 2,
            "prior_width": 1.0,
            "correlator_rescale": 1.0,
            "quality_passed": True,
            "numerical_failure": False,
            "n_data": 8,
            "n_params": 4,
            "Q": 0.9,
            "chi2_dof": 0.8,
        },
    ]
    settings = {
        "fitting_form": "Breit",
        "fit_scope": ["3pt_ratio"],
        "fit_strategy": ["joint"],
        "pt2_windows": [{"tmin": 3, "tmax": 8}, {"tmin": 4, "tmax": 8}],
        "pt3_windows": [{"tsep_ls": [8], "tau_cut": 2}],
        "svdcut": 1e-6,
        "posterior_prior_error_scale": 10.0,
        "q_min": 0.05,
        "chi2_dof_tolerance": 0.25,
        "tune_z_values": [0],
    }
    params = {
        "observable": "matrix_element",
        "analysis_method": "lsqfit",
        "component": "re",
        "nstate": [2],
        "prior_width": [1.0],
        **settings,
    }
    context = ToolContext(
        {"metadata": {"workers": 2, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "matrix",
        params,
        {},
        {},
        {"correlators": {"placeholder": object()}, "matrix_element_candidates": candidates},
        tmp_path,
        np.random.default_rng(2),
    )
    calls = []

    def apply_fit(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["tmin"] == 3
        assert kwargs["fit_samples"] is False
        raise FitNumericalError("sample-average posterior is unusable")

    monkeypatch.setattr(tool, "fit_matrix_element_samples", apply_fit)
    with pytest.raises(FitNumericalError, match="selected candidate matrix_001 failed full-grid"):
        tool.run(context, candidate_id="matrix_001")
    assert len(calls) == 1
    assert candidates[0]["numerical_failure"] is True
    assert context.output is None
    assert candidates[1]["numerical_failure"] is False


def test_numerically_rejected_matrix_fit_counts_as_an_evaluated_candidate(tmp_path) -> None:
    import json
    from lamet_agent.stages.correlator_analysis._publish import run

    attrs = {"observable": "matrix_element", "sample_error_mode": "covariance"}
    data = EnsembleData(
        None,
        "bootstrap",
        [np.array([0.9]), np.array([1.1])],
        ["z"],
        {"z": [0]},
        attrs=attrs,
        name="bare_matrix_element",
    )
    candidates = [
        {
            "id": "matrix_001",
            "method": "joint",
            "fit_scope": "3pt_ratio",
            "observable": "matrix_element",
            "window": {"tmin": 3, "tmax": 8, "tau_min": 2},
            "nstate": 2,
            "prior_width": 1.0,
            "quality_passed": False,
            "numerical_failure": True,
            "error": "sample-average fit failed",
        },
        {
            "id": "matrix_002",
            "method": "joint",
            "fit_scope": "3pt_ratio",
            "observable": "matrix_element",
            "window": {"tmin": 4, "tmax": 8, "tau_min": 2},
            "nstate": 2,
            "prior_width": 1.0,
            "quality_passed": True,
            "numerical_failure": False,
            "data": data,
            "n_data": 8,
            "n_params": 4,
            "Q": 0.8,
            "chi2_dof": 0.9,
        },
    ]
    params = {
        "observable": "matrix_element",
        "analysis_method": "lsqfit",
        "nstate": [2],
        "fit_scope": ["3pt_ratio"],
        "fit_strategy": ["joint"],
        "prior_width": [1.0],
        "pt2_windows": [{"tmin": 3, "tmax": 8}, {"tmin": 4, "tmax": 8}],
        "pt3_windows": [{"tsep_ls": [8], "tau_cut": 2}],
        "q_min": 0.05,
        "chi2_dof_tolerance": 0.25,
    }
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "matrix",
        params,
        {},
        {},
        {"matrix_element_candidates": candidates},
        tmp_path,
        np.random.default_rng(2),
    )
    run(context, candidate_id="matrix_002")
    table = json.loads((tmp_path / "diagnostics" / "candidates.json").read_text(encoding="utf-8"))["candidates"]
    assert table[0]["numerical_failure"] is True
    assert table[0]["error"] == "sample-average fit failed"


@pytest.mark.parametrize("strategy", ["joint", "chained", "independent"])
@pytest.mark.parametrize("scope", ["3pt_ratio", "FH", "3pt_ratio+FH"])
def test_native_matrix_element_fit_supports_authored_strategies_and_scopes(strategy: str, scope: str) -> None:
    ensemble = EnsembleInfo("toy", "toy", 0.1, 0.1, 32, 32, 0.2)
    times = np.arange(16)
    tseps = np.asarray([8, 10])
    tau = np.arange(11)
    rng = np.random.default_rng(81)
    c2_samples = []
    c3_samples = []
    for _ in range(32):
        energy = 0.3 + rng.normal(0.0, 0.003)
        overlap = 1.4 + rng.normal(0.0, 0.01)
        matrix = 0.8 + rng.normal(0.0, 0.01)
        c2 = overlap**2 / (2 * energy) * (np.exp(-energy * times) + np.exp(-energy * (ensemble.L_t - times)))
        c2 = c2 + rng.normal(0.0, 2e-4, c2.shape)
        c3 = np.zeros((tseps.size, tau.size, 1), dtype=complex)
        for tsep_index, tsep in enumerate(tseps):
            valid = tau <= tsep
            ratio = matrix / (2 * energy) + rng.normal(0.0, 0.002, np.count_nonzero(valid))
            c3[tsep_index, valid, 0] = ratio * c2[tsep]
        c2_samples.append(c2.astype(complex))
        c3_samples.append(c3)
    common = {"source_momentum": "[1, 0, 0]", "sink_momentum": "[1, 0, 0]", "resample_id": "shared"}
    c2_data = EnsembleData(
        ensemble,
        "bootstrap",
        c2_samples,
        ["t"],
        {"t": times.tolist()},
        attrs={**common, "correlator_type": "two_point"},
    )
    c3_data = EnsembleData(
        ensemble,
        "bootstrap",
        c3_samples,
        ["tsep", "tau", "z"],
        {"tsep": tseps.tolist(), "tau": tau.tolist(), "z": [0]},
        attrs={**common, "correlator_type": "three_point"},
    )
    result, diagnostics = fit_matrix_element_samples(
        {"c2": c2_data, "c3": c3_data},
        strategy=strategy,
        fitting_form="Breit",
        fit_scope=scope,
        components="real",
        tmin=3,
        tmax=8,
        tsep_values=tseps.tolist(),
        tau_min=2,
        n_states=1,
        prior_width=1.0,
        correlator_rescale=1.0,
        svdcut=1e-8,
        posterior_prior_error_scale=3.0,
        workers=2 if strategy == "joint" and scope == "3pt_ratio" else 1,
    )
    assert result.dims == ["z"]
    assert np.all(np.isfinite(result.values))
    if scope == "3pt_ratio":
        assert np.isclose(np.mean(np.real(result.values[:, 0])), 0.8 / 0.6, atol=0.12)
    assert diagnostics["strategy"] == strategy
    assert diagnostics["fit_scope"] == scope
    production_fit = diagnostics["fits"][0]
    assert len(production_fit["sample_diagnostics"]) == result.n_sample
    assert len(production_fit["E0_samples"]) == result.n_sample
    expected_kinds = {"pt3_ratio"} if scope == "3pt_ratio" else {"fh"} if scope == "FH" else {"pt3_ratio", "fh"}
    assert {plot["kind"] for plot in production_fit["sample0_plot"]["plots"]} == expected_kinds
    if "pt3_ratio" in expected_kinds:
        ratio_plot = next(plot for plot in production_fit["sample0_plot"]["plots"] if plot["kind"] == "pt3_ratio")
        assert len(ratio_plot["series"]) == len(tseps)
        for series in ratio_plot["series"]:
            x = np.asarray(series["x"], dtype=float)
            fit_x = np.asarray(series["fit_x"], dtype=float)
            assert np.isclose(float(np.min(x) + np.max(x)), 0.0)
            assert np.isclose(float(np.min(fit_x) + np.max(fit_x)), 0.0)
    if strategy == "joint" and scope == "3pt_ratio":
        tuned, tuning = fit_matrix_element_samples(
            {"c2": c2_data, "c3": c3_data},
            strategy=strategy,
            fitting_form="Breit",
            fit_scope=scope,
            components="real",
            tmin=3,
            tmax=8,
            tsep_values=tseps.tolist(),
            tau_min=2,
            n_states=1,
            prior_width=1.0,
            correlator_rescale=1.0,
            svdcut=1e-8,
            posterior_prior_error_scale=3.0,
            workers=2,
            tune_z=0,
            fit_samples=False,
        )
        assert tuned is None
        assert tuning["tune_z"] == 0
        assert len(tuning["fits"]) == 1
        assert tuning["n_failed_samples"] == 0


def test_native_nonbreit_fit_uses_distinct_source_and_sink_spectra() -> None:
    ensemble = EnsembleInfo("toy", "toy", 0.1, 0.1, 32, 32, 0.2)
    times = np.arange(16)
    tseps = np.asarray([8, 10])
    tau = np.arange(11)
    rng = np.random.default_rng(18)
    initial_samples = []
    final_samples = []
    three_point_samples = []
    for _ in range(32):
        energy_i = 0.25 + rng.normal(0.0, 0.002)
        energy_f = 0.35 + rng.normal(0.0, 0.002)
        overlap_i = 1.3 + rng.normal(0.0, 0.01)
        overlap_f = 1.5 + rng.normal(0.0, 0.01)
        target = 1.2 + rng.normal(0.0, 0.005)
        initial = (
            overlap_i**2 / (2 * energy_i) * (np.exp(-energy_i * times) + np.exp(-energy_i * (ensemble.L_t - times)))
        )
        final = overlap_f**2 / (2 * energy_f) * (np.exp(-energy_f * times) + np.exp(-energy_f * (ensemble.L_t - times)))
        three_point = np.zeros((tseps.size, tau.size, 1), dtype=complex)
        for tsep_index, tsep in enumerate(tseps):
            valid = tau <= tsep
            correction = (
                initial[tsep - tau[valid]]
                * final[tau[valid]]
                * final[tsep]
                / (final[tsep - tau[valid]] * initial[tau[valid]] * initial[tsep])
            )
            three_point[tsep_index, valid, 0] = target * final[tsep] / np.sqrt(correction)
        initial_samples.append(initial.astype(complex))
        final_samples.append(final.astype(complex))
        three_point_samples.append(three_point)
    common = {"resample_id": "shared"}
    initial_data = EnsembleData(
        ensemble,
        "bootstrap",
        initial_samples,
        ["t"],
        {"t": times.tolist()},
        attrs={**common, "correlator_type": "two_point", "source_momentum": "[0, 0, 0]", "sink_momentum": "[0, 0, 0]"},
    )
    final_data = EnsembleData(
        ensemble,
        "bootstrap",
        final_samples,
        ["t"],
        {"t": times.tolist()},
        attrs={**common, "correlator_type": "two_point", "source_momentum": "[1, 0, 0]", "sink_momentum": "[1, 0, 0]"},
    )
    three_point_data = EnsembleData(
        ensemble,
        "bootstrap",
        three_point_samples,
        ["tsep", "tau", "z"],
        {"tsep": tseps.tolist(), "tau": tau.tolist(), "z": [0]},
        attrs={
            **common,
            "correlator_type": "three_point",
            "source_momentum": "[0, 0, 0]",
            "sink_momentum": "[1, 0, 0]",
        },
    )
    result, diagnostics = fit_matrix_element_samples(
        {"initial": initial_data, "final": final_data, "three_point": three_point_data},
        strategy="joint",
        fitting_form="NonBreit",
        fit_scope="3pt_ratio",
        components="real",
        tmin=3,
        tmax=8,
        tsep_values=tseps.tolist(),
        tau_min=2,
        n_states=1,
        prior_width=1.0,
        correlator_rescale=1.0,
        svdcut=1e-8,
        posterior_prior_error_scale=3.0,
        workers=1,
    )
    assert np.isclose(np.mean(np.real(result.values[:, 0])), 1.2, atol=0.05)
    assert diagnostics["fitting_form"] == "NonBreit"


def test_correlated_spectrum_fit_uses_authored_priors_and_sample_covariance() -> None:
    rng = np.random.default_rng(8)
    times = np.arange(2.0, 10.0)
    center = 1.4 * np.exp(-0.27 * times)
    samples = np.asarray([center + rng.normal(0.0, 2e-4, times.size) for _ in range(80)])
    energies, diagnostics = fit_spectrum_samples(
        samples, times, 1, resample="bootstrap", prior_means={"E0": 0.3, "A0": 1.3}, prior_widths={"E0": 0.2, "A0": 0.5}
    )
    assert np.isclose(np.mean(energies), 0.27, atol=5e-3)
    assert diagnostics["dof"] == 8
    assert 0.0 <= diagnostics["Q"] <= 1.0


def test_extrapolation_supports_block_and_full_x_covariance() -> None:
    rng = np.random.default_rng(81)
    x = [-0.2, 0.2]
    physical = np.asarray([0.8, 1.1])
    data = []
    for index, spacing in enumerate([0.05, 0.07, 0.09, 0.11]):
        center = physical + 0.3 * spacing / 0.1
        samples = [center + rng.normal(0.0, 0.01, 2) for _ in range(40)]
        data.append(
            EnsembleData(
                _ensemble(spacing, f"ensemble_{index}"),
                "bootstrap",
                samples,
                ["x"],
                {"x": x},
                attrs={
                    "momentum_gev": 2.0,
                    "resample_id": f"ensemble_{index}",
                },
            )
        )
    result, diagnostics = fit_candidate(
        data, ["a"], 0.135, {"mean": 0.0, "sdev": 1.0}, x_range=(-0.2, 0.2), pdep_gev=[1.5, 2.0]
    )
    assert result.dims == ["x"]
    assert np.allclose(np.asarray(result.mean), physical, atol=0.05)
    assert diagnostics["dof"] > 0
    assert 0.0 <= diagnostics["Q"] <= 1.0
    assert set(diagnostics["parameter_mean"]) == {"h0", "a"}
    assert set(diagnostics["parameter_sdev"]) == {"h0", "a"}
    assert set(diagnostics["momentum_dependence"]) == {"1.5", "2"}
    np.testing.assert_allclose(diagnostics["momentum_dependence"]["1.5"]["mean"], diagnostics["parameter_mean"]["h0"])
    full_result, full_diagnostics = fit_candidate(
        data,
        ["a"],
        0.135,
        {"mean": 0.0, "sdev": 1.0},
        x_range=(-0.2, 0.2),
        x_independent_terms=["a"],
        x_covariance=True,
    )
    assert full_result.attrs["x_covariance"] == 1
    assert np.allclose(np.asarray(full_result.mean), physical, atol=0.05)
    assert full_diagnostics["x_covariance"] is True
    assert np.asarray(full_diagnostics["parameter_mean"]["a"]).ndim == 0


def test_extrapolation_covariance_is_blocked_by_ensemble_source() -> None:
    from lamet_agent.stages.extrapolation.physics import _grouped_centers_and_covariances

    base = np.arange(12.0).reshape(6, 2)
    values = np.stack([base, 2.0 * base, 3.0 * base, 4.0 * base])
    ensemble_a = EnsembleInfo("test", "A", 0.1, 0.1, 32, 64, 0.14)
    ensemble_b = EnsembleInfo("test", "B", 0.1, 0.1, 32, 64, 0.14)
    data = [
        EnsembleData(
            ensemble_a if index < 2 else ensemble_b,
            "bootstrap",
            list(item),
            ["x"],
            {"x": [0.0, 1.0]},
            attrs={"resample_id": "shared"},
        )
        for index, item in enumerate(values)
    ]

    _centers, per_x = _grouped_centers_and_covariances(values, data, "covariance", x_covariance=False)
    assert per_x[0][0, 1] != 0.0
    assert per_x[0][2, 3] != 0.0
    assert np.allclose(per_x[0][:2, 2:], 0.0)
    _centers, full = _grouped_centers_and_covariances(values, data, "covariance", x_covariance=True)
    assert full[0, 2] != 0.0
    assert full[4, 6] != 0.0
    assert np.allclose(full[:4, 4:], 0.0)


def test_extrapolation_comparison_requires_the_single_reference_candidate() -> None:
    from lamet_agent.stages.extrapolation.workflow import select_single_candidate

    data = EnsembleData(None, "bootstrap", [[0.8, 1.0], [0.9, 1.1]], ["x"], {"x": [-0.2, 0.2]})
    candidate = {
        "id": "extrapolation_001",
        "terms": ["a"],
        "excluded_ensembles": [],
        "data": data,
        "chi2": 1.0,
        "dof": 2.0,
        "chi2_dof": 0.5,
        "Q": 0.8,
        "aic": 3.0,
        "parameter_mean": {"h0": [0.85, 1.05], "a": 0.1},
        "parameter_sdev": {"h0": [0.05, 0.05], "a": 0.02},
        "momentum_dependence": {"2": {"momentum_gev": 2.0, "mean": [0.85, 1.05], "sdev": [0.05, 0.05]}},
    }
    selected, comparison = select_single_candidate([candidate])
    assert comparison["weights"] == [1.0]
    assert selected is data
    with pytest.raises(ValueError, match="exactly one"):
        select_single_candidate([])


def test_self_renormalization_factor_is_not_a_placeholder() -> None:
    z = np.array([0.0, 0.1, 0.2])
    spacings = [0.06, 0.12, 0.18]
    references = []
    for spacing in spacings:
        known = log_m(z, spacing, k=0.4, lambda_qcd_gev=0.2, d=0.0, n_f=3, scale_gev=2.0)
        g = 0.15 * z / 0.1973269804
        f = 0.4 * spacing
        center = np.exp(known + g + f)
        references.append(
            EnsembleData(
                _ensemble(spacing, f"r{spacing}"),
                "bootstrap",
                [center * (1.0 + shift) for shift in (-0.002, 0.0, 0.002)],
                ["z"],
                {"z": z.tolist()},
                attrs={"resample_id": f"r{spacing}"},
            )
        )
    factor = fit_factor(
        references,
        short_distance_max_fm=0.2,
        k=0.4,
        lambda_qcd_gev=0.2,
        d=0.0,
        n_f=3,
        scale_gev=2.0,
        zms_kernel=load_renormalization_kernel("z_msbar_pdf_nlo"),
        kernel_id="z_msbar_pdf_nlo",
        svdcut=1e-12,
    )
    assert factor.dims == ["a", "z"]
    assert factor.n_sample == 1
    assert not np.allclose(factor.values, 1.0)
    assert factor.attrs["m0_convention"] == "reference_inverse_fm"
    assert factor.attrs["kernel_id"] == "z_msbar_pdf_nlo"
    assert np.isfinite(float(factor.attrs["m0_gev"]))


def test_self_renormalization_accepts_one_sample_bearing_a_z_reference() -> None:
    z = np.array([0.0, 0.1, 0.2])
    spacings = [0.06, 0.12, 0.18]
    grids = []
    for spacing in spacings:
        known = log_m(z, spacing, k=0.4, lambda_qcd_gev=0.2, d=0.0, n_f=3, scale_gev=2.0)
        grids.append(np.exp(known + 0.15 * z / HBAR_C_GEV_FM + 0.4 * spacing))
    reference = EnsembleData(
        None, "bootstrap", [np.stack(grids), np.stack(grids) * 1.001], ["a", "z"], {"a": spacings, "z": z.tolist()}
    )
    factor = fit_factor(
        reference,
        short_distance_max_fm=0.2,
        k=0.4,
        lambda_qcd_gev=0.2,
        d=0.0,
        n_f=3,
        scale_gev=2.0,
        zms_kernel=load_renormalization_kernel("z_msbar_pdf_nlo"),
        kernel_id="z_msbar_pdf_nlo",
        svdcut=1e-12,
    )
    assert factor.dims == ["a", "z"]
    assert factor.n_sample == 1
    assert np.allclose(factor.coords["a"], spacings)
    assert factor.attrs["kernel_id"] == "z_msbar_pdf_nlo"


def test_explicit_zmsbar_kernels_preserve_pdf_and_da_finite_terms() -> None:
    z = np.array([0.1, 0.2])
    pdf = np.asarray(load_renormalization_kernel("z_msbar_pdf_nlo")(z, mu=2.0), dtype=float)
    da = np.asarray(load_renormalization_kernel("z_msbar_da_nlo")(z, mu=2.0), dtype=float)
    assert np.all(np.isfinite(pdf)) and np.all(np.isfinite(da))
    assert np.all(da > pdf)
    with pytest.raises(ValueError, match="not available"):
        load_renormalization_kernel("missing_renormalization_formula")


def test_renormalization_kernel_mu_override_warns_and_replaces_context(capsys) -> None:
    import lamet_agent.stages.renormalization.contract as contract
    from lamet_agent.contract import CheckContext
    from lamet_agent.stages.renormalization.physics import zmsbar_log

    seen = {}

    def kernel(z_fm: np.ndarray | float, mu: float = 2.0):
        seen.update({"z_fm": np.asarray(z_fm), "mu": mu})
        return np.ones_like(np.asarray(z_fm), dtype=float)

    params = {
        "strategy": "self_renormalization",
        "kernel_id": "z_msbar_da_nlo",
        "kernel_parameters": {"mu": 3.0},
    }
    context = CheckContext({}, "renormalization", "apply", params, {})
    assert contract.check_kernel(context) == []
    assert "ATTENTION: renormalization kernel_parameters overrides stage context" in capsys.readouterr().out
    overrides = params["kernel_parameters"]
    result = zmsbar_log(kernel, np.asarray([0.1, 0.2]), scale_gev=2.0, kernel_parameters=overrides)

    assert seen["mu"] == 3.0
    np.testing.assert_allclose(seen["z_fm"], [0.1, 0.2])
    np.testing.assert_allclose(result, 0.0)


def test_nla_tail_fit_recovers_a_complex_toy() -> None:
    z = np.arange(-1.0, 1.01, 0.1)
    parameters = {"A2": 0.8, "A2p": -0.2, "phi2": 0.25, "phi2p": -0.1, "Lambda": 0.7}
    values = tail_model_values(np.where(np.abs(z) < 1e-12, 1e-6, z), "gi_nla", parameters)
    values[np.isclose(z, 0.0)] = 1.0
    rng = np.random.default_rng(4)
    samples = [
        values + rng.normal(0.0, 2e-4, values.shape) + 1j * rng.normal(0.0, 2e-4, values.shape) for _ in range(32)
    ]
    data = EnsembleData(None, "bootstrap", samples, ["z"], {"z": z.tolist()}, attrs={"coord_unit": "fm"})
    fitted, diagnostics = fit_tail_parameters(
        data,
        model_id="gi_nla",
        z_min_fm=0.3,
        z_max_fm=1.0,
        prior_means=parameters,
        prior_widths={key: 1.0 for key in parameters},
    )
    assert diagnostics["dof"] > 0
    assert np.isclose(np.mean([record["Lambda"] for record in fitted]), parameters["Lambda"], atol=2e-2)


def test_da_tail_uses_two_endpoint_phases_and_light_light_alias() -> None:
    z = np.arange(-1.0, 1.01, 0.1)
    parameters = {"A1": 0.7, "phi1": 0.2, "Lambda": 0.6}
    nonzero = np.where(np.isclose(z, 0.0), 1e-6, z)
    values = tail_model_values(
        nonzero,
        "gi_nla",
        parameters,
        order="LA",
        observable="DA",
        momentum_gev=2.0,
        psi1_flavor_class="light",
        psi2_flavor_class="light",
    )
    values[np.isclose(z, 0.0)] = 1.0
    rng = np.random.default_rng(23)
    samples = [values + rng.normal(0.0, 2e-4, z.size) + 1j * rng.normal(0.0, 2e-4, z.size) for _ in range(32)]
    data = EnsembleData(
        None, "bootstrap", samples, ["z"], {"z": z.tolist()}, attrs={"coord_unit": "fm", "momentum_gev": 2.0}
    )
    fitted, diagnostics = fit_tail_parameters(
        data,
        model_id="gi_nla",
        z_min_fm=0.3,
        z_max_fm=1.0,
        prior_means=parameters,
        prior_widths={key: 1.0 for key in parameters},
        order="LA",
        observable="DA",
        psi1_flavor_class="light",
        psi2_flavor_class="light",
    )
    assert diagnostics["Q"] >= 0.0
    assert np.isclose(np.mean([record["Lambda"] for record in fitted]), parameters["Lambda"], atol=2e-2)


def test_fourier_transform_parallel_chunks_match_serial_order() -> None:
    z = np.linspace(-0.5, 0.5, 11)
    x = np.linspace(-1.0, 1.0, 13).tolist()
    rng = np.random.default_rng(91)
    samples = [np.exp(-(z**2)) + 1j * z + rng.normal(0.0, 1e-3, z.size) for _ in range(12)]
    data = EnsembleData(None, "bootstrap", samples, ["z"], {"z": z.tolist()}, attrs={"momentum_gev": 2.0})
    serial = fourier_transform(data, x, momentum_gev=2.0, phase_sign=1, prefactor="pz_over_2pi", workers=1)
    parallel = fourier_transform(data, x, momentum_gev=2.0, phase_sign=1, prefactor="pz_over_2pi", workers=2)
    assert np.allclose(parallel.values, serial.values, rtol=1e-13, atol=1e-13)


def test_fourier_transform_uses_dimensionless_lambda_measure_on_uniform_grid() -> None:
    z = [-0.1, 0.0, 0.1]
    momentum = 2.0
    data = EnsembleData(None, "bootstrap", [np.ones(3)], ["z"], {"z": z})
    transformed = fourier_transform(data, [0.0], momentum_gev=momentum, prefactor="pz_over_2pi")
    expected = momentum * 0.1 * len(z) / (2.0 * np.pi * HBAR_C_GEV_FM)
    assert np.allclose(transformed.values, [[expected]], rtol=1e-13, atol=1e-13)


def test_native_fourier_scan_fits_and_transforms_with_one_parallel_entry(monkeypatch) -> None:
    import lamet_agent.stages.fourier_transform.physics as fourier_physics

    z = np.linspace(-1.0, 1.0, 21)
    parameters = {"A2": 0.8, "A2p": -0.05, "phi2": 0.2, "phi2p": -0.1, "Lambda": 0.7}
    center = tail_model_values(np.where(np.isclose(z, 0.0), 1e-6, z), "gi_nla", parameters)
    center[np.isclose(z, 0.0)] = 1.0
    rng = np.random.default_rng(19)
    samples = [center + rng.normal(0.0, 2e-3, z.size) + 1j * rng.normal(0.0, 2e-3, z.size) for _ in range(20)]
    data = EnsembleData(
        None,
        "bootstrap",
        samples,
        ["z"],
        {"z": z.tolist()},
        attrs={"coord_unit": "fm", "momentum_gev": 2.0, "symmetry": '{"imag":"odd","real":"even"}'},
    )
    modes = []
    actual_fit_tail_parameters = fourier_physics.fit_tail_parameters

    def record_fit_mode(*args, **kwargs):
        modes.append(kwargs.get("mode", "resamples"))
        return actual_fit_tail_parameters(*args, **kwargs)

    monkeypatch.setattr(fourier_physics, "fit_tail_parameters", record_fit_mode)
    result = scan_fourier_transform(
        data,
        np.linspace(-1.0, 1.0, 17).tolist(),
        transform={"phase_sign": 1, "x_shift": 0.0, "prefactor": "pz_over_2pi"},
        tail={
            "models": ["gi_nla"],
            "z_min_fm": [0.3, 0.4],
            "z_max_fm": [0.8, 1.0],
            "extent_fm": 1.23,
            "smoothing_method": "linear",
        },
        scan={
            "orders": ["LA", "NLA"],
            "sector": "full",
            "lambda0_gev": 0.1,
            "prior_widths": [1.0],
            "model_average": False,
            "max_schemes": 3,
            "component": "both",
            "output_scale": 1.0,
            "q_min": 0.0,
        },
        workers=2,
    )
    assert result["data"].dims == ["x"]
    assert result["data"].n_sample == data.n_sample
    assert np.all(np.isfinite(result["data"].values))
    assert 1 <= len(result["selected_labels"]) <= 2
    assert np.sum(result["weights"]) == pytest.approx(1.0)
    assert len(result["range_candidates"]) == 3
    assert len(result["model_candidates"]) == 2
    assert all("fit_parameters" in candidate for candidate in result["range_candidates"] if candidate["fit_success"])
    assert all(
        set(candidate["parameter_mean"]) == set(candidate["parameter_sdev"]) for candidate in result["model_candidates"]
    )
    assert max(result["selected_candidate"]["extended"].coords["z"]) == pytest.approx(1.2)
    assert modes == ["center"] * 3 + ["resamples"] * 2
    assert result["workers"] == 2


def test_fourier_scan_plot_draws_extrapolation_only_from_selected_zmin(monkeypatch, tmp_path) -> None:
    import lamet_agent.stages.fourier_transform._scan as tool

    z = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
    source = EnsembleData(
        _ensemble(0.1),
        "bootstrap",
        [np.exp(-np.abs(z)) + 0.1j * np.asarray(z), np.exp(-np.abs(z)) + 0.2j * np.asarray(z)],
        ["z"],
        {"z": z},
        attrs={"coord_unit": "fm", "momentum_gev": 2.0},
    )
    extended = source.copy()
    output = EnsembleData(
        _ensemble(0.1),
        "bootstrap",
        [np.ones(3), 1.1 * np.ones(3)],
        ["x"],
        {"x": [-0.5, 0.0, 0.5]},
    )
    candidate = {
        "label": "gi_nla_NLA_w1_linear_0p1",
        "model_id": "gi_nla",
        "z_min_fm": 0.2,
        "z_max_fm": 0.3,
        "order": "NLA",
        "prior_width": 1.0,
        "smoothing_method": "linear",
        "smoothing_width_fm": 0.1,
        "chi2": 1.0,
        "dof": 2,
        "chi2_dof": 0.5,
        "Q": 0.8,
        "logGBF": 2.0,
        "extended": extended,
    }
    result = {
        "data": output,
        "selected_range": {"model_id": "gi_nla", "z_min_fm": 0.2, "z_max_fm": 0.3},
        "model_candidates": [candidate],
        "weights": [1.0],
        "selected_labels": [candidate["label"]],
        "selected_candidate": candidate,
        "range_candidates": [candidate],
        "workers": 1,
    }
    monkeypatch.setattr(tool, "scan_fourier_transform", lambda *args, **kwargs: result)
    plotted = []
    boundaries = []
    configured = []
    monkeypatch.setattr(tool, "start_plot", lambda: None)
    monkeypatch.setattr(tool, "configure_plot", lambda **kwargs: configured.append(kwargs))
    monkeypatch.setattr(tool, "save_figure", lambda path: Path(path).touch())
    monkeypatch.setattr(tool, "errorband", lambda x, values, **kwargs: plotted.append((kwargs["label"], list(x))))
    monkeypatch.setattr(tool, "vline", lambda value, **kwargs: boundaries.append((value, kwargs)))
    params = {
        "parton": "quark",
        "gfix": "GI",
        "quasi_y_ls": [-0.5, 0.0, 0.5],
        "transform": {"phase_sign": 1, "x_shift": 0.0, "prefactor": "pz_over_2pi"},
        "tail_models": ["gi_nla"],
        "zmin_fm": [0.2],
        "tail_window_step_offset": 0,
        "zmax_fm": [0.3],
        "zmax_ext_fm": 0.4,
        "smooth": "linear",
        "scheme_scan": {
            "order": ["NLA"],
            "sector": "full",
            "Lambda0_gev": 0.1,
            "posterior_prior_error_scale": [1.0],
            "model_average": False,
            "max_schemes": 1,
            "component": "both",
            "output_scale": 1.0,
            "q_min": 0.0,
        },
    }
    context = ToolContext(
        {"metadata": {"workers": 1, "target_observable": "pdf", "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "fourier_transform",
        "fourier",
        params,
        {"metadata": {"sample_error_mode": "covariance"}},
        {},
        {
            "tail_inspection": {},
            "fourier_input": source,
            "fourier_conventions": {
                "parton": "quark",
                "gfix": "GI",
                "transform": {"phase_sign": 1, "x_shift": 0.0, "prefactor": "pz_over_2pi"},
                "tail_models": ["gi_nla"],
                "component": "both",
                "output_scale": 1.0,
                "q_min": 0.05,
            },
        },
        tmp_path,
        np.random.default_rng(2),
    )

    tool.run(context)

    scale = 2.0 / HBAR_C_GEV_FM
    input_curves = [x for label, x in plotted if label == "input"]
    extrapolation_curves = [x for label, x in plotted if label == "extrapolation"]
    assert len(input_curves) == 2
    assert len(extrapolation_curves) == 2
    for coordinates in input_curves:
        np.testing.assert_allclose(coordinates, np.asarray(z) * scale)
    for coordinates in extrapolation_curves:
        np.testing.assert_allclose(coordinates, np.asarray([0.2, 0.3, 0.4]) * scale)
    np.testing.assert_allclose([value for value, _ in boundaries], np.asarray([0.2, 0.3, 0.2, 0.3]) * scale)
    assert all(item == {"color": "black", "linestyle": "dashed"} for _, item in boundaries)
    assert [item["xlabel"] for item in configured[-2:]] == [r"$\lambda = zP^z$", r"$\lambda = zP^z$"]
    assert configured[0]["xlabel"] == r"$x$"
    assert configured[0]["ylabel"] == r"$\tilde q(x)$"


def test_fourier_inspection_applies_systematic_offset_from_ensemble(tmp_path: Path) -> None:
    from lamet_agent.stages.fourier_transform._inspection import effective_zmin_fm, run

    data = EnsembleData(
        _ensemble(0.1),
        "bootstrap",
        [np.ones(11), np.ones(11)],
        ["z"],
        {"z": [-0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25]},
        attrs={
            "coord_unit": "fm",
            "momentum_gev": 2.0,
            "parton": "quark",
            "gfix": "GI",
            "polarization": "unpolarized",
        },
    )
    source = tmp_path / "input.nc"
    data.to_netcdf(source)
    data = EnsembleData.from_netcdf(source)
    context = ToolContext(
        {"metadata": {"target_observable": "pdf", "parton": "quark"}},
        Path("manifest.json"),
        "fourier_transform",
        "offset",
        {
            "zmin_fm": [0.05],
            "zmax_fm": [0.2],
            "zmax_ext_fm": 0.25,
            "tail_window_step_offset": 1,
            "scheme_scan": {"sector": "full"},
        },
        {"input": data},
        {},
        {},
        Path("."),
        np.random.default_rng(1),
    )

    run(context)

    assert effective_zmin_fm(context, data) == [0.15]
    assert context.state["tail_inspection"]["z_grid_step_fm"] == 0.05


def test_fourier_model_choice_is_made_per_sample() -> None:
    from lamet_agent.stages.fourier_transform.physics import _sample_model_weights

    candidates = [
        {
            "sample_failures": [None, None],
            "sample_diagnostics": [
                {"Q": 0.8, "logGBF": 4.0},
                {"Q": 0.8, "logGBF": 1.0},
            ],
        },
        {
            "sample_failures": [None, None],
            "sample_diagnostics": [
                {"Q": 0.8, "logGBF": 1.0},
                {"Q": 0.8, "logGBF": 4.0},
            ],
        },
    ]
    weights = _sample_model_weights(candidates, n_sample=2, q_min=0.05, model_average=False)
    np.testing.assert_array_equal(weights, np.eye(2))


def test_fourier_range_selection_matches_original_q_and_loggbf_rule() -> None:
    from lamet_agent.stages.fourier_transform.physics import _select_fourier_range

    passing = [
        {"id": "lower_evidence", "fit_success": True, "Q": 0.7, "logGBF": 1.0},
        {"id": "higher_evidence", "fit_success": True, "Q": 0.2, "logGBF": 4.0},
        {"id": "failed", "fit_success": False, "Q": 0.9, "logGBF": 8.0},
    ]
    assert _select_fourier_range(passing, q_min=0.05)["id"] == "higher_evidence"

    fallback = [
        {"id": "largest_q", "fit_success": True, "Q": 0.04, "logGBF": float("nan")},
        {"id": "lower_q", "fit_success": True, "Q": 0.03, "logGBF": 20.0},
    ]
    assert _select_fourier_range(fallback, q_min=0.05)["id"] == "largest_q"


def test_fourier_scan_uses_original_fixed_first_pass_priors() -> None:
    from lamet_agent.stages.fourier_transform.physics import _scan_tail_priors

    means, widths = _scan_tail_priors(
        model_id="cg_nla",
        order="NLA",
        lambda0_gev=0.1,
    )
    assert means == {
        "A2": 1.0,
        "phi2": 0.0,
        "A2p": 0.1,
        "phi2p": 0.0,
        "Lambda": 0.4,
        "n": 0.5,
    }
    assert set(widths.values()) == {3.0}


def test_extrapolation_lattice_spacing_basis_uses_original_units() -> None:
    data = EnsembleData(
        _ensemble(0.08, L_s=32),
        "bootstrap",
        [[1.0], [1.0]],
        ["x"],
        {"x": [0.0]},
        attrs={"momentum_gev": 2.0},
    )
    assert basis_terms(data, ["a", "a2", "a4", "ap4"], 0.135) == pytest.approx(
        [0.08, 0.08**2, 0.08**4, (0.08 * 2.0) ** 4]
    )


def test_extrapolation_systematics_budget_uses_envelopes_and_quadrature(tmp_path) -> None:
    import xarray as xr

    from lamet_agent.stages.extrapolation._systematics_budget import run

    x = [0.0, 0.5, 1.0]
    attrs = {"sample_error_mode": "covariance"}

    def distribution(first, second):
        return EnsembleData(
            None,
            "bootstrap",
            [np.asarray(first, dtype=float), np.asarray(second, dtype=float)],
            ["x"],
            {"x": x},
            attrs=attrs,
        )

    main = distribution([1.0, 2.0, 3.0], [1.2, 2.2, 3.2])
    lambda_low = distribution([0.8, 1.8, 2.8], [1.0, 2.0, 3.0])
    lambda_high = distribution([1.3, 2.3, 3.3], [1.5, 2.5, 3.5])
    mu = distribution([1.0, 1.9, 3.0], [1.2, 2.1, 3.2])
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "extrapolation",
        "budget",
        {
            "operation": "systematics_budget",
            "systematics_prescription": "variant_envelope_quadrature",
            "systematics_groups": {
                "main": 0,
                "zs": [],
                "lambda_extrapolation": [1, 2],
                "lamet_scale": [3],
                "other_extrapolations": [],
            },
        },
        {"distributions": [main, lambda_low, lambda_high, mu]},
        {},
        {},
        tmp_path,
        np.random.default_rng(2),
    )

    run(context)

    output = xr.load_dataset(tmp_path / "output.nc")
    lambda_error = np.full(3, 0.5)
    mu_error = np.asarray([0.0, 0.1, 0.0])
    expected_systematic = np.sqrt(lambda_error**2 + mu_error**2)
    assert np.allclose(output["lambda_extrapolation"], lambda_error)
    assert np.allclose(output["lamet_scale"], mu_error)
    assert np.allclose(output["total_systematic_error"], expected_systematic)
    assert np.allclose(
        output["total_error"],
        np.sqrt(np.asarray(main.sdev) ** 2 + expected_systematic**2),
    )
    assert context.output is main
    assert context.summary["result"] == "systematics_budget"


def test_matching_terminal_writes_original_quasi_matched_plot_pair(tmp_path) -> None:
    from lamet_agent.stages.perturbative_matching._apply import run

    x = [-0.5, 0.0, 0.5]
    quasi = EnsembleData(
        None,
        "bootstrap",
        [np.array([0.2, 1.0, 0.3]), np.array([0.3, 1.1, 0.4])],
        ["x"],
        {"x": x},
        attrs={"momentum_gev": 1.722, "sample_error_mode": "covariance"},
        name="quasi_distribution",
    )

    def selection_kernel(x_out, x_in, *, momentum_gev, scale_gev):
        assert momentum_gev == 2.5
        assert scale_gev == 3.0
        assert list(x_out) == [-0.5, 0.5]
        return np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    params = {
        "kernel_id": "quark_pdf_cg_gt_ratio_nlo",
        "scheme": "ratio",
        "mu": 2.0,
        "lc_x_ls": [-0.5, 0.5],
        "kernel_parameters": {"momentum_gev": 2.5, "scale_gev": 3.0},
    }
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "perturbative_matching",
        "match",
        params,
        {},
        {},
        {
            "kernel": selection_kernel,
            "quasi": quasi,
            "kernel_inspection": {"document": "Kernel formula."},
        },
        tmp_path,
        np.random.default_rng(1),
    )
    observation = run(context)
    assert (tmp_path / "plots" / "result.pdf").is_file()
    assert (tmp_path / "plots" / "result.svg").is_file()
    assert "plots/result.pdf" in context.summary["artifacts"]
    assert "plots/result.svg" in observation["artifacts"]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "[PDF](plots/result.pdf)" in report
    result_svg = (tmp_path / "plots" / "result.svg").read_text(encoding="utf-8")
    assert "FillBetweenPolyCollection" in result_svg
    assert r"$P_z=1.72\,\mathrm{GeV}$" in result_svg
    assert r"$x$" in result_svg
    np.testing.assert_allclose(context.output.values, quasi.values[:, [0, 2]])


def test_external_renormalization_terminal_writes_publication_artifacts(tmp_path) -> None:
    from lamet_agent.stages.renormalization._apply import run
    from lamet_agent.stages.renormalization._inspection import run as inspect

    target = EnsembleData(
        _ensemble(0.1),
        "bootstrap",
        [np.array([2.0, 4.0]), np.array([2.2, 4.4])],
        ["z"],
        {"z": [0.0, 0.1]},
        attrs={"coord_unit": "fm", "resample_id": "shared"},
    )
    denominator = EnsembleData(
        _ensemble(0.1),
        "bootstrap",
        [np.array([2.0, 2.0]), np.array([2.2, 2.2])],
        ["z"],
        {"z": [0.0, 0.1]},
        attrs={"coord_unit": "fm", "resample_id": "shared"},
    )
    context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "renormalization",
        "apply",
        {"type": "apply", "scheme": "ratio", "strategy": "external_denominator", "normalization": False},
        {"target": target, "denominator": denominator},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    inspect(context)
    observation = run(context)
    assert observation["summary"] == "published renormalized matrix element"
    assert (tmp_path / "output.nc").is_file()
    assert (tmp_path / "plots" / "result.pdf").is_file()
    assert np.allclose(context.output.values, [[1.0, 2.0], [1.0, 2.0]])


def _self_coverage_context(tmp_path, *, policy: str, scheme: str = "msbar") -> ToolContext:
    factor = EnsembleData(
        None,
        "bootstrap",
        [np.asarray([[2.0, 2.5, 3.0, 4.0]]), np.asarray([[2.0, 2.5, 3.0, 4.0]])],
        ["a", "z"],
        {"a": [0.1], "z": [0.05, 0.1, 0.15, 0.2]},
        attrs={"coord_unit": "fm", "d": 0.0, "m0_gev": 0.0, "k": 0.65, "n_f": 3, "scale_gev": 2.0},
    )
    target = EnsembleData(
        _ensemble(0.1),
        "bootstrap",
        [np.asarray([1.0, 2.0, 4.0, 8.0]), np.asarray([1.0, 2.0, 4.0, 8.0])],
        ["z"],
        {"z": [0.0, 0.1, 0.2, 0.3]},
        attrs={"coord_unit": "fm"},
    )
    inputs = {"target": target, "zR": factor}
    if scheme == "hybrid":
        inputs["denominator"] = EnsembleData(
            _ensemble(0.1),
            "bootstrap",
            [np.full(4, 2.0), np.full(4, 2.0)],
            ["z"],
            {"z": [0.0, 0.1, 0.2, 0.3]},
            attrs={"coord_unit": "fm"},
        )
    params = {
        "type": "apply",
        "scheme": scheme,
        "strategy": "self_renormalization",
        "kernel_id": "z_msbar_da_nlo",
        "kernel_parameters": {},
        "normalization": False,
        "mu": 2.0,
        "LambdaQCD_gev": 0.1,
        "d": 0.0,
        "m0_gev": 0.0,
        "z_coverage_policy": policy,
    }
    if scheme == "hybrid":
        params["zs_fm"] = 0.1
    return ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "renormalization",
        "apply",
        params,
        inputs,
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )


def test_self_renormalization_strict_rejects_uncovered_target_z(tmp_path) -> None:
    from lamet_agent.stages.renormalization._apply import run
    from lamet_agent.stages.renormalization._inspection import run as inspect

    context = _self_coverage_context(tmp_path, policy="strict")
    inspect(context)
    with pytest.raises(ValueError, match="outside the fitted zR range"):
        run(context)


@pytest.mark.parametrize("scheme", ["msbar", "hybrid"])
def test_self_renormalization_intersection_trims_the_output_grid(tmp_path, scheme) -> None:
    from lamet_agent.stages.renormalization._apply import run
    from lamet_agent.stages.renormalization._inspection import run as inspect

    context = _self_coverage_context(tmp_path, policy="intersection", scheme=scheme)
    inspect(context)
    run(context)

    assert context.output.coords["z"] == [0.0, 0.1, 0.2]
    assert context.output.attrs["z_coverage_policy"] == "intersection"
    assert context.output.attrs["n_z_dropped"] == 1
    assert context.summary["diagnostics"]["n_z_coverage_dropped"] == 1
    assert context.summary["diagnostics"]["n_z_extrapolated"] == 0


def test_hybrid_self_renormalization_extrapolates_the_completed_factor(tmp_path) -> None:
    from lamet_agent.stages.renormalization._apply import run
    from lamet_agent.stages.renormalization._inspection import run as inspect

    context = _self_coverage_context(tmp_path, policy="extrapolate", scheme="hybrid")
    inspect(context)
    run(context)

    assert context.output.coords["z"] == [0.0, 0.1, 0.2, 0.3]
    assert np.all(np.isfinite(context.output.values))
    assert context.summary["diagnostics"]["n_z_extrapolated"] == 1


def test_self_renormalization_completes_the_authored_long_distance_ansatz(tmp_path) -> None:
    from lamet_agent.kernels import load_renormalization_kernel
    from lamet_agent.stages.renormalization._apply import run
    from lamet_agent.stages.renormalization._inspection import run as inspect
    from lamet_agent.stages.renormalization.physics import zmsbar_log

    spacing = 0.1
    z_factor = np.array([0.05, 0.1, 0.15, 0.2])
    z_target = np.array([0.0, 0.1, 0.2, 0.3])
    k = 0.6551255749279999
    d = -0.08
    m0 = -0.02
    lambda_qcd = 0.1
    scale = 2.0
    baseline = log_m(z_factor, spacing, k=k, lambda_qcd_gev=lambda_qcd, d=d, n_f=3, scale_gev=scale) + m0 * z_factor
    factor_values = np.exp(baseline + 0.4 * z_factor**2 * spacing)
    factor = EnsembleData(
        None,
        "bootstrap",
        [[factor_values], [factor_values]],
        ["a", "z"],
        {"a": [spacing], "z": z_factor.tolist()},
        attrs={"coord_unit": "fm", "d": d, "m0_gev": m0, "k": k, "n_f": 3, "scale_gev": scale},
    )
    target = EnsembleData(
        _ensemble(spacing),
        "bootstrap",
        [np.ones(4), np.ones(4)],
        ["z"],
        {"z": z_target.tolist()},
        attrs={"coord_unit": "fm"},
    )
    params = {
        "type": "apply",
        "scheme": "ratio",
        "strategy": "self_renormalization",
        "kernel_id": "z_msbar_da_nlo",
        "kernel_parameters": {},
        "normalization": False,
        "mu": scale,
        "LambdaQCD_gev": lambda_qcd,
        "d": d,
        "m0_gev": m0,
        "z_coverage_policy": "extrapolate",
    }
    context = ToolContext(
        {"metadata": {"target_observable": "pdf", "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "renormalization",
        "apply",
        params,
        {"target": target, "zR": factor},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    inspect(context)
    run(context)
    assert context.output.coords["z"] == z_target.tolist()
    assert np.all(np.isfinite(context.output.values))
    assert context.output.attrs["kernel_id"] == "z_msbar_da_nlo"
    assert context.output.attrs["z_coverage_policy"] == "extrapolate"
    assert context.output.attrs["n_z_extrapolated"] == 1
    assert context.output.attrs["z_extrapolation_method"] == "quadratic_f1_tail"
    expected_factor = np.ones_like(z_target)
    expected_factor[1:] = np.exp(
        log_m(
            z_target[1:],
            spacing,
            k=k,
            lambda_qcd_gev=lambda_qcd,
            d=d,
            n_f=3,
            scale_gev=scale,
        )
        + m0 * z_target[1:]
        + 0.4 * z_target[1:] ** 2 * spacing
    )
    expected_factor[1:] *= np.exp(
        zmsbar_log(load_renormalization_kernel("z_msbar_da_nlo"), z_target[1:], scale_gev=scale)
    )
    expected = np.tile(1.0 / expected_factor[None, :], (target.n_sample, 1))
    np.testing.assert_allclose(context.output.values, expected, rtol=1e-10, atol=1e-12)


def test_every_migrated_kernel_owns_its_callable_and_formula_document() -> None:
    from lamet_agent.kernels import implementation

    root = Path(__file__).parents[2] / "lamet_agent" / "kernels"
    kernel_ids = list_kernel_ids()
    assert len(kernel_ids) == 46
    assert set(kernel_ids) == {path.stem for path in root.glob("*.md")}
    for kernel_id in kernel_ids:
        assert not hasattr(implementation, kernel_id)
        source = (root / f"{kernel_id}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert any(isinstance(node, ast.FunctionDef) and node.name == "kernel" for node in tree.body)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "lamet_agent.kernels.implementation":
                assert all(alias.name != kernel_id and alias.asname != "_implementation" for alias in node.names)
        parameters = {"zs_fm": 0.2} if "_hybrid_" in kernel_id else {}
        accepted, required = inspect_callable(load_kernel(kernel_id), parameter_values=parameters)
        assert set(required).issubset(parameters)
        assert "zs_fm" in accepted if "_hybrid_" in kernel_id else "zs_fm" not in accepted
        document = load_kernel_document(kernel_id)
        assert document and kernel_id in document
    with pytest.raises(ValueError, match="not available"):
        load_kernel("implementation")


def test_kernel_implementation_owns_the_unit_conversion_constant() -> None:
    from lamet_agent import data
    from lamet_agent.kernels.implementation import GEV_FM

    assert HBAR_C_GEV_FM == 0.1973269804
    assert GEV_FM is HBAR_C_GEV_FM
    assert "HBAR_C_GEV_FM" not in data.__all__
    assert "GEV_FM" not in data.__all__
    assert not hasattr(data, "HBAR_C_GEV_FM")
    assert not hasattr(data, "GEV_FM")


def test_migrated_kernel_code_has_no_legacy_imports_or_embedded_documentation() -> None:
    root = Path(__file__).parents[2] / "lamet_agent" / "kernels"
    paths = [root / "implementation.py", *(root / f"{kernel_id}.py" for kernel_id in list_kernel_ids())]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("lamet_agent."):
                assert node.module == "lamet_agent.kernels.implementation"
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node, clean=False) is None
        comments = [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        ]
        assert comments == []


def test_ratio_kernel_executes_the_migrated_physics_function() -> None:
    kernel = load_kernel("quark_pdf_cg_gt_ratio_nlo")
    grid = np.array([-0.5, 0.5])
    matrix = kernel(grid, grid, momentum_gev=2.0, scale_gev=2.0)
    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))


def test_extrapolation_fit_uses_reference_median_covariance(monkeypatch) -> None:
    rng = np.random.default_rng(12)
    x = [-0.2, 0.2]
    physical = np.array([0.45, 0.35])
    data = []
    for index, spacing in enumerate((0.06, 0.08, 0.10, 0.12)):
        center = physical + 0.3 * spacing / 0.1
        samples = [center + rng.normal(0.0, 0.003, 2) for _ in range(60)]
        attrs = {
            "momentum_gev": 2.0,
            "resample_id": f"ensemble-{index}",
            "sample_error_mode": "one_sigma",
        }
        data.append(
            EnsembleData(
                _ensemble(spacing, f"ensemble-{index}", L_s=32, m_pi=0.2),
                "bootstrap",
                samples,
                ["x"],
                {"x": x},
                attrs=attrs,
            )
        )
    result, diagnostics = fit_candidate(
        data,
        ["a"],
        0.135,
        {"mean": 0.0, "sdev": 2.0},
        x_range=(-0.2, 0.2),
        x_independent_terms=["a"],
    )
    assert result.dims == ["x"]
    assert result.resample == "bootstrap"
    assert result.n_sample == 60
    assert np.allclose(result.mean, physical, atol=2e-2)
    assert diagnostics["dof"] > 0
    assert json.loads(result.attrs["x_independent_terms"]) == ["a"]


def test_stage_fourier_uniform_grid_keeps_full_endpoint_weights() -> None:
    z = [-0.1, 0.0, 0.1]
    data = EnsembleData(None, "bootstrap", [np.ones(3), np.ones(3)], ["z"], {"z": z})
    result = stage_fourier_transform(
        data,
        [0.0],
        momentum_gev=HBAR_C_GEV_FM,
        prefactor="pz_over_2pi",
        workers=1,
    )
    assert np.allclose(result.values[:, 0], 0.3 / (2.0 * np.pi))
    assert result.attrs["quadrature"] == "reference_uniform_rectangle"


def test_renormalization_loader_maps_reference_m_pi_metadata(tmp_path: Path) -> None:
    import xarray as xr

    path = tmp_path / "reference.nc"
    array = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=["resample", "a", "z"],
        coords={"resample": [0, 1], "a": [0.06, 0.12], "z": [0.1, 0.2]},
        attrs={
            "ensemble": '{"series":"MILC","id":"reference","a_s":0.12,"a_t":0.12,"L_s":0,"L_t":0,"m_pi":0.0}',
            "resample": "bootstrap",
        },
        name="reference",
    )
    array.to_netcdf(path, format="NETCDF4")
    loaded = load_renormalization_data(path)
    assert loaded.ensemble is not None
    assert loaded.ensemble.m_pi == 0.0
    assert loaded.dims == ["a", "z"]
