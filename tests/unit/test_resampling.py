"""Unit tests for the unified resampling helpers in core/resampling.py."""

from __future__ import annotations

import gvar as gv
import numpy as np
import pytest

from lamet_agent.core.data import EnsembleData
from lamet_agent.core.resampling import (
    add_error_to_sample,
    add_error_to_sample_percentile,
    average_mode_from_ensemble,
    bs_ls_avg_percentile,
    ensemble_average_method,
    recenter_sample_values,
    resample_config_samples,
    resample_generation_mode,
    sample_mean_and_sdev,
    sample_mean_err,
    samples_to_gvar,
)


def test_bs_ls_avg_percentile_uses_median_and_16_84_width() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=2.0, scale=0.5, size=(400, 3))
    avg = bs_ls_avg_percentile(samples, axis=0)
    expected_mid = np.median(samples, axis=0)
    p16, p84 = np.percentile(samples, [16, 84], axis=0)
    expected_sdev = 0.5 * (p84 - p16)
    assert np.allclose(np.asarray(gv.mean(avg)), expected_mid)
    assert np.allclose(np.asarray(gv.sdev(avg)), expected_sdev)


def test_sample_mean_err_propagates_nan_without_filtering() -> None:
    samples = np.array(
        [
            [1.0, np.nan],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )
    mean, err = sample_mean_err(samples[:, 0], mode="bs")
    assert mean == pytest.approx(2.0)
    assert err > 0.0
    nan_mean, nan_err = sample_mean_err(samples[:, 1], mode="bs")
    assert np.isnan(nan_mean)
    assert np.isnan(nan_err)


def test_sample_mean_and_sdev_handles_matrix_input() -> None:
    samples = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    mean, sdev = sample_mean_and_sdev(samples, mode="jk", axis=0)
    assert mean.shape == (2,)
    assert sdev.shape == (2,)
    assert mean[0] == pytest.approx(2.0)


def test_bs_percentile_generation_mode_maps_to_bs() -> None:
    assert resample_generation_mode("bs_percentile") == "bs"
    data = np.arange(12.0).reshape(4, 3)
    bs_samples, _ = resample_config_samples(data, mode="bs", n_boot=5, seed=1)
    pct_samples, _ = resample_config_samples(data, mode="bs_percentile", n_boot=5, seed=1)
    assert np.array_equal(bs_samples, pct_samples)


def test_add_error_to_sample_matches_recenter_on_toy_data() -> None:
    samples = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
    avg = samples_to_gvar(samples, mode="bs", axis=0)
    with_errors = add_error_to_sample(samples, mode="bs", axis=0)
    for idx, row in enumerate(with_errors):
        expected = recenter_sample_values(samples[idx], avg)
        assert np.allclose(np.asarray(gv.mean(row)), np.asarray(gv.mean(expected)))
        assert np.allclose(np.asarray(gv.sdev(row)), np.asarray(gv.sdev(expected)))


def test_add_error_to_sample_percentile_uses_diagonal_errors() -> None:
    samples = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    with_errors = add_error_to_sample_percentile(samples, mode="bs", axis=0)
    avg = bs_ls_avg_percentile(samples, axis=0)
    expected_sdev = np.asarray(gv.sdev(avg), dtype=float)
    for row in with_errors:
        assert np.allclose(np.asarray(gv.sdev(row)), expected_sdev)


def test_average_mode_from_ensemble_round_trip() -> None:
    data = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([1.0 + 1j]), np.array([2.0 + 2j])],
        dims=("z",),
        coords={"z": [0.0]},
        attrs={"average_method": ensemble_average_method("bs_percentile")},
    )
    assert average_mode_from_ensemble(data) == "bs_percentile"

    cov_data = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([1.0 + 1j]), np.array([2.0 + 2j])],
        dims=("z",),
        coords={"z": [0.0]},
        attrs={"average_method": "covariance"},
    )
    assert average_mode_from_ensemble(cov_data) == "bs"
