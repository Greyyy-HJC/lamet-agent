"""Unit tests for the unified resampling helpers in core/resampling.py."""

from __future__ import annotations

import gvar as gv
import numpy as np
import pytest

from lamet_agent.core.data import EnsembleData
from lamet_agent.core.resampling import (
    add_error_to_sample,
    bs_ls_avg_percentile,
    recenter_sample_values,
    resample_config_samples,
    sample_mean_and_sdev,
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


def test_sample_mean_and_sdev_propagates_nan_without_filtering() -> None:
    samples = np.array(
        [
            [1.0, np.nan],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )
    mean, err = sample_mean_and_sdev(samples[:, 0], mode="bs")
    assert mean == pytest.approx(2.0)
    assert err > 0.0
    nan_mean, nan_err = sample_mean_and_sdev(samples[:, 1], mode="bs")
    assert np.isnan(nan_mean)
    assert np.isnan(nan_err)


def test_sample_mean_and_sdev_handles_matrix_input() -> None:
    samples = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    mean, sdev = sample_mean_and_sdev(samples, mode="jk", axis=0)
    assert mean.shape == (2,)
    assert sdev.shape == (2,)
    assert mean[0] == pytest.approx(2.0)


def test_removed_bs_percentile_mode_is_rejected() -> None:
    data = np.arange(12.0).reshape(4, 3)
    with pytest.raises(ValueError, match="resample_mode"):
        resample_config_samples(data, mode="bs_percentile", n_boot=5, seed=1)
    with pytest.raises(ValueError, match="sample_error_mode"):
        samples_to_gvar(data, mode="bs", sample_error_mode="bs_percentile")


def test_add_error_to_sample_matches_recenter_on_toy_data() -> None:
    samples = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
    avg = samples_to_gvar(samples, mode="bs", axis=0)
    with_errors = add_error_to_sample(samples, mode="bs", axis=0)
    for idx, row in enumerate(with_errors):
        expected = recenter_sample_values(samples[idx], avg)
        assert np.allclose(np.asarray(gv.mean(row)), np.asarray(gv.mean(expected)))
        assert np.allclose(np.asarray(gv.sdev(row)), np.asarray(gv.sdev(expected)))


def test_add_error_to_sample_median_uses_diagonal_errors() -> None:
    samples = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    with_errors = add_error_to_sample(samples, mode="bs", sample_error_mode="median", axis=0)
    avg = samples_to_gvar(samples, mode="bs", sample_error_mode="median", axis=0)
    expected_sdev = np.asarray(gv.sdev(avg), dtype=float)
    for row in with_errors:
        assert np.allclose(np.asarray(gv.sdev(row)), expected_sdev)


def test_ensemble_data_gvar_respects_sample_error_mode_attr() -> None:
    data = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([1.0]), np.array([2.0]), np.array([100.0])],
        dims=("z",),
        coords={"z": [0.0]},
        attrs={"sample_error_mode": "median"},
    )
    assert np.asarray(gv.mean(data.gvar), dtype=float)[0] == pytest.approx(2.0)
