"""Focused behavior checks for the migrated neo numerical base."""

from __future__ import annotations

import numpy as np
import gvar as gv
import pytest

from lamet_agent.data import EnsembleData


def test_sample_bearing_data_keeps_leading_resample_dimension() -> None:
    data = EnsembleData(None, "bootstrap", [np.array([1.0, 2.0]), np.array([2.0, 4.0])], ["x"], {"x": [0.0, 1.0]}, attrs={"resample_id": "same"}, name="toy")
    assert data.array.dims == ("resample", "x")
    assert data.n_sample == 2
    assert data.coords == {"x": [0.0, 1.0]}
    assert np.allclose(data.mean, [1.5, 3.0])


def test_netcdf_roundtrip(tmp_path) -> None:
    data = EnsembleData(None, "bootstrap", [np.array([1.0, 2.0]), np.array([2.0, 4.0])], ["x"], {"x": [0.0, 1.0]}, name="toy")
    netcdf = tmp_path / "toy.nc"
    data.to_netcdf(netcdf)
    restored = EnsembleData.from_netcdf(netcdf)
    assert restored.resample == data.resample
    assert np.allclose(restored.values, data.values)


@pytest.mark.parametrize("resample", ["raw", "jackknife", "bootstrap"])
def test_gvar_preserves_physical_shape_and_covariance(resample: str) -> None:
    samples = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[4.0, 8.0], [12.0, 16.0]],
        ]
    )
    data = EnsembleData(None, resample, list(samples), ["x", "y"], {"x": [0, 1], "y": [0, 1]})
    result = data.gvar
    flat = samples.reshape(samples.shape[0], -1)
    if resample == "raw":
        expected_covariance = np.cov(flat, rowvar=False, bias=False) / samples.shape[0]
    elif resample == "jackknife":
        expected_covariance = np.cov(flat, rowvar=False, bias=True) * (samples.shape[0] - 1)
    else:
        expected_covariance = np.cov(flat, rowvar=False, bias=False)
    assert result.shape == samples.shape[1:]
    np.testing.assert_allclose(gv.mean(result), samples.mean(axis=0))
    np.testing.assert_allclose(gv.evalcov(result).reshape(4, 4), expected_covariance)


def test_gvar_supports_scalar_and_single_sample_data() -> None:
    scalar = EnsembleData(None, "raw", [np.asarray(1.0), np.asarray(3.0)], [], {})
    assert np.shape(scalar.gvar) == ()
    assert gv.mean(scalar.gvar) == 2.0
    singleton = EnsembleData(None, "bootstrap", [np.asarray([3.0, 5.0])], ["x"], {"x": [0, 1]})
    np.testing.assert_allclose(gv.mean(singleton.gvar), [3.0, 5.0])
    np.testing.assert_allclose(gv.sdev(singleton.gvar), [0.0, 0.0])


def test_gvar_requires_an_explicit_real_or_imaginary_component() -> None:
    data = EnsembleData(None, "bootstrap", [np.asarray([1.0 + 2.0j]), np.asarray([2.0 + 4.0j])], ["x"], {"x": [0]})
    with pytest.raises(TypeError, match="select .real or .imag"):
        _ = data.gvar
    np.testing.assert_allclose(gv.mean(data.real.gvar), [1.5])
    np.testing.assert_allclose(gv.mean(data.imag.gvar), [3.0])


@pytest.mark.parametrize(
    ("resample", "factor"),
    [("raw", 0.5), ("jackknife", np.sqrt(3.0)), ("bootstrap", 1.0)],
)
def test_gvar_median_uses_resampling_scaled_percentile_errors_without_correlations(
    resample: str, factor: float
) -> None:
    samples = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [100.0, 3.0]])
    data = EnsembleData(None, resample, list(samples), ["x"], {"x": [0, 1]})
    result = data.gvar_median
    low, center, high = np.percentile(
        samples, [50 - 34.1344746, 50, 50 + 34.1344746], axis=0
    )
    expected = 0.5 * (high - low) * factor
    np.testing.assert_allclose(gv.mean(result), center)
    np.testing.assert_allclose(gv.sdev(result), expected)
    covariance = gv.evalcov(result)
    assert covariance[0, 1] == covariance[1, 0] == 0.0


def test_average_selects_data_owned_uncertainty_semantics() -> None:
    samples = np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [100.0, 6.0]])
    data = EnsembleData(None, "bootstrap", list(samples), ["x"], {"x": [0, 1]})

    covariance = data.average("covariance")
    diagonal = data.average("mean")
    median = data.average("median")

    np.testing.assert_allclose(gv.mean(covariance), gv.mean(data.gvar))
    np.testing.assert_allclose(gv.mean(diagonal), gv.mean(data.gvar))
    np.testing.assert_allclose(gv.sdev(diagonal), gv.sdev(data.gvar))
    assert gv.evalcov(covariance)[0, 1] != 0.0
    assert gv.evalcov(diagonal)[0, 1] == 0.0
    np.testing.assert_allclose(gv.mean(median), gv.mean(data.gvar_median))
    np.testing.assert_allclose(gv.sdev(median), gv.sdev(data.gvar_median))
    with pytest.raises(ValueError, match="average mode"):
        data.average("unsupported")


def test_aligned_ensemble_data_preserves_ratio_correlations() -> None:
    numerator = EnsembleData(None, "raw", [np.array([2.0, 4.0]), np.array([4.0, 8.0]), np.array([6.0, 10.0])], ["z"], {"z": [0.0, 1.0]})
    denominator = EnsembleData(None, "raw", [np.array([1.0, 2.0]), np.array([2.0, 4.0]), np.array([3.0, 5.0])], ["z"], {"z": [0.0, 1.0]})
    sampled_numerator = numerator.bootstrap(12, seed=7)
    sampled_denominator = denominator.bootstrap(12, seed=7)
    ratio = sampled_numerator.div(sampled_denominator)
    assert np.allclose(ratio.values[:, 0], 2.0)
    assert np.allclose(ratio.values[:, 1], 2.0)
