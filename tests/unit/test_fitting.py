"""Checks for deterministic, sample-parallel neo nonlinear fitting.

Purpose: verify the lsqfit-like API, bootstrap seeding, and worker-independent
results. Inputs are deterministic toy ensembles; outputs are fitted parameter
samples. Example: ``pytest tests/unit/test_neo_fitting.py``.
"""

from __future__ import annotations

import os
from concurrent.futures import Future

import gvar as gv
import numpy as np
import pytest

from lamet_agent.data import EnsembleData
import lamet_agent.parallel.fitting as fitting
import lamet_agent.parallel._pool as parallel_pool
from lamet_agent.parallel import FitNumericalError, nonlinear_fit
from lamet_agent.parallel._pool import _ParallelPool


def linear_model(x: np.ndarray, p: gv.BufferDict) -> np.ndarray:
    """Evaluate a straight line for the process-pool regression test."""
    return p["intercept"] + p["slope"] * x


def test_bootstrap_fit_is_seeded_and_worker_independent() -> None:
    x = np.linspace(-1.0, 1.0, 8)
    rng = np.random.default_rng(9)
    values = [1.2 + 0.7 * x + rng.normal(0.0, 0.03, x.size) for _ in range(40)]
    data = EnsembleData(None, "raw", values, ["x"], {"x": x.tolist()})
    prior = gv.BufferDict({"intercept": gv.gvar(0.0, 5.0), "slope": gv.gvar(0.0, 5.0)})
    serial = nonlinear_fit((x, data), linear_model, prior, resampling="bootstrap", n_resample=12, seed=17, workers=1)
    parallel = nonlinear_fit((x, data), linear_model, prior, resampling="bootstrap", n_resample=12, seed=17, workers=2)
    changed = nonlinear_fit((x, data), linear_model, prior, resampling="bootstrap", n_resample=12, seed=18, workers=1)
    serial_slopes = np.asarray([sample["slope"] for sample in serial.samples], dtype=float)
    parallel_slopes = np.asarray([sample["slope"] for sample in parallel.samples], dtype=float)
    changed_slopes = np.asarray([sample["slope"] for sample in changed.samples], dtype=float)
    np.testing.assert_allclose(parallel_slopes, serial_slopes, rtol=0.0, atol=1e-12)
    assert not np.allclose(changed_slopes, serial_slopes)
    assert np.isclose(gv.mean(serial.p["slope"]), 0.7, atol=0.03)
    assert serial.resample == "bootstrap"


def test_injected_parallel_pool_is_used_for_sample_fits(monkeypatch) -> None:
    class SharedParallel:
        def __init__(self) -> None:
            self.calls = 0

        def map(self, function, tasks, *, description, unit):
            self.calls += 1
            assert description == "Sample fits"
            assert unit == "fit"
            return [function(task) for task in tasks]

    data = EnsembleData(None, "bootstrap", [[1.0], [1.1]], ["x"], {"x": [0]})
    prior = gv.BufferDict({"amplitude": gv.gvar(1.0, 1.0)})
    shared = SharedParallel()
    nonlinear_fit(
        data,
        lambda p: np.asarray([p["amplitude"]]),
        prior,
        workers=4,
        _parallel=shared,
    )
    assert shared.calls == 1


def test_center_fit_wraps_zero_division_as_a_numerical_candidate_failure() -> None:
    data = EnsembleData(None, "bootstrap", [[1.0], [1.1]], ["x"], {"x": [0]})
    prior = gv.BufferDict({"amplitude": gv.gvar(1.0, 1.0)})

    def singular_model(_parameters):
        raise ZeroDivisionError("float division")

    with pytest.raises(FitNumericalError, match="sample-average fit failed.*ZeroDivisionError"):
        nonlinear_fit(data, singular_model, prior, workers=1)


def test_tolerated_sample_failure_preserves_sample_alignment(monkeypatch) -> None:
    data = EnsembleData(None, "bootstrap", [[1.0], [1.1]], ["x"], {"x": [0]})
    prior = gv.BufferDict({"amplitude": gv.gvar(1.0, 1.0)})
    successful = gv.BufferDict({"amplitude": 1.0})

    class FailedSampleParallel:
        def map(self, _function, _tasks, *, description, unit):
            assert description == "Sample fits"
            assert unit == "fit"
            return [
                (successful, None, {"chi2": 1.0, "dof": 1.0, "Q": 0.5, "logGBF": 0.0}, None),
                (None, "ZeroDivisionError: float division", None, None),
            ]

    result = nonlinear_fit(
        data,
        lambda p: np.asarray([p["amplitude"]]),
        prior,
        workers=1,
        tolerate_sample_failures=True,
        _parallel=FailedSampleParallel(),
    )
    assert result.samples == (successful, None)
    assert result.sample_errors == (None, "ZeroDivisionError: float division")
    assert result.sample_diagnostics[0]["Q"] == 0.5
    assert result.n_failed_samples == 1


def test_sample_posterior_capture_is_explicit_and_indexed() -> None:
    data = EnsembleData(None, "bootstrap", [[1.0], [1.1], [0.9]], ["x"], {"x": [0]})
    prior = gv.BufferDict({"amplitude": gv.gvar(1.0, 1.0)})
    default = nonlinear_fit(data, lambda p: np.asarray([p["amplitude"]]), prior, workers=1)
    captured = nonlinear_fit(
        data,
        lambda p: np.asarray([p["amplitude"]]),
        prior,
        workers=1,
        capture_sample_posteriors=(0,),
    )
    assert default.sample_posteriors == (None, None, None)
    assert captured.sample_posteriors[0] is not None
    assert captured.sample_posteriors[1:] == (None, None)
    assert isinstance(captured.sample_posteriors[0]["amplitude"], gv.GVar)


def test_center_mode_averages_raw_source_without_scheduling_resamples(monkeypatch) -> None:
    data = EnsembleData(None, "raw", [[1.0], [1.1]], ["x"], {"x": [0]})
    prior = gv.BufferDict({"amplitude": gv.gvar(1.0, 1.0)})

    def reject_sample_fits(*args, **kwargs):
        raise AssertionError("sample fits were scheduled during candidate tuning")

    monkeypatch.setattr(fitting, "_ParallelPool", reject_sample_fits)
    result = nonlinear_fit(data, lambda p: np.asarray([p["amplitude"]]), prior, workers=4, mode="center")
    assert result.samples == ()
    assert result.sample_errors == ()
    assert result.n_failed_samples == 0
    assert result.resample == "raw"


def _worker_omp_threads(_value: int) -> str | None:
    return os.environ.get("OMP_NUM_THREADS")


def test_parallel_pool_sets_worker_omp_threads_and_restores_parent() -> None:
    previous = os.environ.get("OMP_NUM_THREADS")
    with _ParallelPool(2) as parallel:
        assert parallel.map(
            _worker_omp_threads,
            [0, 1],
            description="OMP check",
            unit="task",
        ) == ["1", "1"]
    assert os.environ.get("OMP_NUM_THREADS") == previous


def test_parallel_pool_submits_one_balanced_batch_per_worker(monkeypatch) -> None:
    class ImmediateExecutor:
        def __init__(self) -> None:
            self.batch_sizes = []

        def submit(self, function, worker_function, batch):
            self.batch_sizes.append(len(batch))
            future = Future()
            future.set_result(function(worker_function, batch))
            return future

        def shutdown(self) -> None:
            return None

    executor = ImmediateExecutor()
    monkeypatch.setattr(
        parallel_pool,
        "ProcessPoolExecutor",
        lambda **_kwargs: executor,
    )
    monkeypatch.setattr(
        parallel_pool.multiprocessing,
        "get_context",
        lambda _method: object(),
    )
    with _ParallelPool(3) as parallel:
        results = parallel.map(abs, [-1, -2, -3, -4, -5, -6, -7, -8], description="Batched", unit="task")
    assert results == [1, 2, 3, 4, 5, 6, 7, 8]
    assert executor.batch_sizes == [3, 3, 2]


def test_jackknife_fit_rejects_bootstrap_only_arguments() -> None:
    x = np.arange(4.0)
    values = [1.0 + 0.5 * x + offset for offset in (0.0, 0.1, -0.1, 0.05)]
    data = EnsembleData(None, "raw", values, ["x"], {"x": x.tolist()})
    prior = gv.BufferDict({"intercept": gv.gvar(0.0, 5.0), "slope": gv.gvar(0.0, 5.0)})
    with pytest.raises(ValueError, match="only valid for bootstrap"):
        nonlinear_fit((x, data), linear_model, prior, resampling="jackknife", seed=1)


def test_resamples_mode_rejects_unresampled_source_data() -> None:
    data = EnsembleData(None, "raw", [[1.0], [1.1]], ["x"], {"x": [0]})
    prior = gv.BufferDict({"amplitude": gv.gvar(1.0, 1.0)})
    with pytest.raises(ValueError, match="requires jackknife or bootstrap"):
        nonlinear_fit(data, lambda p: np.asarray([p["amplitude"]]), prior)
