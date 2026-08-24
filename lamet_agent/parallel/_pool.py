"""Shared grouped process execution for independent resample tasks."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable

from tqdm import tqdm


_MISSING = object()


def _run_batch(
    function: Callable[[Any], Any],
    indexed_tasks: list[tuple[int, Any]],
) -> list[tuple[int, Any]]:
    """Run one worker-local task batch while retaining original indices."""
    return [(index, function(task)) for index, task in indexed_tasks]


def _balanced_batches(tasks: list[Any], workers: int) -> list[list[tuple[int, Any]]]:
    """Split indexed tasks into at most one balanced batch per worker."""
    batch_count = min(workers, len(tasks))
    width, remainder = divmod(len(tasks), batch_count)
    batches = []
    start = 0
    for batch_index in range(batch_count):
        stop = start + width + (batch_index < remainder)
        batches.append(list(enumerate(tasks[start:stop], start=start)))
        start = stop
    return batches


class _ParallelPool:
    """Lazily own one spawn-based worker pool and grouped ordered mapping."""

    def __init__(self, workers: int) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.workers = workers
        self._executor: ProcessPoolExecutor | None = None
        self._previous_omp_threads: object | str = _MISSING

    def _start(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._previous_omp_threads = os.environ.get("OMP_NUM_THREADS", _MISSING)
            os.environ["OMP_NUM_THREADS"] = "1"
            try:
                self._executor = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            except Exception:
                self._restore_environment()
                raise
        return self._executor

    def map(
        self,
        function: Callable[[Any], Any],
        tasks: Iterable[Any],
        *,
        description: str,
        unit: str,
    ) -> list[Any]:
        """Run balanced worker batches with sample progress and ordered output."""
        task_list = list(tasks)
        if not task_list:
            return []
        if self.workers == 1:
            return [
                function(task)
                for task in tqdm(task_list, desc=description, unit=unit)
            ]
        executor = self._start()
        batches = _balanced_batches(task_list, self.workers)
        futures: dict[Future[Any], int] = {
            executor.submit(_run_batch, function, batch): len(batch)
            for batch in batches
        }
        results: list[Any] = [None] * len(task_list)
        try:
            with tqdm(total=len(task_list), desc=description, unit=unit) as progress:
                for future in as_completed(futures):
                    batch_results = future.result()
                    for index, result in batch_results:
                        results[index] = result
                    progress.update(futures[future])
        except Exception:
            for future in futures:
                future.cancel()
            raise
        return results

    def _restore_environment(self) -> None:
        previous = self._previous_omp_threads
        if previous is _MISSING:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = str(previous)
        self._previous_omp_threads = _MISSING

    def close(self) -> None:
        """Close any started workers and restore the parent OpenMP setting."""
        executor = self._executor
        self._executor = None
        try:
            if executor is not None:
                executor.shutdown()
        finally:
            if self._previous_omp_threads is not _MISSING:
                self._restore_environment()

    def __enter__(self) -> "_ParallelPool":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
