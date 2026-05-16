import queue
import threading
import time
from typing import cast

MetricItem = tuple[str, float, int]


class AsyncMetricLogger:
    """
    Async Metric Logger for MLFLow logger.
    """

    def __init__(
        self, logger=None, max_batch_size: int = 256, close_timeout: float = 30.0
    ):
        self._logger = logger
        self._queue: queue.Queue[MetricItem | object] | None = None
        self._thread: threading.Thread | None = None
        self._stop = object()
        self._error: Exception | None = None
        self._closed = False
        self._max_batch_size = max_batch_size
        self._close_timeout = close_timeout

        if logger:
            self._queue = queue.Queue()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _raise_if_error(self) -> None:
        if self._error is None:
            return
        raise RuntimeError("Async metric logger worker failed") from self._error

    def _log_batch(self, batch: list[MetricItem]) -> None:
        log_metrics = getattr(self._logger, "log_metrics", None)
        if callable(log_metrics):
            log_metrics(batch)
            return

        for name, value, step in batch:
            self._logger.log_metric(name, value, step)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            batch = []
            saw_stop = False
            try:
                if item is self._stop:
                    saw_stop = True
                else:
                    batch.append(cast(MetricItem, item))
                    while len(batch) < self._max_batch_size:
                        try:
                            next_item = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if next_item is self._stop:
                            saw_stop = True
                            break
                        batch.append(cast(MetricItem, next_item))

                    if batch:
                        self._log_batch(batch)
            except Exception as exc:  # noqa: BLE001
                self._error = exc
            finally:
                tasks_done = len(batch)
                if item is self._stop:
                    tasks_done = 1
                elif saw_stop:
                    tasks_done += 1
                for _ in range(tasks_done):
                    self._queue.task_done()

            if saw_stop:
                break

    def log_metric(self, name: str, value: float, step: int) -> None:
        if self._queue is None or self._closed:
            return
        self._raise_if_error()
        self._queue.put((name, value, step))

    def flush(self, timeout: float | None = None) -> bool:
        if self._queue is None:
            return True

        self._raise_if_error()
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            self._raise_if_error()
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

        self._raise_if_error()
        return True

    def close(self) -> bool:
        if self._queue is None or self._thread is None:
            return True
        if self._closed:
            return not self._thread.is_alive()

        self._closed = True
        drained = self.flush(timeout=self._close_timeout)
        self._queue.put(self._stop)
        self._thread.join(timeout=self._close_timeout)
        self._raise_if_error()
        return drained and not self._thread.is_alive()
