import queue
import threading


class AsyncMetricLogger:
    """
    Async Metric Logger for MLFLow logger.
    """

    def __init__(self, logger=None):
        self._logger = logger
        self._queue = None
        self._thread = None
        self._stop = object()

        if logger:
            self._queue = queue.Queue()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._stop:
                break
            name, value, step = item
            self._logger.log_metric(name, value, step)

    def log_metric(self, name: str, value: float, step: int) -> None:
        if self._queue is None:
            return
        self._queue.put((name, value, step))

    def close(self) -> None:
        if self._queue is None or self._thread is None:
            return
        self._queue.put(self._stop)
        self._thread.join()
