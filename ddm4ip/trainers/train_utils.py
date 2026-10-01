import contextlib
import time


def get_next_batch(loader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


class Timer:
    """Accumulated wall-clock time of the measured blocks since the last read."""

    def __init__(self, cpu_only: bool = False):
        self.cpu_only = cpu_only
        self._cpu_time = 0

    @contextlib.contextmanager
    def measure(self):
        start_cpu = time.time()
        try:
            yield
        finally:
            self._cpu_time += time.time() - start_cpu

    def cuda_time(self) -> float:
        return self.cpu_time()

    def cpu_time(self) -> float:
        out = self._cpu_time
        self._cpu_time = 0
        return out
