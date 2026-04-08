import time
import contextlib
from typing import Optional, Callable

class Timer:
    """
    A simple context manager to measure execution time.
    
    Usage:
        with Timer() as t:
            do_something()
        print(f"Elapsed: {t.elapsed_ms:.2f}ms")
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def stop(self):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __str__(self):
        if self.name:
            return f"{self.name} took {self.elapsed_ms:.2f}ms"
        return f"Took {self.elapsed_ms:.2f}ms"

@contextlib.asynccontextmanager
async def async_timer(name: Optional[str] = None, callback: Optional[Callable[[float], None]] = None):
    """
    An asynchronous context manager to measure execution time.
    
    Usage:
        async with async_timer("Task") as t:
            await do_something()
    """
    start_time = time.perf_counter()
    timer_obj = Timer(name)
    try:
        yield timer_obj
    finally:
        end_time = time.perf_counter()
        timer_obj.elapsed_ms = (end_time - start_time) * 1000
        if callback:
            callback(timer_obj.elapsed_ms)
