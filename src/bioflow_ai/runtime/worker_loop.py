"""Persistent per-process asyncio event loop for synchronous Celery workers."""

import asyncio
import atexit
import contextvars
import os
import threading
from collections.abc import Coroutine
from typing import Any


class _RunnerState(threading.local):
    pid: int | None = None
    runner: asyncio.Runner | None = None


_state = _RunnerState()


def _get_runner() -> asyncio.Runner:
    current_pid = os.getpid()
    if _state.pid != current_pid:
        # Never close an inherited runner after os.fork(); doing so corrupts parent fds.
        _state.runner = None
        _state.pid = current_pid

    runner = _state.runner
    if runner is not None:
        try:
            if runner.get_loop().is_closed():
                runner = None
        except RuntimeError:
            runner = None

    if runner is None:
        runner = asyncio.Runner()
        _state.runner = runner
        _state.pid = current_pid
    return runner


def run_in_worker_loop[T](coro: Coroutine[Any, Any, T]) -> T:
    """Execute one coroutine on this worker thread's persistent event loop.

    Reusing one open event loop across sequential Celery tasks keeps cached
    ``httpx.AsyncClient`` connection pools and LLM clients valid, while running
    each call in a fresh ``contextvars.Context`` prevents state leakage.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        coro.close()
        msg = 'run_in_worker_loop() cannot be called from a running event loop'
        raise RuntimeError(msg)

    runner = _get_runner()
    return runner.run(coro, context=contextvars.copy_context())


def close_worker_loop() -> None:
    """Close this worker thread's persistent event loop if owned by this process."""
    runner = _state.runner
    pid = _state.pid
    _state.runner = None
    _state.pid = None
    if runner is not None and pid == os.getpid():
        runner.close()


def reset_worker_loop() -> None:
    """Reset event loop state cleanly after worker process initialization or fork."""
    if _state.pid == os.getpid():
        close_worker_loop()
        return
    _state.runner = None
    _state.pid = os.getpid()


atexit.register(close_worker_loop)
