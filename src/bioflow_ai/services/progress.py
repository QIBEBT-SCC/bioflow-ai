"""Optional presentation callbacks shared by long-running services."""

from collections.abc import Callable, Iterable
from typing import Any, Protocol

ProgressCallback = Callable[[str, Any], None]


class ProgressIterator(Protocol):
    """Caller-provided presentation around a service's iterable."""

    def __call__[T](self, items: Iterable[T]) -> Iterable[T]:
        """Wrap items without changing their order or values."""
        ...


def with_progress[T](items: Iterable[T], progress: ProgressIterator | None) -> Iterable[T]:
    """Apply optional iteration presentation supplied by the caller."""
    return items if progress is None else progress(items)


def emit_progress(callback: ProgressCallback | None, event: str, payload: Any) -> None:
    """Deliver an event when a caller supplies a presentation callback."""
    if callback is not None:
        callback(event, payload)
