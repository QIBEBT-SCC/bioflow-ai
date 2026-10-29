"""Shared text truncation and safe JSON formatting helpers."""

from typing import Any

from bioflow_ai.utils.jsonable import dumps_jsonable

DEFAULT_FORMAT_MAX_CHARS = 50_000


def truncate_text(content: Any, limit: int, *, notice: str = '...') -> str:
    """Convert content to text and truncate it to a bounded preview."""
    text = content if isinstance(content, str) else str(content)
    if len(text) <= limit:
        return text
    return f'{text[:limit]}{notice}'


def format_text(content: Any, *, max_chars: int = DEFAULT_FORMAT_MAX_CHARS) -> str:
    """Format text with an explicit notice when content is omitted."""
    text = content if isinstance(content, str) else str(content)
    omitted = max(0, len(text) - max_chars)
    return truncate_text(text, max_chars, notice=f'\n... [truncated {omitted} characters]')


def format_json(
    value: Any,
    *,
    max_chars: int | None = DEFAULT_FORMAT_MAX_CHARS,
    indent: int | None = 2,
) -> str:
    """Format a safe JSON value, optionally bounding the result."""
    formatted = dumps_jsonable(value, indent=indent)
    if max_chars is None:
        return formatted
    return format_text(formatted, max_chars=max_chars)
