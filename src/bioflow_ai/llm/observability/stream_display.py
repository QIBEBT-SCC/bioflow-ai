"""Shared stream object selection and terminal-specific preview helpers."""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from bioflow_ai.services.agents.images import media_preview

TOOL_CONTENT_PREVIEW = 200
REASONING_CONTENT_PREVIEW = 200
TOOL_ARGUMENT_PREVIEW = 50


@dataclass(frozen=True)
class StreamDisplay:
    """The contents shown by the terminal renderer for one graph update."""

    namespace: tuple[str, ...]
    model_message: Any
    tools_update: Mapping[str, Any]
    middleware: tuple[tuple[str, Any], ...]


def select_stream_display(chunk: Mapping[str, Any]) -> StreamDisplay | None:
    """Select only model, tool and visible middleware updates; custom events are UI-only."""
    if chunk.get('type') != 'updates':
        return None
    data = chunk['data']
    messages = data.get('model', {}).get('messages')
    return StreamDisplay(
        namespace=tuple(chunk.get('ns', ()) or ()),
        model_message=messages[0] if messages else None,
        tools_update=data.get('tools', {}),
        middleware=tuple(
            (key, value)
            for key, value in data.items()
            if value and key not in ('model', 'tools') and '.' in key and 'before_agent' not in key
        ),
    )


def display_preview(content: Any, limit: int) -> str:
    """Use the terminal's exact string conversion and truncation rule."""
    content = media_preview(content)
    text = content if isinstance(content, str) else str(content)
    return text if len(text) <= limit else f'{text[:limit]}...'


def _truncate_value(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return display_preview(value, limit)
    if isinstance(value, dict):
        return {key: _truncate_value(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item, limit) for item in value]
    return value


def tool_arguments_text(arguments: Any) -> str:
    """Format tool arguments with the terminal's per-string preview length."""
    preview = _truncate_value(arguments, TOOL_ARGUMENT_PREVIEW)
    try:
        return json.dumps(preview, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(preview)


def tool_call_parts(tool_calls: list) -> Iterator[tuple[str, str]]:
    """Yield the name and formatted arguments displayed for each tool call."""
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            name = tool_call.get('name', 'unknown')
            arguments = tool_call.get('args', {})
        else:
            name = getattr(tool_call, 'name', 'unknown')
            arguments = getattr(tool_call, 'args', {})
        yield name, tool_arguments_text(arguments)
