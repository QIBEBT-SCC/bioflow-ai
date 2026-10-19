"""Shared stream object selection and terminal-specific preview helpers."""

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

TOOL_CONTENT_PREVIEW = 200
REASONING_CONTENT_PREVIEW = 200
TOOL_ARGUMENT_PREVIEW = 50
INLINE_IMAGE = re.compile(r'data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+')


class StreamDisplay(BaseModel):
    """The contents shown by the terminal renderer for one graph update."""

    model_config = ConfigDict(frozen=True)

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


def redact_image_payloads(value: Any) -> Any:
    """Preserve image labels and metadata while replacing encoded pixel data."""
    if isinstance(value, str):
        return INLINE_IMAGE.sub('[inline image]', value)
    if isinstance(value, dict):
        return {
            key: '[inline image]'
            if key in {'base64', 'data'} and value.get('type') == 'image'
            else redact_image_payloads(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_image_payloads(item) for item in value]
    return value


def display_preview(content: Any, limit: int) -> str:
    """Use the terminal's exact string conversion and truncation rule."""
    content = redact_image_payloads(content)
    text = content if isinstance(content, str) else str(content)
    return text if len(text) <= limit else f'{text[:limit]}...'


def message_text_parts(message: Any) -> tuple[str, str]:
    """Split a model message into displayed reasoning and text.

    Chat Completions put text in a string and reasoning in ``additional_kwargs``; the Responses API returns
    a list of ``reasoning``, ``text`` and ``function_call`` blocks whose tool calls are shown separately.
    """
    reasoning = [str(value)] if (value := message.additional_kwargs.get('reasoning_content')) else []
    content = message.content
    if isinstance(content, str):
        return '\n'.join(reasoning), content

    texts = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif not isinstance(block, dict):
            continue
        elif block.get('type') == 'text':
            texts.append(str(block.get('text', '')))
        elif block.get('type') == 'reasoning':
            if summary := block.get('summary'):
                reasoning.extend(str(part.get('text', '')) for part in summary if isinstance(part, dict))
            elif block.get('reasoning'):
                reasoning.append(str(block['reasoning']))
    return '\n'.join(part for part in reasoning if part), ''.join(texts)


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
