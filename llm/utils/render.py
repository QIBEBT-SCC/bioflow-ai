import json
from typing import TYPE_CHECKING, Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, ToolMessage

TOOL_CONTENT_PREVIEW = 200


def print_stream_chunk(
    console: Console,
    stream_mode: Literal['custom', 'updates'],
    chunk: Any,
) -> None:
    """Rend LangGraph stream chunk as rich Panel."""
    if stream_mode == 'custom':
        console.print(
            Panel(
                Pretty(chunk),
                # title='[bold yellow]🔔 Custom Event[/bold yellow]',
                border_style='yellow',
            )
        )
        return

    if stream_mode != 'updates' or not isinstance(chunk, dict):
        return

    if 'model' in chunk and 'messages' in chunk['model']:
        messages = chunk['model'].get('messages', [])
        last_msg: AIMessage = messages[0] if messages else None

        if last_msg is not None:
            tool_calls = getattr(last_msg, 'tool_calls', None)
            content = getattr(last_msg, 'content', '')
            model_name = last_msg.response_metadata.get('model_name', '')

            if tool_calls:
                console.print(
                    Panel(
                        _format_tool_calls(tool_calls),
                        title=f'[bold magenta]{model_name}[/bold magenta]',
                        border_style='magenta',
                    )
                )
            elif content:
                console.print(
                    Panel(
                        Text(str(content)),
                        title=f'[bold blue]{model_name}[/bold blue]',
                        border_style='blue',
                    )
                )

    if 'tools' in chunk:
        _render_tools_update(console, chunk['tools'])


def _render_tools_update(console: Console, tools_state: dict[str, Any]) -> None:
    """Render tool output."""
    tool_msg: ToolMessage = tools_state['messages'][0]
    other_keys = [k for k in tools_state if k != 'messages']

    tool_name = tool_msg.name
    preview = _preview(tool_msg.content, TOOL_CONTENT_PREVIEW)

    body = Text()
    body.append(f'{preview}\n\n')
    body.append('Updated keys: ', style='bold blue')
    body.append(
        ', '.join(other_keys) if other_keys else '(none)',
        style='dim' if not other_keys else '',
    )

    console.print(
        Panel(
            body,
            title=f'[bold green]{tool_name}[/bold green]',
            border_style='green',
        )
    )


def _preview(content: Any, limit: int) -> str:
    """把任意返回值转成字符串并截断到 limit 个字符."""
    text = content if isinstance(content, str) else str(content)
    if len(text) <= limit:
        return text
    return f'{text[:limit]}... [truncated, total {len(text):,} chars]'


def _truncate_value(value: Any, limit: int) -> Any:
    """递归截断 dict/list 里的字符串值."""
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return f'{value[:limit]}...'
    if isinstance(value, dict):
        return {k: _truncate_value(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_value(v, limit) for v in value]
    return value


def _format_tool_calls(tool_calls: list) -> str:
    parts: list[str] = []
    for i, tc in enumerate(tool_calls, 1):
        if isinstance(tc, dict):
            name = tc.get('name', 'unknown')
            args = tc.get('args', {})
        else:
            name = getattr(tc, 'name', 'unknown')
            args = getattr(tc, 'args', {})

        parts.append(f'[bold cyan]{i}. {name}[/bold cyan]')

        truncated_args = _truncate_value(args, 50)
        try:
            parts.append(json.dumps(truncated_args, ensure_ascii=False, indent=2, default=str))
        except (TypeError, ValueError):
            parts.append(str(truncated_args))
    return '\n'.join(parts)
