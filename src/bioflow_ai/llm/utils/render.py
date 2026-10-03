import json
from typing import TYPE_CHECKING, Any

from langgraph.types import CustomStreamPart, MessagesStreamPart, UpdatesStreamPart
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, ToolMessage

TOOL_CONTENT_PREVIEW = 200
REASONING_CONTENT_PREVIEW = 200
SUBGRAPH_INDENT = 4


def print_stream_chunk(
    console: Console,
    chunk: UpdatesStreamPart | MessagesStreamPart | CustomStreamPart,
) -> None:
    """Render a LangGraph stream chunk as a Rich panel."""
    namespace = chunk.get('ns', ())

    if chunk['type'] == 'custom':
        # _print_panel(
        #     console,
        #     Panel(
        #         Pretty(chunk['data']),
        #         title='[bold yellow]🔔 Custom Event[/bold yellow]',
        #         border_style='yellow',
        #     ),
        #     namespace,
        # )
        return

    if chunk['type'] == 'updates':
        models_messages = chunk['data'].get('model', {}).get('messages')
        tools_updates = chunk['data'].get('tools', {})
        middleware_data = {k: v for k, v in chunk['data'].items() if v and k not in ('model', 'tools')}

        if models_messages:
            last_msg: AIMessage = models_messages[0]

            if last_msg is not None:
                reasoning_content = last_msg.additional_kwargs.get('reasoning_content')
                model_name = last_msg.response_metadata.get('model_name', '')

                has_tools = bool(last_msg.tool_calls)
                color = 'blue' if has_tools else 'magenta'

                body = Text()
                if reasoning_content:
                    body.append(f'* {_preview(str(reasoning_content), REASONING_CONTENT_PREVIEW)}', style='dim')
                    body.append('\n')

                if last_msg.content:
                    body.append(str(last_msg.content))
                    if has_tools:
                        body.append('\n\n')

                if has_tools:
                    body.append_text(_format_tool_calls(last_msg.tool_calls))

                if body:
                    _print_panel(
                        console,
                        Panel(
                            body,
                            title=f'[bold {color}]LLM: {model_name}[/bold {color}]',
                            border_style=color,
                        ),
                        namespace,
                    )

        if tools_updates:
            _render_tools_update(console, tools_updates, namespace)

        if middleware_data:
            for key, value in middleware_data.items():
                if '.' in key and 'before_agent' not in key:
                    _print_panel(
                        console,
                        Panel(
                            Pretty(value),
                            title=f'Middleware: {key}',
                            border_style='grey50',
                        ),
                        namespace,
                    )


def _print_panel(console: Console, panel: Panel, namespace: tuple[str, ...]) -> None:
    """Print a panel indented according to its subgraph depth."""
    indent = len(namespace) * SUBGRAPH_INDENT
    console.print(Padding(panel, (0, 0, 0, indent)) if indent else panel)


def _render_tools_update(console: Console, tools_state: dict[str, Any], namespace: tuple[str, ...]) -> None:
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

    _print_panel(
        console,
        Panel(
            body,
            title=f'[bold green]Tool: {tool_name}[/bold green]',
            border_style='green',
        ),
        namespace,
    )


def _preview(content: Any, limit: int) -> str:
    """把任意返回值转成字符串并截断到 limit 个字符."""
    text = content if isinstance(content, str) else str(content)
    if len(text) <= limit:
        return text
    return f'{text[:limit]}...'


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


def _format_tool_calls(tool_calls: list) -> Text:
    text = Text()
    for i, tc in enumerate(tool_calls, 1):
        if isinstance(tc, dict):
            name = tc.get('name', 'unknown')
            args = tc.get('args', {})
        else:
            name = getattr(tc, 'name', 'unknown')
            args = getattr(tc, 'args', {})

        text.append(f'{i}. {name}', style='bold cyan')
        text.append('\n')

        truncated_args = _truncate_value(args, 50)
        try:
            args_str = json.dumps(truncated_args, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            args_str = str(truncated_args)
        text.append(args_str)
        text.append('\n')
    return text
