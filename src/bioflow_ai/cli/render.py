from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from langgraph.types import CustomStreamPart, MessagesStreamPart, UpdatesStreamPart
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text
from tqdm import tqdm

from bioflow_ai.llm.observability.stream_display import (
    REASONING_CONTENT_PREVIEW,
    TOOL_CONTENT_PREVIEW,
    display_preview,
    select_stream_display,
    tool_call_parts,
)
from bioflow_ai.services.progress import ProgressCallback, ProgressIterator

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, ToolMessage

SUBGRAPH_INDENT = 4


def create_iterator_renderer(
    *, description: str | None = None, unit: str = 'it', dynamic_ncols: bool = False
) -> ProgressIterator:
    """Keep terminal progress bars at the CLI presentation boundary."""

    def render[T](items: Iterable[T]) -> Iterable[T]:
        return tqdm(items, desc=description, unit=unit, dynamic_ncols=dynamic_ncols)

    return render


def create_progress_renderer() -> ProgressCallback:
    """Create one terminal presentation callback for a service invocation."""
    console = Console()

    def render(event: str, payload: Any) -> None:
        if event == 'chunk':
            print_stream_chunk(console, payload)
        elif event == 'interrupt':
            console.print(payload)

    return render


def print_stream_chunk(
    console: Console,
    chunk: UpdatesStreamPart | MessagesStreamPart | CustomStreamPart,
) -> None:
    """Render a LangGraph stream chunk as a Rich panel."""
    display = select_stream_display(chunk)
    if display is None:
        return
    namespace = display.namespace
    last_msg: AIMessage | None = display.model_message
    if last_msg is not None:
        reasoning_content = last_msg.additional_kwargs.get('reasoning_content')
        model_name = last_msg.response_metadata.get('model_name', '')

        has_tools = bool(last_msg.tool_calls)
        color = 'blue' if has_tools else 'magenta'

        body = Text()
        if reasoning_content:
            body.append(f'* {display_preview(str(reasoning_content), REASONING_CONTENT_PREVIEW)}', style='dim')
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

    if display.tools_update:
        _render_tools_update(console, display.tools_update, namespace)

    for key, value in display.middleware:
        _print_panel(
            console,
            Panel(Pretty(value), title=f'Middleware: {key}', border_style='grey50'),
            namespace,
        )


def _print_panel(console: Console, panel: Panel, namespace: tuple[str, ...]) -> None:
    """Print a panel indented according to its subgraph depth."""
    indent = len(namespace) * SUBGRAPH_INDENT
    console.print(Padding(panel, (0, 0, 0, indent)) if indent else panel)


def _render_tools_update(console: Console, tools_state: Mapping[str, Any], namespace: tuple[str, ...]) -> None:
    """Render tool output."""
    tool_msg: ToolMessage = tools_state['messages'][0]
    other_keys = [k for k in tools_state if k != 'messages']

    tool_name = tool_msg.name
    preview = display_preview(tool_msg.content, TOOL_CONTENT_PREVIEW)

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


def _format_tool_calls(tool_calls: list) -> Text:
    text = Text()
    for i, (name, args_str) in enumerate(tool_call_parts(tool_calls), 1):
        text.append(f'{i}. {name}', style='bold cyan')
        text.append('\n')
        text.append(args_str)
        text.append('\n')
    return text
