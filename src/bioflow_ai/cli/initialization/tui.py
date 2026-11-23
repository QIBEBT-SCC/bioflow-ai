from pathlib import Path
from typing import Any

import questionary
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# Replace the leading question mark with a green check mark after each answer.
DONE_MARK = '\x1b[1A\r\x1b[38;2;16;185;129m✓\x1b[0m\x1b[1B\r'
PROMPT_STYLE = questionary.Style(
    [
        ('qmark', 'fg:#22d3ee bold'),
        ('question', 'fg:#e5e7eb bold'),
        ('answer', 'fg:#10b981 bold'),
        ('pointer', 'fg:#f59e0b bold'),
        ('highlighted', 'fg:#22d3ee bold'),
        ('selected', 'fg:#10b981'),
        ('separator', 'fg:#64748b'),
        ('instruction', 'fg:#94a3b8'),
        ('text', 'fg:#e5e7eb'),
        ('disabled', 'fg:#64748b italic'),
    ]
)


def finish(question: questionary.Question) -> Any:
    """Submit a question and replace the leading question mark with a check mark."""
    answer = question.unsafe_ask()
    _render_prompt_done()
    return answer


async def finish_async(question: questionary.Question) -> Any:
    """Asynchronously submit a question and mark it as completed."""
    answer = await question.unsafe_ask_async()
    _render_prompt_done()
    return answer


def _render_prompt_done() -> None:
    """Replace the leading question mark of the previous prompt with a check mark."""
    console.file.write(DONE_MARK)
    console.file.flush()


def prompt_text(message: str, *, default: str = '', validate: Any = None) -> str:
    """Prompt for text input with the shared style."""
    return finish(questionary.text(message, default=default, validate=validate, style=PROMPT_STYLE))


def prompt_password(message: str, *, validate: Any = None) -> str:
    """Prompt for password input with the shared style."""
    return finish(questionary.password(message, validate=validate, style=PROMPT_STYLE))


async def prompt_text_async(message: str, *, default: str = '', validate: Any = None) -> str:
    """Asynchronously prompt for text input with the shared style."""
    return await finish_async(questionary.text(message, default=default, validate=validate, style=PROMPT_STYLE))


async def prompt_password_async(message: str, *, validate: Any = None) -> str:
    """Asynchronously prompt for password input with the shared style."""
    return await finish_async(questionary.password(message, validate=validate, style=PROMPT_STYLE))


def prompt_confirm(message: str, *, default: bool = False) -> bool:
    """Prompt for confirmation with the shared style."""
    return finish(questionary.confirm(message, default=default, style=PROMPT_STYLE))


async def prompt_confirm_async(message: str, *, default: bool = False) -> bool:
    """Asynchronously prompt for confirmation with the shared style."""
    return await finish_async(questionary.confirm(message, default=default, style=PROMPT_STYLE))


def prompt_select(message: str, *, choices: list[str], default: str) -> str:
    """Prompt for a single selection with the shared style."""
    return finish(questionary.select(message, choices=choices, default=default, style=PROMPT_STYLE))


def prompt_checkbox(message: str, *, choices: list[Any], validate: Any = None) -> list[Any]:
    """Prompt for multiple selections with the shared style."""
    return finish(questionary.checkbox(message, choices=choices, validate=validate, style=PROMPT_STYLE))


def prompt_path(message: str, *, default: str) -> str:
    """Prompt for a path with the shared style."""
    return finish(questionary.path(message, default=default, style=PROMPT_STYLE))


def render_hero() -> None:
    """Render the initialization wizard hero."""
    title = Text('BioFlowAI', style='bold #22d3ee')
    title.append(' Configuration & Initialization Wizard', style='bold #f8fafc')

    body = Table.grid(padding=(0, 1))
    body.add_column(justify='center')
    body.add_row(title)
    body.add_row(Text('Follow the steps to generate config.toml. Press Enter to accept defaults.', style='#94a3b8'))

    console.print()
    console.print(
        Panel(
            Align.center(body),
            border_style='#22d3ee',
            box=box.ROUNDED,
            padding=(1, 4),
        )
    )
    console.print()


def render_section(title: str, description: str) -> None:
    """Render a section heading."""
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(Text(title, style='bold #f8fafc'))
    body.add_row(Text(description, style='#94a3b8'))

    console.print()
    console.print(
        Panel(
            body,
            border_style='#334155',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def render_notice(message: str, *, style: str = '#f59e0b') -> None:
    """Render a notice message."""
    console.print(Panel(Text(message, style=style), border_style=style, box=box.ROUNDED, padding=(0, 2)))


def render_subsection(title: str) -> None:
    """Render a model setup heading within the current section."""
    console.print()
    console.print(Text(title, style='bold #f8fafc'))


def render_success(config_path: Path) -> None:
    """Render deployment configuration success."""
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(Text('Deployment configuration saved', style='bold #10b981'))
    body.add_row(Text(f'Written to {config_path}', style='#cbd5e1'))

    console.print()
    console.print(
        Panel(
            body,
            title='[bold #10b981]Done[/bold #10b981]',
            border_style='#10b981',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def render_preset_database_summary(result: Any) -> None:
    """Render preset database initialization summary."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style='#cbd5e1')
    table.add_column(style='#f8fafc')

    admin_status = 'Created' if result.admin.created else 'Already exists; skipped'
    table.add_row('Administrator account', admin_status)

    for item in (result.images, result.docs, result.tools, result.tool_memories):
        parts = [f'Imported {item.inserted_count}']
        if item.deleted_count is not None:
            parts.append(f'Removed {item.deleted_count} old records')
        if item.downloaded:
            parts.append('Downloaded preset')
        if item.validation_error_count:
            parts.append(f'{item.validation_error_count} validation failures')
        if item.warning_count:
            parts.append(f'{item.warning_count} warnings')
        table.add_row(item.name, ', '.join(parts))

    table.add_row('Vector indexes', 'Rebuilt')

    console.print()
    console.print(
        Panel(
            table,
            title='[bold #10b981]Preset Data Initialized[/bold #10b981]',
            border_style='#10b981',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def render_model_role_assignments(assignments: list[tuple[str, str]]) -> None:
    """Render the model currently assigned to each system role."""
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style='#64748b', justify='right')
    table.add_column(style='#cbd5e1')
    table.add_column(style='#10b981')
    for index, (role_label, model_label) in enumerate(assignments, start=1):
        table.add_row(str(index), role_label, model_label)

    console.print()
    console.print(
        Panel(
            table,
            title='[bold #f8fafc]System Agent Models[/bold #f8fafc]',
            border_style='#334155',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def render_system_success() -> None:
    """Render system initialization success."""
    console.print()
    console.print(
        Panel(
            Text('System initialization complete.', style='bold #10b981'),
            title='[bold #10b981]Done[/bold #10b981]',
            border_style='#10b981',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
