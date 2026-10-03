from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bioflow_ai.services.development.database import TOOL_STATUS_COMPLETED, TOOL_STATUS_PENDING


def render_tools(rows: list[dict[str, Any]]) -> None:
    """Display the matching development tool records."""
    console = Console()
    for row in rows:
        content = Text()
        content.append('Tool: ', style='bold cyan')
        content.append(f'{row["tool_name"]}\n', style='magenta')
        content.append('Status: ', style='bold cyan')
        status = row['status']
        status_style = (
            'green' if status == TOOL_STATUS_COMPLETED else 'yellow' if status == TOOL_STATUS_PENDING else 'red'
        )
        content.append(f'{status}\n', style=status_style)
        content.append('Tool UID: ', style='bold cyan')
        content.append(f'{row["tool_uid"]}\n', style='white')
        content.append('Description:\n', style='bold cyan')
        content.append(row['function_description'], style='white')

        console.print(
            Panel(
                content,
                title=f'[yellow]#{row["index"]} {row["thread_id"]}[/yellow]',
                border_style='blue',
                expand=True,
            )
        )


def render_updated_requirement(index: int, previous: dict[str, Any], updated: dict[str, Any]) -> None:
    """Display the saved requirement beside its previous value."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style='bold cyan', justify='right', no_wrap=True)
    table.add_column(overflow='fold')
    table.add_row('previous requirement:', Text(str(previous['function_description'])))
    for key, value in updated.items():
        if value is None:
            display = Text('None', style='dim')
        elif key == 'status':
            status_style = (
                'green' if value == TOOL_STATUS_COMPLETED else 'yellow' if value == TOOL_STATUS_PENDING else 'red'
            )
            display = Text(str(value), style=status_style)
        else:
            display = Text(str(value))
        table.add_row(f'{key}:', display)

    Console().print(
        Panel(
            table,
            title=f'[bold green]Updated Tool Requirement #{index}[/bold green]',
            border_style='green',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def render_statistics(values: dict[str, Any]) -> None:
    """Present development statistics and latest run records."""
    total = values['total']
    processed = values['processed']
    success = values['success']
    unique_nodes = values['unique_nodes']
    failed = values['failed']
    interrupted = values['interrupted']
    running = values['running']
    console = Console()

    stats = Table.grid(padding=(0, 2))
    stats.add_column(style='bold', justify='right')
    stats.add_column()

    pct = (processed / total * 100) if total else 0

    stats.add_row('Total records', f'[cyan]{total:,}[/cyan]')
    stats.add_row('Processed', f'[cyan]{processed:,}[/cyan] [dim]({pct:.1f}%)[/dim]')
    stats.add_row('Succeeded', f'[green]{success:,}[/green]  Nodes generated [green]{unique_nodes:,}[/green]')
    stats.add_row(
        'Incomplete',
        f'Running [cyan]{running:,}[/cyan]  Interrupted [yellow]{interrupted:,}[/yellow]  Failed [red]{failed:,}[/red]',
    )

    console.print(
        Panel(
            stats,
            title='[bold]Run Statistics[/bold]',
            border_style='blue',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    def _render_record_panel(row: dict | None, title: str, border_style: str = 'cyan') -> None:
        if not row:
            console.print(Panel('[dim](No data)[/dim]', title=title, border_style=border_style))
            return

        body = Table.grid(padding=(0, 2))
        body.add_column(style='bold cyan', justify='right', no_wrap=True)
        body.add_column(overflow='fold')

        for key, value in row.items():
            if value is None:
                display = '[dim]None[/dim]'
            elif key == 'status':
                style = (
                    'green' if value == TOOL_STATUS_COMPLETED else 'yellow' if value == TOOL_STATUS_PENDING else 'red'
                )
                display = f'[{style}]{value}[/{style}]'
            else:
                display = str(value)
            body.add_row(f'{key}:', display)

        console.print(
            Panel(
                body,
                title=title,
                border_style=border_style,
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    _render_record_panel(values['latest'], '[bold green]Latest Run Record[/bold green]')
    _render_record_panel(values['latest_interrupt'], '[bold yellow]Latest Interrupted Record[/bold yellow]')
