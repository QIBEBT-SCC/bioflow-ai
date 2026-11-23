"""CLI commands and Rich presentation helpers for managed BioFlowAI services."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bioflow_ai.cli.errors import handle_service_errors
from bioflow_ai.services.runtime import (
    ALL_SERVICES,
    DEFAULT_START_TIMEOUT,
    DEFAULT_STOP_TIMEOUT,
    ServiceName,
    ServiceOperationResult,
    ServiceStatusInfo,
    get_all_service_statuses,
    iter_follow_logs,
    read_log_tail,
    restart_services,
    service_log_path,
    start_services,
    stop_services,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400

_ACTION_LABELS: dict[str, tuple[str, str]] = {
    'started': ('started', '#10b981'),
    'stopped': ('stopped', '#10b981'),
    'already_running': ('already running', '#94a3b8'),
    'not_running': ('not running', '#94a3b8'),
    'cleaned_stale': ('cleaned stale state', '#94a3b8'),
    'force_killed': ('force killed', '#f59e0b'),
    'failed': ('failed', '#ef4444'),
}


def format_uptime(uptime_seconds: float | None) -> str:
    """Format an uptime duration in seconds into a compact human-readable string."""
    if uptime_seconds is None:
        return '-'
    total = max(0, int(uptime_seconds))
    days, rem = divmod(total, _SECONDS_PER_DAY)
    hours, rem = divmod(rem, _SECONDS_PER_HOUR)
    minutes, seconds = divmod(rem, _SECONDS_PER_MINUTE)
    if days > 0:
        return f'{days}d {hours}h'
    if hours > 0:
        return f'{hours}h {minutes}m'
    if minutes > 0:
        return f'{minutes}m {seconds}s'
    return f'{seconds}s'


def render_operation_results(
    results: list[ServiceOperationResult],
    *,
    title: str,
    console: Console | None = None,
) -> None:
    """Render a summary table and any startup failure log tails."""
    target_console = console or Console()
    table = Table(title=title, box=box.ROUNDED, show_header=True)
    table.add_column('Service', style='bold #f8fafc', no_wrap=True)
    table.add_column('Result', no_wrap=True)
    table.add_column('PID', justify='right', style='#cbd5e1', no_wrap=True)
    table.add_column('Details', style='#cbd5e1')
    table.add_column('Log File', style='#94a3b8')

    for item in results:
        label, color = _ACTION_LABELS.get(item.action, (item.action, '#cbd5e1'))
        table.add_row(
            item.service,
            Text(label, style=f'bold {color}'),
            str(item.pid) if item.pid is not None else '-',
            Text(item.message),
            Text(str(item.log_file)),
        )

    target_console.print()
    target_console.print(table)

    for item in results:
        if not item.success and item.log_tail:
            target_console.print()
            target_console.print(
                Panel(
                    Text('\n'.join(item.log_tail)),
                    title=f'[bold #ef4444]{item.service} log tail ({item.log_file})[/bold #ef4444]',
                    border_style='#ef4444',
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )


def render_status_table(statuses: list[ServiceStatusInfo], *, console: Console | None = None) -> None:
    """Render the current status of BioFlowAI services."""
    target_console = console or Console()
    table = Table(title='BioFlowAI Services', box=box.ROUNDED, show_header=True)
    table.add_column('Service', style='bold #f8fafc', no_wrap=True)
    table.add_column('Status', no_wrap=True)
    table.add_column('PID', justify='right', style='#cbd5e1', no_wrap=True)
    table.add_column('Uptime', justify='right', style='#cbd5e1', no_wrap=True)
    table.add_column('Details', style='#cbd5e1')
    table.add_column('Log File', style='#94a3b8')

    for item in statuses:
        if item.status == 'running':
            if item.healthy is True:
                status_text = Text('running (healthy)', style='bold #10b981')
            elif item.healthy is False:
                status_text = Text('running (unhealthy)', style='bold #f59e0b')
            else:
                status_text = Text('running', style='bold #10b981')
        elif item.status == 'exited':
            status_text = Text('exited', style='bold #ef4444')
        else:
            status_text = Text('stopped', style='#64748b')

        detail_parts = [part for part in (item.detail, 'request log' if item.log_requests else None) if part]
        detail_display = ', '.join(detail_parts) if detail_parts else '-'

        table.add_row(
            item.service,
            status_text,
            str(item.pid) if item.pid is not None else '-',
            format_uptime(item.uptime_seconds),
            Text(detail_display),
            Text(str(item.log_file)),
        )

    target_console.print()
    target_console.print(table)


def _run_with_progress(
    console: Console,
    initial_message: str,
    action: Callable[[Callable[[ServiceName, str], None] | None], list[ServiceOperationResult]],
) -> list[ServiceOperationResult]:
    if not console.is_terminal:
        return action(None)
    with console.status(initial_message) as spinner:

        def on_progress(service: ServiceName, stage: str) -> None:
            if stage == 'starting':
                spinner.update(f'Starting {service}...')

        return action(on_progress)


@click.command()
@click.argument('services', nargs=-1, type=click.Choice(ALL_SERVICES))
@click.option(
    '--log',
    'enable_request_log',
    is_flag=True,
    help='Enable request body logging middleware for the API.',
)
@click.option(
    '--timeout',
    type=click.FloatRange(min=1.0),
    default=DEFAULT_START_TIMEOUT,
    show_default=True,
    help='Seconds to wait for each service to become ready.',
)
@handle_service_errors
def start(services: tuple[ServiceName, ...], *, enable_request_log: bool, timeout: float) -> None:
    """Start BioFlowAI backend services in the background."""
    console = Console()
    targets = services or None
    results = _run_with_progress(
        console,
        'Starting BioFlowAI services...',
        lambda on_progress: start_services(
            targets,
            log_requests=enable_request_log,
            timeout=timeout,
            on_progress=on_progress,
        ),
    )
    render_operation_results(results, title='Start Services', console=console)
    if any(not item.success for item in results):
        message = 'One or more services failed to start.'
        raise click.ClickException(message)


@click.command()
@click.argument('services', nargs=-1, type=click.Choice(ALL_SERVICES))
@click.option(
    '--timeout',
    type=click.FloatRange(min=0.0),
    default=DEFAULT_STOP_TIMEOUT,
    show_default=True,
    help='Seconds to wait for graceful shutdown before SIGKILL.',
)
@click.option('--force', '-f', is_flag=True, help='Send SIGKILL immediately instead of graceful SIGTERM.')
@handle_service_errors
def stop(services: tuple[ServiceName, ...], *, timeout: float, force: bool) -> None:
    """Stop running BioFlowAI backend services."""
    console = Console()
    targets = services or None
    with console.status('Stopping BioFlowAI services...') if console.is_terminal else nullcontext():
        results = stop_services(targets, timeout=timeout, force=force)
    render_operation_results(results, title='Stop Services', console=console)
    if any(not item.success for item in results):
        message = 'One or more services failed to stop.'
        raise click.ClickException(message)


@click.command()
@click.argument('services', nargs=-1, type=click.Choice(ALL_SERVICES))
@click.option(
    '--log/--no-log',
    'enable_request_log',
    default=None,
    help='Enable or disable API request body logging (preserves current setting if omitted).',
)
@click.option(
    '--stop-timeout',
    type=click.FloatRange(min=0.0),
    default=DEFAULT_STOP_TIMEOUT,
    show_default=True,
    help='Seconds to wait for graceful shutdown before SIGKILL.',
)
@click.option(
    '--start-timeout',
    type=click.FloatRange(min=1.0),
    default=DEFAULT_START_TIMEOUT,
    show_default=True,
    help='Seconds to wait for each service to become ready.',
)
@handle_service_errors
def restart(
    services: tuple[ServiceName, ...],
    *,
    enable_request_log: bool | None,
    stop_timeout: float,
    start_timeout: float,
) -> None:
    """Restart BioFlowAI backend services in the background."""
    console = Console()
    targets = services or None
    results = _run_with_progress(
        console,
        'Restarting BioFlowAI services...',
        lambda on_progress: restart_services(
            targets,
            log_requests=enable_request_log,
            stop_timeout=stop_timeout,
            start_timeout=start_timeout,
            on_progress=on_progress,
        ),
    )
    render_operation_results(results, title='Restart Services', console=console)
    if any(not item.success for item in results):
        message = 'One or more services failed to restart.'
        raise click.ClickException(message)


@click.command()
@click.argument('services', nargs=-1, type=click.Choice(ALL_SERVICES))
@handle_service_errors
def status(services: tuple[ServiceName, ...]) -> None:
    """Show the status of BioFlowAI backend services."""
    statuses = get_all_service_statuses(services or None, check_health=True)
    render_status_table(statuses)


@click.command()
@click.argument('service', required=False, type=click.Choice(ALL_SERVICES))
@click.option(
    '-n',
    '--lines',
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help='Number of trailing lines to show.',
)
@click.option('-f', '--follow', is_flag=True, help='Follow log output in real time.')
@handle_service_errors
def logs(service: ServiceName | None, *, lines: int, follow: bool) -> None:
    """Show or follow logs for BioFlowAI backend services."""
    console = Console()
    targets: list[ServiceName] = [service] if service is not None else list(ALL_SERVICES)
    log_files = {target: service_log_path(target) for target in targets}

    if not follow:
        if service is not None:
            path = log_files[service]
            if not path.is_file():
                console.print(Text(f'No log file found for {service} at {path}.', style='#94a3b8'))
                return
            tail = read_log_tail(path, lines=lines)
            if not tail:
                console.print(Text(f'Log file for {service} is empty ({path}).', style='#94a3b8'))
                return
            for line in tail:
                click.echo(line)
            return

        any_found = False
        for target, path in log_files.items():
            if not path.is_file():
                continue
            any_found = True
            tail = read_log_tail(path, lines=lines)
            console.print(Text(f'==> {target} ({path}) <==', style='bold #22d3ee'))
            if not tail:
                console.print(Text('(empty)', style='#94a3b8'))
            else:
                for line in tail:
                    click.echo(line)
        if not any_found:
            console.print(Text('No service log files found yet.', style='#94a3b8'))
        return

    console.print(Text(f'Following logs ({", ".join(targets)}). Press Ctrl+C to exit.', style='#94a3b8'))
    try:
        for svc, line in iter_follow_logs(log_files, lines=lines):
            if service is not None:
                click.echo(line)
            else:
                click.echo(f'[{svc}] {line}')
    except KeyboardInterrupt:
        return
