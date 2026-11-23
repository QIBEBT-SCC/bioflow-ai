"""Initialization helpers for stopping, starting, and inspecting background services."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from bioflow_ai.cli.initialization import tui
from bioflow_ai.cli.services import render_operation_results
from bioflow_ai.services.initialization.ports import SERVICE_DEFAULT_PORTS, UpdatePort
from bioflow_ai.services.runtime import (
    get_all_service_statuses,
    has_running_services,
    restart_services,
    start_services,
    stop_services,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def stop_running_services_before_reconfigure(confirm: Callable[..., bool]) -> None:
    """Offer to stop managed services before reconfiguring ports and settings."""
    running = [item for item in get_all_service_statuses() if item.status == 'running']
    if not running:
        return

    names = ', '.join(item.service for item in running)
    tui.render_notice(
        f'Managed backend services are already running ({names}). '
        'Stopping them prevents port preflight conflicts and applies the new configuration cleanly.',
        style='#f59e0b',
    )
    if not confirm('Stop running BioFlowAI backend services before reconfiguring?', default=True):
        return

    with tui.console.status('Stopping BioFlowAI backend services...') if tui.console.is_terminal else nullcontext():
        results = stop_services()
    render_operation_results(results, title='Stop Services', console=tui.console)


def initialize_backend_services(confirm: Callable[..., bool]) -> bool | None:
    """Prompt to start or restart the API server and Celery workers in the background."""
    tui.render_section(
        'Backend Services',
        'Start the API server, execution worker, and agent worker in the background.',
    )

    already_running = has_running_services()
    prompt = (
        'Restart BioFlowAI backend services in the background now?'
        if already_running
        else 'Start BioFlowAI backend services in the background now?'
    )
    if not confirm(prompt, default=True):
        tui.render_notice('Backend service startup skipped. Run `uv run bioflow-ai start` when ready.', style='#94a3b8')
        return None

    action_label = 'Restarting' if already_running else 'Starting'
    try:
        if tui.console.is_terminal:
            with tui.console.status(f'{action_label} BioFlowAI backend services...') as spinner:

                def on_progress(service: str, stage: str) -> None:
                    if stage == 'starting':
                        spinner.update(f'Starting {service}...')

                results = (
                    restart_services(on_progress=on_progress)
                    if already_running
                    else start_services(on_progress=on_progress)
                )
        else:
            results = restart_services() if already_running else start_services()
    except ValueError as exc:
        tui.render_notice(str(exc), style='#ef4444')
        return False

    title = 'Restart Services' if already_running else 'Start Services'
    render_operation_results(results, title=title, console=tui.console)
    return all(item.success for item in results)


def _extract_host_port(entry: object) -> int | None:
    if isinstance(entry, dict):
        published = entry.get('published')
        if isinstance(published, int) and published > 0:
            return published
        if isinstance(published, str) and published.isdigit():
            return int(published)
        return None

    if isinstance(entry, str):
        base = entry.split('/', maxsplit=1)[0]
        parts = base.split(':')
        host_part = parts[0] if len(parts) == 1 else parts[-2]
        if host_part.isdigit():
            return int(host_part)
    return None


def resolve_web_ui_port(
    update_port: UpdatePort | None,
    compose_file: Path = Path('docker-compose.yml'),
) -> int:
    """Determine the host port exposed for the Web UI container."""
    if update_port is not None and update_port.web_ui is not None:
        return update_port.web_ui

    default_port = SERVICE_DEFAULT_PORTS['web_ui']
    if not compose_file.is_file():
        return default_port

    try:
        compose_data = yaml.safe_load(compose_file.read_text(encoding='utf-8'))
        ports = compose_data['services']['web-ui']['ports']
        if isinstance(ports, list) and ports:
            parsed = _extract_host_port(ports[0])
            if parsed is not None:
                return parsed
    except (OSError, yaml.YAMLError, KeyError, IndexError, TypeError, ValueError):
        return default_port

    return default_port
