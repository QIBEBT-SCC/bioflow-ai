from collections.abc import Callable

from bioflow_ai.cli.initialization import tui
from bioflow_ai.core.config import Settings
from bioflow_ai.services.initialization.containers import (
    COMPOSE_FILE,
    REQUIRED_SERVICES,
    check_compose_command,
    check_exist_container,
    compose_template_file,
    load_compose_file,
    prepare_container_volumes,
    start_project_services,
    wait_for_project_services,
)
from bioflow_ai.services.initialization.ports import UpdatePort


def initialize_project_containers(
    settings: Settings,
    confirm: Callable[..., bool],
    *,
    update_port: UpdatePort | None = None,
) -> bool | None:
    """Prompt for and optionally start project compose containers."""
    tui.render_section(
        'Project Containers',
        'Check the container runtime, synchronize the Web UI port, and start project services.',
    )

    runtime = settings.docker.CONTAINER
    if not confirm(f'Start project containers with {runtime}?', default=True):
        tui.render_notice('Project container startup skipped.', style='#94a3b8')
        return None

    compose_result = check_compose_command(settings.docker)
    if compose_result.warning:
        tui.render_notice(compose_result.warning, style='#f59e0b')
    if compose_result.error:
        tui.render_notice(compose_result.error, style='#ef4444')
        return False

    compose_command = compose_result.command
    if compose_command is None:
        return False

    template_file = compose_template_file(runtime)
    compose_file_result = load_compose_file(settings.network.PORT, update_port, template_file)
    if not compose_file_result.success:
        tui.render_notice(compose_file_result.error or 'Failed to generate docker-compose.yml.', style='#ef4444')
        return False
    if compose_file_result.written:
        tui.render_notice(f'Generated {COMPOSE_FILE} from {template_file}.', style='#10b981')
    tui.console.print(f'[bold]web-ui BACKEND_API_URL port:[/bold] {settings.network.PORT}')
    for message in compose_file_result.updated_port:
        tui.render_notice(f'Updated {message}.', style='#10b981')

    running_result = check_exist_container(compose_command, runtime)
    if not running_result.success:
        tui.render_notice(running_result.error or 'Unable to read Compose service status.', style='#f59e0b')
    running_services = set(running_result.services)
    if running_services:
        tui.console.print(
            f'[bold #10b981]Running services detected:[/bold #10b981] {", ".join(running_result.services)}'
        )
    if running_services >= REQUIRED_SERVICES:
        tui.render_notice('Project containers are already running; checking service health.', style='#94a3b8')
        ready_result = wait_for_project_services(compose_command, runtime)
        if ready_result.success:
            tui.render_notice(ready_result.message or 'Project services are ready.', style='#10b981')
            return True
        tui.render_notice(ready_result.error or 'Project services are not ready.', style='#ef4444')
        return False

    volume_result = prepare_container_volumes(runtime)
    if not volume_result.success:
        tui.render_notice(volume_result.error or 'Failed to prepare Compose volume directories.', style='#ef4444')
        return False
    if volume_result.message:
        tui.render_notice(volume_result.message, style='#10b981')

    command = [*compose_command, '-f', str(COMPOSE_FILE), 'up', '-d']
    tui.console.print()
    tui.console.print(f'[bold #22d3ee]Running:[/bold #22d3ee] {" ".join(command)}')
    result = start_project_services(command)
    if result.returncode != 0:
        tui.render_notice(
            f'Project container startup failed; Compose exited with code {result.returncode}.', style='#ef4444'
        )
        return False

    tui.render_notice('Project containers started; waiting for services to become ready.', style='#10b981')
    ready_result = wait_for_project_services(compose_command, runtime)
    if not ready_result.success:
        tui.render_notice(ready_result.error or 'Project services are not ready.', style='#ef4444')
        return False
    tui.render_notice(ready_result.message or 'Project services are ready.', style='#10b981')
    return True
