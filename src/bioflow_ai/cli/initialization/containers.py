from collections.abc import Callable

from bioflow_ai.cli.initialization import tui
from bioflow_ai.core.config import Settings
from bioflow_ai.services.initialization.common import proxy_environment
from bioflow_ai.services.initialization.containers import (
    COMPOSE_FILE,
    check_compose_command,
    compose_template_file,
    inspect_docker_proxy,
    load_compose_file,
    project_start_commands,
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

    if runtime == 'docker' and settings.network.USE_PROXY:
        docker_proxy = inspect_docker_proxy()
        if (
            docker_proxy is not None
            and not docker_proxy.http_proxy
            and not docker_proxy.https_proxy
            and not docker_proxy.mirrors
        ):
            tui.render_notice(
                'Docker daemon has no HTTP/HTTPS proxy or registry mirror configured. '
                'Docker pulls images via the dockerd service and does not read config.toml proxy settings; '
                'if image pulls stall, configure daemon proxy in /etc/systemd/system/docker.service.d/http-proxy.conf '
                'or ~/.docker/config.json.',
                style='#f59e0b',
            )

    compose_env = proxy_environment(settings.network) if runtime == 'podman' else None
    tui.console.print()
    for command in project_start_commands(compose_command, runtime):
        command_text = ' '.join(command)
        tui.console.print(f'[bold #22d3ee]Running:[/bold #22d3ee] {command_text}')
        result = start_project_services(command, env=compose_env)
        if result.returncode != 0:
            tui.render_notice(
                f'Compose exited with code {result.returncode} while running {command_text}.', style='#ef4444'
            )
            return False

    tui.render_notice('Project containers started; waiting for services to become ready.', style='#10b981')
    ready_result = wait_for_project_services(compose_command, runtime)
    if not ready_result.success:
        tui.render_notice(ready_result.error or 'Project services are not ready.', style='#ef4444')
        return False
    tui.render_notice(ready_result.message or 'Project services are ready.', style='#10b981')
    return True
