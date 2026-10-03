import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field

from bioflow_ai.core.config import Settings
from bioflow_ai.infrastructure import health_check
from bioflow_ai.utils.init import tui
from bioflow_ai.utils.init.common import run_command
from bioflow_ai.utils.init.ports import UpdatePort

if TYPE_CHECKING:
    from bioflow_ai.core.config import DockerConfig

from bioflow_ai.core.paths import DEPLOY_DIR

COMPOSE_TEMPLATE_ROOT = DEPLOY_DIR
COMPOSE_FILE = Path('docker-compose.yml')
REQUIRED_SERVICES = frozenset(
    {
        'etcd',
        'rustfs',
        'standalone',
        'rabbitmq-db',
        'redis-db',
        'postgres-db',
        'web-ui',
    }
)
SERVICE_HEALTH_CHECKS: dict[str, Callable[[list[str], Path], bool]] = {
    'postgres-db': health_check.check_postgres,
    'rustfs': health_check.check_rustfs,
    'standalone': health_check.check_milvus,
}
SERVICE_READY_TIMEOUT_SECONDS = 180.0
SERVICE_READY_POLL_SECONDS = 2.0


class ComposeCommandResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: list[str] | None = None
    error: str | None = None
    warning: str | None = None


class ComposeFileResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    written: bool = False
    updated_port: list[str] = Field(default_factory=list)
    error: str | None = None


class RunningServicesResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    services: list[str]
    error: str | None = None


class OperationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    message: str | None = None
    error: str | None = None


def compose_template_file(runtime: str) -> Path:
    """Return the compose template for the selected runtime."""
    return COMPOSE_TEMPLATE_ROOT / runtime / 'docker-compose.yml'


def check_compose_command(docker: 'DockerConfig') -> ComposeCommandResult:
    """Detect the selected container runtime and a usable compose command."""
    runtime = docker.CONTAINER
    if shutil.which(runtime) is None:
        return ComposeCommandResult(
            command=None, error=f'The {runtime} command was not found. Install {runtime} first.'
        )

    if runtime == 'docker':
        compose_command = ['docker', 'compose']
        compose_result = run_command([*compose_command, 'version'])
        if compose_result.returncode == 0:
            return ComposeCommandResult(command=compose_command)

        return ComposeCommandResult(
            command=None,
            error='Docker is installed, but `docker compose` is unavailable. Install the Docker Compose plugin.',
        )

    compose_command = ['podman', 'compose']
    compose_result = run_command([*compose_command, 'version'])
    if compose_result.returncode == 0:
        return ComposeCommandResult(command=compose_command)

    if shutil.which('podman-compose') is not None:
        fallback_command = ['podman-compose']
        fallback_result = run_command([*fallback_command, 'version'])
        if fallback_result.returncode == 0:
            return ComposeCommandResult(
                command=fallback_command,
                warning='`podman compose` is unavailable; using `podman-compose` instead.',
            )

    return ComposeCommandResult(
        command=None,
        error='Podman is installed, but Compose is unavailable. Install podman-compose or the Podman Compose plugin.',
    )


def load_compose_file(
    api_port: int,
    update_port: UpdatePort | None,
    template_file: Path,
) -> ComposeFileResult:
    """Render root docker-compose.yml from the fixed compose template."""
    if not template_file.exists():
        return ComposeFileResult(
            success=False,
            error=f'Compose template {template_file} was not found; project containers cannot be started.',
        )

    compose_object = yaml.safe_load(template_file.read_text(encoding='utf-8'))
    updated_port: list[str] = []

    try:
        services = compose_object['services']
        services['web-ui']['environment']['BACKEND_API_URL'] = f'http://host.docker.internal:{api_port}/api/v1'
        if update_port:
            if update_port.api:
                updated_port.append(f'web-ui BACKEND_API_URL port: {api_port}')
            if postgres_port := update_port.postgres:
                services['postgres-db']['ports'][0] = f'{postgres_port}:5432'
                updated_port.append(f'PostgreSQL host port: {postgres_port}')
            if redis_port := update_port.redis:
                services['redis-db']['ports'][0] = f'{redis_port}:6379'
                updated_port.append(f'Redis host port: {redis_port}')
            if rabbitmq_port := update_port.rabbitmq:
                services['rabbitmq-db']['ports'][0] = f'{rabbitmq_port}:5672'
                updated_port.append(f'RabbitMQ host port: {rabbitmq_port}')
            if rustfs_port := update_port.rustfs:
                services['rustfs']['ports'][0] = f'{rustfs_port}:9000'
                updated_port.append(f'RustFS/S3 host port: {rustfs_port}')
            if milvus_port := update_port.milvus:
                services['standalone']['ports'][0] = f'{milvus_port}:19530'
                updated_port.append(f'Milvus host port: {milvus_port}')
            if web_ui_port := update_port.web_ui:
                services['web-ui']['ports'][0] = f'{web_ui_port}:3000'
                updated_port.append(f'Web UI host port: {web_ui_port}')
    except (KeyError, IndexError, TypeError) as exc:
        return ComposeFileResult(success=False, error=f'The Compose template has an invalid structure: {exc}')

    content = yaml.safe_dump(compose_object, allow_unicode=True, sort_keys=False)
    COMPOSE_FILE.write_text(content, encoding='utf-8')

    return ComposeFileResult(success=True, written=True, updated_port=updated_port)


def check_exist_container(compose_command: list[str], runtime: str) -> RunningServicesResult:
    """Check whether this compose project already has running services."""
    if runtime == 'docker':
        result = run_command([*compose_command, '-f', str(COMPOSE_FILE), 'ps', '--services', '--status', 'running'])
        if result.returncode != 0:
            return RunningServicesResult(
                success=False,
                services=[],
                error=f'Unable to read Compose service status: {result.stderr.strip()}',
            )
        services = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
        return RunningServicesResult(success=True, services=services)

    result = run_command([*compose_command, '-f', str(COMPOSE_FILE), 'ps', '--format', 'json'])
    if result.returncode != 0:
        return RunningServicesResult(
            success=False,
            services=[],
            error=f'Unable to read Compose service status: {result.stderr.strip()}',
        )

    try:
        records = _parse_podman_ps_json(result.stdout)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return RunningServicesResult(success=False, services=[], error=f'Unable to parse Compose service status: {exc}')

    services = sorted(
        {
            service
            for record in records
            if _podman_ps_state(record) == 'running' and (service := _podman_ps_service(record))
        }
    )
    return RunningServicesResult(success=True, services=services)


def _parse_podman_ps_json(output: str) -> list[dict[str, object]]:
    """Parse JSON emitted by ``podman compose ps --format json``."""
    content = output.strip()
    if not content:
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in content.splitlines() if line.strip()]

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or any(not isinstance(record, dict) for record in parsed):
        msg = f'Expected a list of JSON objects, got {type(parsed).__name__}'
        raise TypeError(msg)
    return parsed


def _podman_ps_state(record: dict[str, object]) -> str:
    """Return a normalized state from a Podman container record."""
    state = record.get('State', record.get('state', ''))
    return str(state).strip().lower()


def _podman_ps_service(record: dict[str, object]) -> str:
    """Return the podman-compose service label from a Podman container record."""
    labels = record.get('Labels', record.get('labels', {}))
    if isinstance(labels, dict):
        if value := labels.get('io.podman.compose.service'):
            return str(value).strip()
    elif isinstance(labels, str):
        for label in labels.split(','):
            key, separator, value = label.partition('=')
            if separator and key.strip() == 'io.podman.compose.service':
                return value.strip()
    return ''


def check_unready_services(compose_command: list[str], runtime: str) -> list[str]:
    """Return missing, stopped, or unhealthy project services."""
    running_result = check_exist_container(compose_command, runtime)
    if not running_result.success:
        return sorted(REQUIRED_SERVICES)

    running_services = set(running_result.services)
    unready = set(REQUIRED_SERVICES - running_services)
    for service, check_service in SERVICE_HEALTH_CHECKS.items():
        if service in running_services and not check_service(compose_command, COMPOSE_FILE):
            unready.add(service)
    return sorted(unready)


def wait_for_project_services(
    compose_command: list[str],
    runtime: str,
    *,
    timeout: float = SERVICE_READY_TIMEOUT_SECONDS,
    poll_interval: float = SERVICE_READY_POLL_SECONDS,
) -> OperationResult:
    """Wait until all compose services run and initialization dependencies report healthy."""
    deadline = time.monotonic() + timeout
    while True:
        unready_services = check_unready_services(compose_command, runtime)
        if not unready_services:
            return OperationResult(success=True, message='Project services are ready.')
        if time.monotonic() >= deadline:
            names = ', '.join(unready_services)
            return OperationResult(success=False, error=f'Timed out waiting for project services: {names}')
        time.sleep(poll_interval)


def prepare_container_volumes(runtime: str) -> OperationResult:
    """Prepare compose mount directories and their required ownership."""
    volume_owners = {
        Path('volumes') / 'rustfs': (10001, 10001),
        Path('volumes') / 'milvus': (999, 999),
    }
    for path, (uid, gid) in volume_owners.items():
        path_existed = path.exists()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return OperationResult(success=False, error=f'Unable to create container data directory {path}: {exc}')

        if runtime == 'podman':
            # Rootless Podman must translate container IDs through its user namespace.
            result = run_command(['podman', 'unshare', 'chown', f'{uid}:{gid}', str(path)])
            if result.returncode != 0:
                error = result.stderr.strip() or f'exit code {result.returncode}'
                return OperationResult(
                    success=False, error=f'Unable to set ownership for Podman data directory {path}: {error}'
                )
            continue

        if path_existed:
            continue
        try:
            os.chown(path, uid, gid)
        except OSError as exc:
            return OperationResult(
                success=False,
                error=f'Unable to set owner {uid}:{gid} for container data directory {path}: {exc}',
            )

    rustfs_path, milvus_path = volume_owners
    return OperationResult(
        success=True,
        message=f'Initialized RustFS and Milvus data directories: {rustfs_path}, {milvus_path}',
    )


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
    result = subprocess.run(command, check=False)
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
