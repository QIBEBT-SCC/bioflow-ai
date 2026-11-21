import json
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field

from bioflow_ai.infrastructure import health_check
from bioflow_ai.services.initialization.common import run_command
from bioflow_ai.services.initialization.ports import UpdatePort

if TYPE_CHECKING:
    from bioflow_ai.core.config import DockerConfig

from bioflow_ai.core.paths import DEPLOY_DIR

COMPOSE_TEMPLATE_ROOT = DEPLOY_DIR
COMPOSE_FILE = Path('docker-compose.yml')
VOLUME_INIT_SERVICES = frozenset({'milvus-volume-init', 'rustfs-volume-init'})
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
        records = _parse_compose_ps_json(result.stdout)
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


def _parse_compose_ps_json(output: str) -> list[dict[str, object]]:
    """Parse a JSON array or newline-delimited objects emitted by Compose ps."""
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


def check_unready_volume_initializers(compose_command: list[str], runtime: str) -> list[str]:
    """Verify Docker initializer exits. Podman checks the exit codes before starting services."""
    if runtime == 'podman':
        return []
    result = run_command([*compose_command, '-f', str(COMPOSE_FILE), 'ps', '--all', '--format', 'json'])
    if result.returncode != 0:
        return sorted(VOLUME_INIT_SERVICES)

    try:
        records = _parse_compose_ps_json(result.stdout)
    except (json.JSONDecodeError, TypeError, ValueError):
        return sorted(VOLUME_INIT_SERVICES)

    completed = set()
    for record in records:
        if _podman_ps_state(record) in {'exited', 'stopped'} and record.get('ExitCode') == 0:
            completed.add(str(record.get('Service', '')))
    return sorted(VOLUME_INIT_SERVICES - completed)


def check_unready_services(compose_command: list[str], runtime: str) -> list[str]:
    """Return missing, stopped, or unhealthy project services."""
    running_result = check_exist_container(compose_command, runtime)
    if not running_result.success:
        return sorted(REQUIRED_SERVICES)

    running_services = set(running_result.services)
    unready = set(REQUIRED_SERVICES - running_services)
    unready.update(check_unready_volume_initializers(compose_command, runtime))
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
    """Wait for volume initialization, running services, and dependency health checks."""
    deadline = time.monotonic() + timeout
    while True:
        unready_services = check_unready_services(compose_command, runtime)
        if not unready_services:
            return OperationResult(success=True, message='Project services are ready.')
        if time.monotonic() >= deadline:
            names = ', '.join(unready_services)
            return OperationResult(success=False, error=f'Timed out waiting for project services: {names}')
        time.sleep(poll_interval)


def project_start_commands(compose_command: list[str], runtime: str) -> list[list[str]]:
    """Run Podman volume jobs outside the long-running services' dependency graph."""
    command = [*compose_command, '-f', str(COMPOSE_FILE)]
    up = [*command, 'up', '-d']
    if runtime == 'podman':
        run = [*command, '--profile', 'volume-init', 'run', '--rm', '-T', '--no-deps']
        return [[*run, service] for service in sorted(VOLUME_INIT_SERVICES)] + [up]
    return [up]


def start_project_services(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Start the rendered Compose services using their native command."""
    return subprocess.run(command, check=False)
