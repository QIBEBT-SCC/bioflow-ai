import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.utils.init import tui
from app.utils.init.common import run_command

if TYPE_CHECKING:
    from app.core.config import DockerConfig

COMPOSE_TEMPLATE_FILE = Path('docker') / 'docker-compose.yml'
COMPOSE_FILE = Path('docker-compose.yml')
WEB_BACKEND_API_PATTERN = re.compile(r'(BACKEND_API_URL:\s*http://host\.docker\.internal:)(\d+)(/api/v1)')


@dataclass(frozen=True)
class ComposeCommandResult:
    command: list[str] | None
    error: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class PortSyncResult:
    success: bool
    current_port: int | None = None
    written: bool = False
    updated: bool = False
    error: str | None = None


@dataclass(frozen=True)
class RunningServicesResult:
    success: bool
    services: list[str]
    error: str | None = None


@dataclass(frozen=True)
class OperationResult:
    success: bool
    message: str | None = None
    error: str | None = None


def detect_compose_command(docker: 'DockerConfig') -> ComposeCommandResult:
    """Detect the selected container runtime and a usable compose command."""
    runtime = docker.CONTAINER
    if shutil.which(runtime) is None:
        return ComposeCommandResult(command=None, error=f'未找到 {runtime} 命令, 请先安装 {runtime}。')

    version_result = run_command([runtime, '--version'])
    if version_result.returncode != 0:
        return ComposeCommandResult(command=None, error=f'{runtime} 无法运行: {version_result.stderr.strip()}')

    if runtime == 'docker':
        compose_command = ['docker', 'compose']
        compose_result = run_command([*compose_command, 'version'])
        if compose_result.returncode == 0:
            return ComposeCommandResult(command=compose_command)

        return ComposeCommandResult(
            command=None, error='Docker 已安装, 但 `docker compose` 无法运行。请安装 Docker Compose 插件。'
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
                warning='`podman compose` 不可用, 已改用 `podman-compose`。',
            )

    return ComposeCommandResult(
        command=None,
        error='Podman 已安装, 但 `podman compose` 无法运行。请安装 podman-compose 或 Podman Compose 插件。',
    )


def sync_web_backend_port(
    port: int,
    compose_file: Path = COMPOSE_FILE,
    template_file: Path = COMPOSE_TEMPLATE_FILE,
) -> PortSyncResult:
    """Render root docker-compose.yml from template and sync web-ui BACKEND_API_URL."""
    if not template_file.exists():
        return PortSyncResult(success=False, error=f'未找到 compose 模板 {template_file}, 无法启动项目容器。')

    content = template_file.read_text(encoding='utf-8')
    match = WEB_BACKEND_API_PATTERN.search(content)
    if match is None:
        return PortSyncResult(
            success=False, error=f'未在 compose 模板 {template_file} 中找到 web-ui BACKEND_API_URL, 请手动检查。'
        )

    current_port = int(match.group(2))
    updated = current_port != port
    if updated:
        content = WEB_BACKEND_API_PATTERN.sub(rf'\g<1>{port}\g<3>', content, count=1)

    existing_content = compose_file.read_text(encoding='utf-8') if compose_file.exists() else None
    written = existing_content != content
    if written:
        compose_file.write_text(content, encoding='utf-8')

    return PortSyncResult(success=True, current_port=current_port, written=written, updated=updated)


def compose_project_running(compose_command: list[str]) -> RunningServicesResult:
    """Check whether this compose project already has running services."""
    result = run_command([*compose_command, '-f', str(COMPOSE_FILE), 'ps', '--services', '--status', 'running'])
    if result.returncode != 0:
        return RunningServicesResult(
            success=False, services=[], error=f'无法读取 compose 运行状态: {result.stderr.strip()}'
        )

    services = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return RunningServicesResult(success=True, services=services)


def prepare_container_volumes() -> OperationResult:
    """Prepare first-run compose mount directories."""
    rustfs_path = Path('volumes') / 'rustfs'
    if rustfs_path.exists():
        return OperationResult(success=True)
    rustfs_path.mkdir(parents=True, exist_ok=True)
    os.chown(rustfs_path, 10001, 10001)

    return OperationResult(success=True, message=f'已准备 RustFS 数据目录: {rustfs_path}')


def initialize_project_containers(settings: Settings, confirm: Callable[..., bool]) -> bool | None:
    """Prompt for and optionally start project compose containers."""
    tui.render_section('项目容器', '检查容器运行时、同步 web-ui 端口并启动依赖服务。')

    runtime = settings.docker.CONTAINER
    if not confirm(f'是否使用 {runtime} 启动项目容器?', default=True):
        tui.render_notice('已跳过项目容器启动。', style='#94a3b8')
        return None

    compose_result = detect_compose_command(settings.docker)
    if compose_result.warning:
        tui.render_notice(compose_result.warning, style='#f59e0b')
    if compose_result.error:
        tui.render_notice(compose_result.error, style='#ef4444')
        return False

    compose_command = compose_result.command
    if compose_command is None:
        return False

    port_result = sync_web_backend_port(settings.network.PORT)
    if not port_result.success:
        tui.render_notice(port_result.error or '同步 web-ui BACKEND_API_URL 端口失败。', style='#ef4444')
        return False
    if port_result.written:
        tui.render_notice(f'已从 {COMPOSE_TEMPLATE_FILE} 生成 {COMPOSE_FILE}', style='#10b981')
    if port_result.updated:
        tui.render_notice(
            f'已将 web-ui BACKEND_API_URL 端口从 {port_result.current_port} 同步为 {settings.network.PORT}',
            style='#10b981',
        )
    else:
        tui.console.print(f'[bold]web-ui BACKEND_API_URL 端口:[/bold] {settings.network.PORT}')

    running_result = compose_project_running(compose_command)
    if not running_result.success:
        tui.render_notice(running_result.error or '无法读取 compose 运行状态。', style='#f59e0b')
    if running_result.services:
        tui.console.print(f'[bold #10b981]检测到已运行服务:[/bold #10b981] {", ".join(running_result.services)}')
        tui.render_notice('项目容器已在运行, 跳过启动。', style='#94a3b8')
        return True

    volume_result = prepare_container_volumes()
    if not volume_result.success:
        tui.render_notice(volume_result.error or '准备 compose 挂载目录失败。', style='#ef4444')
        return False
    if volume_result.message:
        tui.render_notice(volume_result.message, style='#10b981')

    command = [*compose_command, '-f', str(COMPOSE_FILE), 'up', '-d']
    tui.console.print()
    tui.console.print(f'[bold #22d3ee]执行:[/bold #22d3ee] {" ".join(command)}')
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        tui.render_notice(f'项目容器启动失败, compose 退出码: {result.returncode}', style='#ef4444')
        return False

    tui.render_notice('项目容器启动命令已执行。', style='#10b981')
    return True
