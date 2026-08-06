import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.utils.init import tui
from app.utils.init.common import run_command
from app.utils.init.ports import UpdatePort

if TYPE_CHECKING:
    from app.core.config import DockerConfig

COMPOSE_TEMPLATE_ROOT = Path('deploy')
COMPOSE_FILE = Path('docker-compose.yml')


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
        return ComposeCommandResult(command=None, error=f'未找到 {runtime} 命令, 请先安装 {runtime}。')

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


def load_compose_file(
    api_port: int,
    update_port: UpdatePort | None,
    template_file: Path,
) -> ComposeFileResult:
    """Render root docker-compose.yml from the fixed compose template."""
    if not template_file.exists():
        return ComposeFileResult(success=False, error=f'未找到 compose 模板 {template_file}, 无法启动项目容器。')

    compose_object = yaml.safe_load(template_file.read_text(encoding='utf-8'))
    updated_port: list[str] = []

    try:
        services = compose_object['services']
        services['web-ui']['environment']['BACKEND_API_URL'] = f'http://host.docker.internal:{api_port}/api/v1'
        if update_port:
            if update_port.api:
                updated_port.append(f'web-ui BACKEND_API_URL 端口: {api_port}')
            if postgres_port := update_port.postgres:
                services['postgres-db']['ports'][0] = f'{postgres_port}:5432'
                updated_port.append(f'PostgreSQL 宿主机端口: {postgres_port}')
            if redis_port := update_port.redis:
                services['redis-db']['ports'][0] = f'{redis_port}:6379'
                updated_port.append(f'Redis 宿主机端口: {redis_port}')
            if rabbitmq_port := update_port.rabbitmq:
                services['rabbitmq-db']['ports'][0] = f'{rabbitmq_port}:5672'
                updated_port.append(f'RabbitMQ 宿主机端口: {rabbitmq_port}')
            if rustfs_port := update_port.rustfs:
                services['rustfs']['ports'][0] = f'{rustfs_port}:9000'
                updated_port.append(f'RustFS/S3 宿主机端口: {rustfs_port}')
            if milvus_port := update_port.milvus:
                services['standalone']['ports'][0] = f'{milvus_port}:19530'
                updated_port.append(f'Milvus 宿主机端口: {milvus_port}')
            if web_ui_port := update_port.web_ui:
                services['web-ui']['ports'][0] = f'{web_ui_port}:3000'
                updated_port.append(f'Web UI 宿主机端口: {web_ui_port}')
    except (KeyError, IndexError, TypeError) as exc:
        return ComposeFileResult(success=False, error=f'compose 模板结构不符合预期: {exc}')

    content = yaml.safe_dump(compose_object, allow_unicode=True, sort_keys=False)
    COMPOSE_FILE.write_text(content, encoding='utf-8')

    return ComposeFileResult(success=True, written=True, updated_port=updated_port)


def check_exist_container(compose_command: list[str]) -> RunningServicesResult:
    """Check whether this compose project already has running services."""
    result = run_command([*compose_command, '-f', str(COMPOSE_FILE), 'ps', '--services', '--status', 'running'])
    if result.returncode != 0:
        return RunningServicesResult(
            success=False, services=[], error=f'无法读取 compose 运行状态: {result.stderr.strip()}'
        )

    services = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return RunningServicesResult(success=True, services=services)


def prepare_container_volumes() -> OperationResult:
    """Prepare compose mount directories and their required ownership."""
    rustfs_path = Path('volumes') / 'rustfs'
    if not rustfs_path.exists():
        rustfs_path.mkdir(parents=True, exist_ok=True)
        os.chown(rustfs_path, 10001, 10001)

    milvus_path = Path('volumes') / 'milvus'
    if not milvus_path.exists():
        milvus_path.mkdir(parents=True, exist_ok=True)
        os.chown(milvus_path, 999, 999)

    return OperationResult(success=True, message=f'初始化 RustFS 和 Milvus 数据目录: {rustfs_path}, {milvus_path}')


def initialize_project_containers(
    settings: Settings,
    confirm: Callable[..., bool],
    *,
    update_port: UpdatePort | None = None,
) -> bool | None:
    """Prompt for and optionally start project compose containers."""
    tui.render_section('项目容器', '检查容器运行时、同步 web-ui 端口并启动依赖服务。')

    runtime = settings.docker.CONTAINER
    if not confirm(f'是否使用 {runtime} 启动项目容器?', default=True):
        tui.render_notice('已跳过项目容器启动。', style='#94a3b8')
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
        tui.render_notice(compose_file_result.error or '生成 docker-compose.yml 失败。', style='#ef4444')
        return False
    if compose_file_result.written:
        tui.render_notice(f'已从 {template_file} 生成 {COMPOSE_FILE}', style='#10b981')
    tui.console.print(f'[bold]web-ui BACKEND_API_URL 端口:[/bold] {settings.network.PORT}')
    for message in compose_file_result.updated_port:
        tui.render_notice(f'已同步 {message}', style='#10b981')

    running_result = check_exist_container(compose_command)
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
