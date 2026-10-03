import datetime
import time
from collections.abc import Generator
from functools import lru_cache
from typing import Any, Optional

import psutil
from loguru import logger
from podman import PodmanClient  # pyrefly: ignore [missing-import]
from podman.errors import ImageNotFound  # pyrefly: ignore [missing-import]

from bioflow_ai.core.config import get_settings
from bioflow_ai.schemas.tool_config import ImageConfig
from bioflow_ai.utils.container._general import CONTAINER_ROOT_USER, _strip_ansi_escape_sequences
from bioflow_ai.utils.system import get_cpu_cores


@lru_cache
def _check_runtime_mode() -> None:
    """Warn when the connected Podman service is not rootless."""
    with PodmanClient() as client:
        host = client.info().get('host') or {}
        security = host.get('security') or {}
        if security.get('rootless') is not True:
            logger.warning(
                'Podman service mode: rootful; bind-mount outputs created by container root may be root-owned'
            )


def _check_image(client: PodmanClient, image_cfg: ImageConfig) -> None:
    """检查镜像是否存在.

    若不存在镜像, 会进行拉取, 并根据registry设置相应的身份验证

    Args:
        client: Podman client
        image_cfg: 镜像设置
    """
    try:
        client.images.get(image_cfg.get_image())
    except ImageNotFound:
        logger.error(f'image {image_cfg.get_image()} not found, pull from {image_cfg.registry}.')
        if image_cfg.registry == 'docker.io':
            client.images.pull(
                f'{image_cfg.registry}/{image_cfg.namespace}/{image_cfg.repository}',
                image_cfg.tag,
                auth_config={
                    'username': get_settings().docker.USERNAME,
                    'password': get_settings().docker.PASSWORD.get_secret_value(),
                },
                progress_bar=True,
            )
        else:
            client.images.pull(
                f'{image_cfg.registry}/{image_cfg.namespace}/{image_cfg.repository}', image_cfg.tag, progress_bar=True
            )


def _get_gpu_devices() -> list[str]:
    """Return Podman CDI device names configured for container creation."""
    settings = get_settings().docker
    if not settings.ENABLE_GPU:
        return []
    if settings.GPU_DEVICES == 'all':
        return ['nvidia.com/gpu=all']
    return [f'nvidia.com/gpu={gpu_id}' for gpu_id in settings.GPU_DEVICES.split(',')]


def run_podman(
    image_cfg: ImageConfig,
    command: str,
    mounts: list[dict],
    workdir: Optional[str] = None,
    *,
    user: str | None = CONTAINER_ROOT_USER,
    **kwargs: Any,
) -> tuple[int, str]:
    """在podman容器中运行命令并返回执行结果.

    同步运行容器命令, 等待命令执行完成后返回退出状态码和输出日志。
    适用于短时间运行的命令, 容器会在命令执行完成后自动清理。

    Args:
        image_cfg: 容器镜像设置
        command: 容器内执行的命令
        mounts: 文件挂载列表
        workdir: 命令运行的工作目录
        user: 容器内的用户身份。默认使用容器 root, 由 rootless Podman 映射到 service 所属宿主机用户
        **kwargs: 其他容器参数, 如环境变量、端口映射等

    Returns:
        tuple[int, str]: (容器退出状态码, 命令输出日志)
            - 退出状态码: 0表示成功, 非0表示失败
            - 输出日志: 命令的标准输出和错误输出

    Raises:
        podman.errors.ContainerError: 容器运行失败时抛出
        podman.errors.ImageNotFound: 镜像不存在且拉取失败时抛出
        AssertionError: Podman客户端连接失败时抛出
    """
    with PodmanClient() as client:
        if not client.ping():
            raise RuntimeError

        _check_runtime_mode()
        _check_image(client, image_cfg)

        if user is not None:
            kwargs['user'] = user

        try:
            container = client.containers.run(
                image_cfg.get_image(),
                command,
                mounts=mounts,
                entrypoint=['/bin/sh', '-c'],
                remove=False,
                detach=True,
                workdir=workdir,
                devices=_get_gpu_devices(),
                **kwargs,
            )
            status_code = container.wait()
            log = b''.join(container.logs())
            return status_code, log.decode('utf-8').rstrip()
        finally:
            if container:
                container.remove()


def run_podman_background(
    image_cfg: ImageConfig,
    command: str,
    mounts: list[dict],
    interval: float = 1.0,
    workdir: Optional[str] = None,
    *,
    user: str | None = CONTAINER_ROOT_USER,
    **kwargs: Any,
) -> Generator[dict[str, Any], None, None]:
    """后台运行podman容器, 并实时返回容器资源占用信息.

    在容器内运行长耗时任务, 定期采样并返回容器的资源使用情况,
    包括CPU、内存、网络I/O、块I/O等关键指标。

    Args:
        image_cfg: 容器镜像设置
        command: 容器内执行的命令
        mounts: 文件挂载列表
        interval: 资源采样间隔 (s), 默认1.0秒
        workdir: 命令运行的工作目录
        **kwargs: 其他容器参数

    Yields:
        dict[str, Any]: 当前时刻容器的资源占用信息 (如CPU、内存等)

    Raises:
        podman.errors.ContainerError: 容器运行失败时抛出
        podman.errors.ImageNotFound: 镜像不存在且拉取失败时抛出
    """
    with PodmanClient() as client:
        if not client.ping():
            raise RuntimeError

        _check_runtime_mode()
        _check_image(client, image_cfg)
        container = None

        if user is not None:
            kwargs['user'] = user

        try:
            container = client.containers.run(
                image_cfg.get_image(),
                command,
                mounts=mounts,
                entrypoint=['/bin/sh', '-c'],
                remove=False,
                detach=True,
                workdir=workdir,
                devices=_get_gpu_devices(),
                **kwargs,
            )

            container_info = container.inspect()
            start_time = datetime.datetime.fromisoformat(container_info['State']['StartedAt'])
            pid = container_info['State']['Pid']

            while True:
                container.reload()
                if container.status == 'exited':
                    break

                stat = container.stats(stream=False, decode=True)
                yield calculate_podman_stats(stat, start_time, pid)  # pyrefly: ignore [invalid-yield]
                time.sleep(interval)
        finally:
            timestamp = datetime.datetime.now(datetime.UTC)
            yield {
                'cpu_usage': 0.0,
                'mem_usage': 0.0,
                'mem_used': 0.0,
                'io_in': 0.0,
                'io_out': 0.0,
                'time': timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f') + '000Z',
            }
            if container:
                container.remove()


def create_container(
    image_cfg: ImageConfig,
    mounts: list[dict],
    *,
    debug: bool = False,
    **kwargs: int,
) -> str:
    """创建容器并返回其ID.

    Args:
        image_cfg: 所使用的镜像
        mounts: 外部文件夹挂载
        debug: 是否显示调试信息
        **kwargs:

    Returns:
        容器的ID
    """
    with PodmanClient() as client:
        if not client.ping():
            raise RuntimeError

        _check_runtime_mode()
        _check_image(client, image_cfg)

        container = client.containers.create(
            image_cfg.get_image(),
            command='tail -f /dev/null',  # 保证容器常驻
            mounts=mounts,
            entrypoint=['/bin/sh', '-c'],
            remove=False,
            detach=True,
            user=CONTAINER_ROOT_USER,
            devices=_get_gpu_devices(),
            **kwargs,
        )
        container.start()
        if debug:
            logger.debug(f'create container {container.name}')

        return container.id


def run_podman_by_id(
    container_id: str,
    command: str,
    workdir: Optional[str] = None,
) -> tuple[int, str]:
    """在特定容器中运行命令并返回结果.

    Args:
        container_id: 目标容器ID
        command: 要运行的指令
        workdir: 命令执行的工作目录

    Returns:
        - int: 运行的状态码
        - str: 日志输出结果
    """
    with PodmanClient() as client:
        if not client.ping():
            raise RuntimeError

        container = client.containers.get(container_id)
        code, result = container.exec_run(['/bin/sh', '-c', command], tty=False, stream=False, workdir=workdir)
        if isinstance(result, bytes):
            try:
                result = result.decode()
            except UnicodeDecodeError as e:
                result = f'{e}'

        result = _strip_ansi_escape_sequences(result.strip())
        return code, result


def remove_container(container_id: str) -> None:
    """停止并移除容器.

    Args:
        container_id: 要移除的容器ID
    """
    with PodmanClient() as client:
        if not client.ping():
            raise RuntimeError

        container = client.containers.get(container_id)
        container.remove(force=True)


def calculate_podman_stats(stats: dict[str, Any], start_time: datetime.datetime, pid: int) -> Optional[dict[str, Any]]:
    """提取容器日志中的监控内容.

    从 Podman stats 输出中提取关键的监控指标, 包括CPU利用率、内存利用率、
    实际内存占用、网络I/O状态、块I/O状态等。

    Args:
        stats (dict[str, Any]): Podman stats 原始输出数据
        start_time: 容器开始时间
        pid: 容器的宿主机pid

    Returns:
        dict[str, float]: 格式化后的监控指标
            - cpu_usage: CPU 使用百分比
            - mem_usage: 内存使用百分比
            - mem_used: 内存使用量 (MB)
            - io_in: 输入统计
            - io_out: 输出统计
            - uptime_seconds: 日志时间戳

    Raises:
        KeyError: 当 stats 数据结构不符合预期时抛出
        ValueError: 当数值转换失败时抛出
    """
    if 'Stats' not in stats or not stats['Stats']:
        logger.error("Stats data is empty or missing 'Stats' key")
        return None

    try:
        container_stats: dict[str, Any] = stats['Stats'][0]

        # CPU 相关指标
        cpu_percent = container_stats.get('CPU', 0.0) / get_cpu_cores()

        # 内存相关指标
        mem_percent = container_stats.get('MemPerc', 0.0) * 100
        mem_used_mb = container_stats.get('MemUsage', 0.0) / (1024 * 1024)

        # IO 相关指标
        try:
            p = psutil.Process(pid)

            io_in = sum([sub_p.io_counters().read_bytes for sub_p in p.children()])
            io_out = sum([sub_p.io_counters().write_bytes for sub_p in p.children()])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # logger.error(f'PID {pid} not found, skipping.')
            io_in = 0
            io_out = 0

        # 运行时间
        timestamp = start_time + datetime.timedelta(milliseconds=container_stats.get('Duration', 0.0) / 1000_000)

        return {
            'cpu_usage': cpu_percent,
            'mem_usage': mem_percent,
            'mem_used': mem_used_mb,
            'io_in': io_in / (1024 * 1024),
            'io_out': io_out / (1024 * 1024),
            'time': timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f') + '000Z',
        }

    except (KeyError, ValueError, TypeError) as e:
        logger.error(f'Failed to format container stats: {e}')
        return None
