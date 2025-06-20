import datetime
import time
from typing import Generator, Any, Optional

import psutil
from docker.types import Mount
from loguru import logger
from podman import PodmanClient
from podman.errors import ImageNotFound

from app.core.config import get_settings
from app.models.tools import ImageConfig
from app.utils.system import get_cpu_cores


def _check_image(client: PodmanClient, image_cfg: ImageConfig):
    """检查镜像是否存在

    若不存在镜像，会进行拉取，并根据registry设置相应的身份验证

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


def run_podman(image_cfg: ImageConfig, command: str, mounts: list[Mount], **kwargs) -> tuple[int, str]:
    """在podman容器中运行命令并返回执行结果。

    同步运行容器命令，等待命令执行完成后返回退出状态码和输出日志。
    适用于短时间运行的命令，容器会在命令执行完成后自动清理。

    Args:
        image_cfg (ImageConfig): 容器镜像设置
        command (str): 容器内执行的命令
        mounts (list[Mount]): 文件挂载列表
        **kwargs: 其他容器参数，如环境变量、端口映射等

    Returns:
        tuple[int, str]: (容器退出状态码, 命令输出日志)
            - 退出状态码: 0表示成功，非0表示失败
            - 输出日志: 命令的标准输出和错误输出

    Raises:
        podman.errors.ContainerError: 容器运行失败时抛出
        podman.errors.ImageNotFound: 镜像不存在且拉取失败时抛出
        AssertionError: Podman客户端连接失败时抛出

    Example:
        >>> status, output = run_podman(image_cfg, "echo 'Hello World'", mounts)
        >>> print(f'Status: {status}, Output: {output}')
        Status: 0, Output: Hello World
    """
    with PodmanClient() as client:
        assert client.ping()

        _check_image(client, image_cfg)

        try:
            container = client.containers.run(
                image_cfg.get_image(),
                command,
                mounts=mounts,
                entrypoint=['/bin/bash', '-c'],
                remove=False,
                detach=True,
                **kwargs,
            )
            status_code = container.wait()
            log = b''.join(container.logs())
            return status_code, log.decode('utf-8').rstrip()
        finally:
            if container:
                container.remove()


def run_podman_background(
    image_cfg: ImageConfig, command: str, mounts: list[Mount], interval: float = 1.0, **kwargs
) -> Generator[dict[str, Any], None, None]:
    """后台运行podman容器，并实时返回容器资源占用信息。

    在容器内运行长耗时任务，定期采样并返回容器的资源使用情况，
    包括CPU、内存、网络I/O、块I/O等关键指标。

    Args:
        image_cfg (ImageConfig): 容器镜像设置
        command (str): 容器内执行的命令
        mounts (list[Mount]): 文件挂载列表
        interval (float): 资源采样间隔（秒），默认1.0秒
        **kwargs: 其他容器参数

    Yields:
        dict[str, Any]: 当前时刻容器的资源占用信息（如CPU、内存等）

    Raises:
        podman.errors.ContainerError: 容器运行失败时抛出
        podman.errors.ImageNotFound: 镜像不存在且拉取失败时抛出

    Example:
        >>> for stats in run_podman_background(image_cfg, 'sleep 10', mounts, interval=0.5):
        ...     print(f'CPU: {stats["cpu_percent"]:.2f}%, Memory: {stats["mem_usage_mb"]:.2f}MB')
    """
    with PodmanClient() as client:
        assert client.ping()

        _check_image(client, image_cfg)
        container = None

        try:
            container = client.containers.run(
                image_cfg.get_image(),
                command,
                mounts=mounts,
                entrypoint=['/bin/bash', '-c'],
                remove=False,
                detach=True,
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
                yield formate_container_stats(stat, start_time, pid)
                time.sleep(interval)
        finally:
            if container:
                container.remove()


def formate_container_stats(
    stats: dict[str, Any], start_time: datetime.datetime.timestamp, pid: int
) -> Optional[dict[str, Any]]:
    """提取容器日志中的监控内容

    从 Podman stats 输出中提取关键的监控指标，包括CPU利用率、内存利用率、
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
    try:
        if 'Stats' not in stats or not stats['Stats']:
            raise KeyError("Stats data is empty or missing 'Stats' key")

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
        except psutil.NoSuchProcess:
            logger.error(f'PID {pid} not found, skipping.')
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
