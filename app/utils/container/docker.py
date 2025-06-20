import time
from typing import Any, Optional

import docker
from docker import DockerClient
from docker.errors import ImageNotFound
from docker.types import Mount
from loguru import logger

from app.core.config import get_settings
from app.models.tools import ImageConfig
from app.utils.system import get_current_user_identity


def _check_image(client: DockerClient, image_cfg: ImageConfig):
    try:
        client.images.get(image_cfg.get_image())
    except ImageNotFound:
        logger.error(f'image {image_cfg.get_image()} not found, pull from {image_cfg.registry}')
        if image_cfg.registry == 'docker.io':
            client.images.pull(
                f'{image_cfg.registry}/{image_cfg.namespace}/{image_cfg.repository}',
                image_cfg.tag,
                auth_config={
                    'username': get_settings().docker.USERNAME,
                    'password': get_settings().docker.PASSWORD.get_secret_value(),
                },
            )
        else:
            client.images.pull(f'{image_cfg.registry}/{image_cfg.namespace}/{image_cfg.repository}', image_cfg.tag)
        time.sleep(1)


def run_docker(image_cfg: ImageConfig, command: str, mounts: list[Mount], **kwargs) -> tuple[int, str]:
    """在docker容器中运行命令并返回执行结果。

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
        docker.errors.ContainerError: 容器运行失败时抛出
        docker.errors.ImageNotFound: 镜像不存在且拉取失败时抛出

    Example:
        >>> status, output = run_docker(image_cfg, "echo 'Hello World'", mounts)
        >>> print(f'Status: {status}, Output: {output}')
        Status: 0, Output: Hello World
    """
    client = docker.from_env()
    _check_image(client, image_cfg)

    container = client.containers.run(
        image_cfg.get_image(),
        command,
        stderr=True,
        entrypoint=['/bin/bash', '-c'],
        user=get_current_user_identity(),
        auto_remove=False,
        detach=True,
        *kwargs,
    )

    try:
        status_code = container.wait()['StatusCode']
        result = container.logs(stderr=True)

        return status_code, result.decode('utf-8').rstrip()
    finally:
        if container:
            container.remove(force=True)


def run_docker_background(image_cfg: ImageConfig, command: str, mounts: list[Mount], interval: float = 1.0, **kwargs):
    """后台运行docker容器，并实时返回容器资源占用信息。

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
        docker.errors.ContainerError: 容器运行失败时抛出
        docker.errors.ImageNotFound: 镜像不存在且拉取失败时抛出

    Example:
        >>> import rich
        >>> for docker_stats in run_docker_background(image_cfg, 'sleep 10', mounts, interval=1.5):
        >>>     rich.print(docker_stats)
    """
    client = docker.from_env()
    _check_image(client, image_cfg)

    container = None

    try:
        container = client.containers.run(
            image_cfg.get_image(),
            f'"{command}"',
            mounts=mounts,
            entrypoint=['/bin/bash', '-c'],
            user=get_current_user_identity(),
            auto_remove=False,
            detach=True,
            *kwargs,
        )

        while True:
            container.reload()
            if container.status == 'exited':
                break
            stats = container.stats(stream=False)
            yield calculate_container_stats(stats)
            time.sleep(interval)

        # log = container.logs(stderr=True, stream=False).decode('utf-8')
        # print(log)
    finally:
        if container:
            container.remove(force=True)


def calculate_container_stats(stats: dict[str, Any]) -> Optional[dict[str, Any]]:
    """提取容器日志中的监控内容

    从 Docker stats 输出中提取关键的监控指标，包括CPU利用率、内存利用率、
    实际内存占用、网络I/O状态、块I/O状态等。

    Args:
        stats (dict[str, Any]): Podman stats 原始输出数据

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
        # CPU 相关参数
        try:
            cpu_usage = (
                stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            )
            system_cpu_usage = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_usage / system_cpu_usage) * 100.0 if system_cpu_usage > 0 else 0.0
        except TypeError:
            cpu_percent = 0.0

        # 内存相关参数
        try:
            mem_usage = stats['memory_stats']['usage']
            mem_limit = stats['memory_stats']['limit']
            mem_percent = (mem_usage / mem_limit) * 100.0 if mem_limit > 0 else 0.0
            mem_used_mb = mem_usage / (1024 * 1024)  # Convert to MB
        except TypeError:
            mem_percent = 0.0
            mem_used_mb = 0.0

        # IO 相关参数
        try:
            io_in = 0
            io_out = 0
            for io_stat in stats['blkio_stats']['io_service_bytes_recursive']:
                if io_stat['op'] == 'read':
                    io_in = io_stat['value']
                elif io_stat['op'] == 'write':
                    io_out = io_stat['value']
        except TypeError:
            io_in = 0
            io_out = 0

        # timestamp
        timestamp = stats['read']

        return {
            'cpu_usage': cpu_percent,
            'mem_usage': mem_percent,
            'mem_used': mem_used_mb,
            'io_in': io_in / (1024 * 1024),
            'io_out': io_out / (1024 * 1024),
            'time': timestamp,
        }
    except KeyError:
        return None
