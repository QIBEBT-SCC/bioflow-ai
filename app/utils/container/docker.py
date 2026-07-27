import datetime
import time
from collections.abc import Generator
from typing import Any, Optional

import docker
import requests
from docker.errors import ImageNotFound
from docker.types import DeviceRequest, Mount
from loguru import logger

from app.core.config import get_settings
from app.schemas.tool_config import ImageConfig
from app.utils.container._general import _strip_ansi_escape_sequences
from app.utils.system import get_current_user_identity

client = docker.from_env()
settings = get_settings().docker


def _check_image(image_cfg: ImageConfig) -> None:
    try:
        client.images.get(image_cfg.get_image())
    except ImageNotFound:
        logger.error(f'image {image_cfg.get_image()} not found, pull from {image_cfg.registry}')
        if image_cfg.registry == 'docker.io':
            client.images.pull(
                f'{image_cfg.registry}/{image_cfg.namespace}/{image_cfg.repository}',
                image_cfg.tag,
                auth_config={
                    'username': settings.USERNAME,
                    'password': settings.PASSWORD.get_secret_value(),
                },
            )
        else:
            client.images.pull(f'{image_cfg.registry}/{image_cfg.namespace}/{image_cfg.repository}', image_cfg.tag)
        time.sleep(1)


def run_docker(
    image_cfg: ImageConfig,
    command: str,
    mounts: list[Mount],
    workdir: Optional[str] = None,
    *,
    ensure_ascii: bool = True,
    user: str | None = None,
    **kwargs,
) -> tuple[int, str]:
    """在docker容器中运行命令并返回执行结果.

    同步运行容器命令, 等待命令执行完成后返回退出状态码和输出日志。
    适用于短时间运行的命令, 容器会在命令执行完成后自动清理。

    Args:
        image_cfg (ImageConfig): 容器镜像设置
        command (str): 容器内执行的命令
        mounts (list[Mount]): 文件挂载列表
        workdir: 命令运行的工作目录
        ensure_ascii: 是否过滤命令行富文本格式
        user: 传递给 Docker 的用户身份, 通常为 ``UID:GID``
        **kwargs: 其他容器参数, 如环境变量、端口映射等

    Returns:
        tuple[int, str]: (容器退出状态码, 命令输出日志)
            - 退出状态码: 0表示成功, 非0表示失败
            - 输出日志: 命令的标准输出和错误输出

    Raises:
        docker.errors.ContainerError: 容器运行失败时抛出
        docker.errors.ImageNotFound: 镜像不存在且拉取失败时抛出
    """
    _check_image(image_cfg)

    # 处理 GPU 配置
    device_requests = None
    if settings.ENABLE_GPU:
        gpu_count = -1 if settings.GPU_DEVICES == 'all' else len(settings.GPU_DEVICES.split(','))
        device_requests = [DeviceRequest(count=gpu_count, capabilities=[['gpu']])]
        if settings.GPU_DEVICES != 'all':
            device_requests[0]['device_ids'] = settings.GPU_DEVICES.split(',')
    if user is not None:
        kwargs['user'] = user

    # pyrefly: ignore [no-matching-overload]
    container = client.containers.run(
        image_cfg.get_image(),
        [command],
        stderr=True,
        entrypoint=['/bin/sh', '-c'],
        working_dir=workdir,
        auto_remove=False,
        detach=True,
        mounts=mounts,
        device_requests=device_requests,
        **kwargs,
    )

    try:
        status_code = container.wait()['StatusCode']
        result = container.logs(stderr=True)
        result = result.decode('utf-8').rstrip()

        if ensure_ascii:
            return status_code, _strip_ansi_escape_sequences(result)
        return status_code, result
    finally:
        if container:
            container.remove(force=True)


def run_docker_background(
    image_cfg: ImageConfig,
    command: str,
    mounts: list[Mount],
    working_dir: str,
    interval: float = 1.0,
    **kwargs: int,
) -> Generator[dict[str, Any] | None, None, None]:
    """后台运行docker容器, 并实时返回容器资源占用信息.

    在容器内运行长耗时任务, 定期采样并返回容器的资源使用情况,
    包括CPU、内存、网络I/O、块I/O等关键指标。

    Args:
        image_cfg: 容器镜像设置
        command: 容器内执行的命令
        mounts: 文件挂载列表
        working_dir: 工作目录
        interval: 资源采样间隔 (秒), 默认1.0秒
        **kwargs: 其他容器参数

    Yields:
        dict[str, Any]: 当前时刻容器的资源占用信息 (如CPU、内存等)

    Raises:
        docker.errors.ContainerError: 容器运行失败时抛出
        docker.errors.ImageNotFound: 镜像不存在且拉取失败时抛出
    """
    _check_image(image_cfg)
    container = None

    # 处理 GPU 配置
    device_requests = None
    if settings.ENABLE_GPU:
        gpu_count = -1 if settings.GPU_DEVICES == 'all' else len(settings.GPU_DEVICES.split(','))
        device_requests = [DeviceRequest(count=gpu_count, capabilities=[['gpu']])]
        if settings.GPU_DEVICES != 'all':
            device_requests[0]['device_ids'] = settings.GPU_DEVICES.split(',')

    try:
        # pyrefly: ignore [no-matching-overload]
        container = client.containers.run(
            image_cfg.get_image(),
            [command],
            mounts=mounts,
            entrypoint=['/bin/sh', '-c'],
            working_dir=working_dir,
            user=get_current_user_identity(),
            auto_remove=False,
            detach=True,
            device_requests=device_requests,
            **kwargs,
        )

        while True:
            container.reload()
            if container.status == 'exited':
                break
            stats = container.stats(stream=False)
            yield calculate_docker_stats(stats)
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
            container.remove(force=True)


def create_container(image_cfg: ImageConfig, mounts: list[Mount | dict], **kwargs: int) -> str:
    """创建容器并返回其ID.

    Args:
        image_cfg: 所使用的镜像
        mounts: 外部文件夹挂载
        **kwargs:

    Returns:
        容器的ID

    Raises:
        RuntimeError: 容器启动失败时抛出
    """
    _check_image(image_cfg)

    # 处理 GPU 配置
    device_requests = None
    if settings.ENABLE_GPU:
        gpu_count = -1 if settings.GPU_DEVICES == 'all' else len(settings.GPU_DEVICES.split(','))
        device_requests = [DeviceRequest(count=gpu_count, capabilities=[['gpu']])]
        if settings.GPU_DEVICES != 'all':
            device_requests[0]['device_ids'] = settings.GPU_DEVICES.split(',')

    container = client.containers.create(
        image_cfg.get_image(),
        command='"tail -f /dev/null"',  # 保证容器常驻
        # pyrefly: ignore [bad-argument-type]
        mounts=mounts,
        entrypoint=['/bin/sh', '-c'],
        detach=True,
        user=get_current_user_identity(),
        device_requests=device_requests,
        # pyrefly: ignore [bad-argument-type]
        **kwargs,
    )
    container.start()

    # 等待容器稳定并检查状态
    time.sleep(0.5)
    container.reload()

    if container.status != 'running':
        logs = container.logs().decode('utf-8', errors='replace')
        container.remove(force=True)
        err_msg = f'Container startup failed, status: {container.status}, image: {image_cfg.get_image()}, logs: {logs}'
        raise RuntimeError(err_msg)

    logger.debug(f'create container {container.name}')

    # pyrefly: ignore [bad-return]
    return container.id


def run_docker_by_id(container_id: str, command: str, workdir: Optional[str] = None) -> tuple[int, str]:
    """在特定容器中运行命令并返回结果.

    Args:
        container_id: 目标容器ID
        command: 要运行的指令
        workdir: 命令执行的工作目录

    Returns:
        - int: 运行的状态码
        - str: 日志输出结果 (去除 ANSI 转义序列)
    """
    container = client.containers.get(container_id)

    code, result = container.exec_run(['/bin/sh', '-c', command], tty=False, stream=False, workdir=workdir)
    if isinstance(result, bytes):
        try:
            result = result.decode()
        except UnicodeDecodeError as e:
            result = f'{e}'

    # pyrefly: ignore [missing-attribute]
    result = _strip_ansi_escape_sequences(result.strip())
    # pyrefly: ignore [bad-return]
    return code, result


def remove_container(container_id: str) -> None:
    """停止并移除容器.

    Args:
        container_id: 要移除的容器ID
    """
    try:
        container = client.containers.get(container_id)
        container.remove(force=True)
    except docker.errors.NotFound:
        pass
    except requests.exceptions.ReadTimeout:
        pass


def calculate_docker_stats(stats: dict[str, Any]) -> Optional[dict[str, Any]]:
    """提取容器日志中的监控内容.

    从 Docker stats 输出中提取关键的监控指标, 包括CPU利用率、内存利用率、
    实际内存占用、网络I/O状态、块I/O状态等。

    Args:
        stats (dict[str, Any]): Docker stats 原始输出数据

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
