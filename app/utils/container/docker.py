import datetime
import time
from typing import Optional

import docker
from docker import DockerClient
from docker.errors import ImageNotFound
from docker.types import Mount
from loguru import logger

from app.core.config import get_settings
from app.models.tools import ImageConfig
from app.utils.container._general import _strip_ansi_escape_sequences, calculate_container_stats
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


def run_docker(
    image_cfg: ImageConfig, command: str, mounts: list[Mount], workdir: Optional[str] = None, **kwargs
) -> tuple[int, str]:
    """在docker容器中运行命令并返回执行结果。

    同步运行容器命令，等待命令执行完成后返回退出状态码和输出日志。
    适用于短时间运行的命令，容器会在命令执行完成后自动清理。

    Args:
        image_cfg (ImageConfig): 容器镜像设置
        command (str): 容器内执行的命令
        mounts (list[Mount]): 文件挂载列表
        workdir: 命令运行的工作目录
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
        f'"{command}"',
        stderr=True,
        entrypoint=['/bin/bash', '-c'],
        working_dir=workdir,
        user=get_current_user_identity(),
        auto_remove=False,
        detach=True,
        *kwargs,
    )

    try:
        status_code = container.wait()['StatusCode']
        result = container.logs(stderr=True)
        result = result.decode('utf-8').rstrip()

        return status_code, _strip_ansi_escape_sequences(result)
    finally:
        if container:
            container.remove(force=True)


def run_docker_background(
    image_cfg: ImageConfig, command: str, mounts: list[Mount], working_dir: str, interval: float = 1.0, **kwargs
):
    """后台运行docker容器，并实时返回容器资源占用信息。

    在容器内运行长耗时任务，定期采样并返回容器的资源使用情况，
    包括CPU、内存、网络I/O、块I/O等关键指标。

    Args:
        image_cfg: 容器镜像设置
        command: 容器内执行的命令
        mounts: 文件挂载列表
        working_dir: 工作目录
        interval: 资源采样间隔（秒），默认1.0秒
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
            working_dir=working_dir,
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


def create_container(image_cfg: ImageConfig, mounts: list[Mount | dict], **kwargs) -> str:
    """创建容器并返回其ID

    Args:
        image_cfg: 所使用的镜像
        mounts: 外部文件夹挂载
        **kwargs:

    Returns:
        容器的ID
    """
    client = docker.from_env()
    _check_image(client, image_cfg)

    container = client.containers.create(
        image_cfg.get_image(),
        command='"sleep infinity"',  # 保证容器常驻
        mounts=mounts,
        entrypoint=['/bin/bash', '-c'],
        detach=True,
        user=get_current_user_identity(),
        **kwargs,
    )
    container.start()
    logger.debug(f'create container {container.name}')

    return container.id


def run_docker_by_id(container_id: str, command: str, workdir: Optional[str] = None) -> tuple[int, str]:
    """在特定容器中运行命令并返回结果

    Args:
        container_id: 目标容器ID
        command: 要运行的指令
        workdir: 命令执行的工作目录

    Returns:
        - int: 运行的状态码
        - str: 日志输出结果（已去除 ANSI 转义序列）
    """
    client = docker.from_env()
    container = client.containers.get(container_id)

    code, result = container.exec_run(['/bin/bash', '-c', command], tty=True, stream=False, workdir=workdir)
    if isinstance(result, bytes):
        try:
            result = result.decode()
        except UnicodeDecodeError as e:
            result = f'{e}'

    result = _strip_ansi_escape_sequences(result.strip())
    return code, result


def remove_container(container_id: str):
    """停止并移除容器

    Args:
        container_id: 要移除的容器ID
    """
    client = docker.from_env()
    try:
        container = client.containers.get(container_id)
        container.remove(force=True)
    except docker.errors.NotFound:
        pass
