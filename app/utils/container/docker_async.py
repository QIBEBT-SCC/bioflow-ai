import asyncio
from typing import Any, AsyncGenerator, Optional

import aiodocker
from docker.types import Mount
from loguru import logger

from app.core.config import get_settings
from app.models.tools import ImageConfig
from app.utils.container._general import _strip_ansi_escape_sequences, calculate_container_stats
from app.utils.system import get_current_user_identity


async def _check_image(client: aiodocker.Docker, image_cfg: ImageConfig):
    """检查镜像是否存在 (异步)

    Args:
        client: aiodocker client
        image_cfg: 镜像设置
    """
    image_name = image_cfg.get_image()
    try:
        await client.images.inspect(image_name)
    except aiodocker.exceptions.DockerError:
        logger.error(f'image {image_name} not found, pull from {image_cfg.registry}')
        auth_config = None
        if image_cfg.registry == 'docker.io':
            auth_config = {
                'username': get_settings().docker.USERNAME,
                'password': get_settings().docker.PASSWORD.get_secret_value(),
            }

        # aiodocker pull returns a stream of progress
        await client.images.pull(image_name, auth=auth_config)


async def run_docker(image_cfg: ImageConfig, command: str, mounts: list[Mount], **kwargs) -> tuple[int, str]:
    """在docker容器中运行命令并返回执行结果 (异步)。

    Args:
        image_cfg: 容器镜像设置
        command: 容器内执行的命令
        mounts: 文件挂载列表
        **kwargs: 其他容器参数

    Returns:
        tuple[int, str]: (容器退出状态码, 命令输出日志)
    """
    async with aiodocker.Docker() as client:
        await _check_image(client, image_cfg)

        # Convert Mount objects to bind strings if necessary, or use HostConfig binds
        # aiodocker uses 'HostConfig': {'Binds': [...]} usually
        binds = [f'{mount["source"]}:{mount["target"]}:{mount.get("type", "rw")}' for mount in mounts] if mounts else []

        container_config = {
            'Image': image_cfg.get_image(),
            'Cmd': ['/bin/bash', '-c', command],
            'User': str(get_current_user_identity()),
            'HostConfig': {
                'Binds': binds,
                'AutoRemove': False,
            },
            **kwargs,
        }

        container = await client.containers.create(config=container_config)

        try:
            await container.start()

            # Wait for container to finish
            result = await container.wait()
            status_code = result.get('StatusCode', 0)

            logs = await container.log(stdout=True, stderr=True)
            # logs is a list of strings in aiodocker
            log_str = ''.join(logs).rstrip()

            return status_code, log_str
        finally:
            await container.delete(force=True)


async def run_docker_background(
    image_cfg: ImageConfig, command: str, mounts: list[Mount], interval: float = 1.0, **kwargs
) -> AsyncGenerator[dict[str, Any], None]:
    """后台运行docker容器，并实时返回容器资源占用信息 (异步)。

    Args:
        image_cfg: 容器镜像设置
        command: 容器内执行的命令
        mounts: 文件挂载列表
        interval: 资源采样间隔（秒）
        **kwargs: 其他容器参数

    Yields:
        dict[str, Any]: 当前时刻容器的资源占用信息
    """
    async with aiodocker.Docker() as client:
        await _check_image(client, image_cfg)

        binds = [f'{mount["source"]}:{mount["target"]}:{mount.get("type", "rw")}' for mount in mounts] if mounts else []

        container_config = {
            'Image': image_cfg.get_image(),
            'Cmd': ['/bin/bash', '-c', command],
            'User': str(get_current_user_identity()),
            'HostConfig': {
                'Binds': binds,
                'AutoRemove': False,
            },
            **kwargs,
        }

        container = await client.containers.create(config=container_config)

        try:
            await container.start()

            while True:
                # Check if container is still running
                info = await container.show()
                if info['State']['Status'] == 'exited':
                    break

                # Get stats
                # stream=False returns a single snapshot
                stats = await container.stats(stream=False)
                # stats is a list with one element when stream=False in some versions, or a dict
                # aiodocker docs say: "Get container stats. Returns a stream if stream parameter is True."
                # If stream=False, it should return a single dict or list containing it.
                # Checking source/docs behavior: aiodocker might return list of 1 item.

                stat_data = stats[0] if isinstance(stats, list) else stats

                calculated = calculate_container_stats(stat_data)
                if calculated:
                    yield calculated

                await asyncio.sleep(interval)

        finally:
            # Ensure container is removed
            try:
                await container.delete(force=True)
            except Exception:
                pass


async def create_container(image_cfg: ImageConfig, mounts: list[Mount | dict], debug: bool = False, **kwargs) -> str:
    """创建容器并返回其ID (异步)

    Args:
        image_cfg: 所使用的镜像
        mounts: 外部文件夹挂载
        debug: 是否显示调试信息
        **kwargs:

    Returns:
        容器的ID
    """
    # Note: We create a new client here but we must close it or use context manager.
    # Since we return ID and don't keep client open, we can use context manager for creation.
    async with aiodocker.Docker() as client:
        await _check_image(client, image_cfg)

        binds = [f'{mount["source"]}:{mount["target"]}:{mount.get("type", "rw")}' for mount in mounts] if mounts else []

        container_config = {
            'Image': image_cfg.get_image(),
            'Cmd': ['/bin/bash', '-c', 'sleep infinity'],
            'User': str(get_current_user_identity()),
            'HostConfig': {
                'Binds': binds,
            },
            **kwargs,
        }

        container = await client.containers.create(config=container_config)
        await container.start()

        if debug:
            logger.debug(f'create container {container.id}')

        return container.id


async def run_docker_by_id(container_id: str, command: str, workdir: Optional[str] = None) -> tuple[int, str]:
    """在特定容器中运行命令并返回结果 (异步)

    Args:
        container_id: 目标容器ID
        command: 要运行的指令
        workdir: 命令执行的工作目录

    Returns:
        - int: 运行的状态码
        - str: 日志输出结果
    """
    async with aiodocker.Docker() as client:
        try:
            container = await client.containers.get(container_id)

            exec_config = {
                'Cmd': ['/bin/bash', '-c', command],
                'AttachStdout': True,
                'AttachStderr': True,
                'Tty': True,
            }
            if workdir:
                exec_config['WorkingDir'] = workdir

            exec_obj = await container.exec(exec_config)

            # start() with detach=False returns a BytesStream or similar
            stream = await exec_obj.start(detach=False)

            # Read all output
            output_bytes = await stream.read_out()
            # output_bytes might be bytes or list of bytes.

            if isinstance(output_bytes, list):
                output_data = b''.join(output_bytes)
            else:
                output_data = output_bytes

            # Get exit code
            inspect_result = await exec_obj.inspect()
            exit_code = inspect_result.get('ExitCode', 0)

            try:
                result_str = output_data.decode('utf-8')
            except UnicodeDecodeError as e:
                result_str = str(e)

            result = _strip_ansi_escape_sequences(result_str.strip())
            return exit_code, result

        except aiodocker.exceptions.DockerError as e:
            logger.error(f'Error running command in container {container_id}: {e}')
            return -1, str(e)


async def remove_container(container_id: str):
    """停止并移除容器 (异步)

    Args:
        container_id: 要移除的容器ID
    """
    async with aiodocker.Docker() as client:
        try:
            container = await client.containers.get(container_id)
            await container.delete(force=True)
        except aiodocker.exceptions.DockerError:
            pass
