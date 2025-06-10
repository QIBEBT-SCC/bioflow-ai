import os
import time

from docker.types import Mount
from loguru import logger

from app.core.config import get_settings
from app.models.tools import ImageConfig


def get_current_user_identity() -> str:
    """获取当前用户的UID:GID

    Returns:
        str: 格式为 "UID:GID" 的字符串
    """
    uid = os.getuid()
    gid = os.getgid()
    return f'{uid}:{gid}'


def check_image(client, image_cfg: ImageConfig):
    from docker.errors import ImageNotFound

    try:
        client.images.get(image_cfg.get_image())
    except ImageNotFound:
        logger.error(f'image {image_cfg.get_image()} not found, pull from docker hub.')
        client.images.pull(
            image_cfg.repository,
            image_cfg.tag,
            auth_config={
                'username': get_settings().docker.USERNAME,
                'password': get_settings().docker.PASSWORD.get_secret_value(),
            },
        )
        time.sleep(1)


async def run_docker_async(image: str, commands: list[str], mounts: list[Mount]) -> None:
    """docker异步运行封装

    Args:
        image: 镜像
        commands: 指令列表
        mounts: 宿主目录映射列表
    """
    import aiodocker

    async with aiodocker.Docker() as docker:
        await docker.pull(image)

        container_config = {
            'Image': image,
            'Cmd': commands,
            'HostConfig': {'Binds': [f'{mount["source"]}:{mount["target"]}:rw' for mount in mounts]},
            'User': get_current_user_identity(),
        }
        container = await docker.containers.run(config=container_config)

        try:
            await container.start()
            response = await container.wait()
            exit_code = response.get('StatusCode', 0)

            if exit_code != 0:
                logs = await container.log(stdout=True, stderr=True)
                logs_str = ''.join(logs)
                raise RuntimeError(f'Container exited with code {exit_code}. log: {logs_str}')
        finally:
            await container.delete(force=True)


def run_docker(image_cfg: ImageConfig, commands: list[str], mounts: list[Mount], **kwargs):
    import docker

    client = docker.from_env()
    check_image(client, image_cfg)

    container = client.containers.run(
        image_cfg.get_image(), commands, mounts=mounts, user=get_current_user_identity(), auto_remove=False, detach=True
    )

    try:
        while True:
            container.reload()
            if container.status == 'exited':
                break
            stats = container.stats(stream=False)
            yield stats
            time.sleep(1)
    finally:
        container.remove(force=True)


def run_podman(image_cfg: ImageConfig, commands: list[str], mounts: list[Mount], auto_remove: bool = True) -> str:
    from podman import PodmanClient
    from podman.errors import ImageNotFound

    if len(image_cfg.repository.split('/')) == 2:
        image_cfg.repository = f'docker.io/{image_cfg.repository}'

    with PodmanClient() as client:
        if client.ping():
            try:
                client.images.get(image_cfg.get_image())
            except ImageNotFound:
                logger.error(f'image {image_cfg.get_image()} not found, pull from docker hub.')
                client.images.pull(
                    image_cfg.repository,
                    image_cfg.tag,
                    auth_config={
                        'username': get_settings().docker.USERNAME,
                        'password': get_settings().docker.PASSWORD.get_secret_value(),
                    },
                    progress_bar=True,
                )

            result = client.containers.run(
                image_cfg.get_image(), commands, stderr=True, auto_remove=auto_remove, mounts=mounts
            )
            return result.decode('utf-8')

    return ''


def get_help_doc(image_cfg: ImageConfig, help_command: str) -> str:
    import docker

    client = docker.from_env()
    check_image(client, image_cfg)

    result = client.containers.run(
        image_cfg.get_image(),
        help_command,
        stderr=True,
        entrypoint=['/bin/bash', '-c'],
        user=get_current_user_identity(),
        auto_remove=True,
    )

    return result.decode('utf-8')


async def get_help_doc_async(image_cfg: ImageConfig, help_command: str) -> str:
    import aiodocker

    async with aiodocker.Docker() as docker:
        container_config = {
            'Image': image_cfg.get_image(),
            'Cmd': ['/bin/bash', '-c', help_command],
            'User': get_current_user_identity(),
        }
        container = await docker.containers.run(
            config=container_config,
            auth={
                'username': get_settings().docker.USERNAME,
                'password': get_settings().docker.PASSWORD.get_secret_value(),
            },
        )

        await container.start()
        result = ''
        async for chunk in container.log(stdout=True, stderr=True, follow=True):
            result += chunk
        await container.delete(force=True)

    return result


def calculate_container_stats(stats):
    """
    Calculate container usage statistics from runtime stats.

    Args:
        stats (dict): Container runtime stats dictionary

    Returns:
        dict: Dictionary containing calculated metrics
    """

    try:
        # Calculate CPU usage
        try:
            cpu_usage = (
                stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            )
            system_cpu_usage = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_usage / system_cpu_usage) * 100.0 if system_cpu_usage > 0 else 0.0
        except TypeError:
            cpu_percent = 0.0

        # Calculate memory usage
        try:
            mem_usage = stats['memory_stats']['usage']
            mem_limit = stats['memory_stats']['limit']
            mem_percent = (mem_usage / mem_limit) * 100.0 if mem_limit > 0 else 0.0
            mem_used_mb = mem_usage / (1024 * 1024)  # Convert to MB
        except TypeError:
            mem_percent = 0.0
            mem_used_mb = 0.0

        # Calculate IO
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

        # Get timestamp
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
