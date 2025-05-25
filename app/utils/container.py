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
            'HostConfig': {
                'Binds': [f'{mount["source"]}:{mount["target"]}:rw' for mount in mounts]
            },
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


def run_docker(
    image_cfg: ImageConfig, commands: list[str], mounts: list[Mount], auto_remove: bool = True
):
    import docker
    from docker.errors import ImageNotFound

    client = docker.from_env()
    try:
        client.images.get(image_cfg.get_image())
    except ImageNotFound:
        logger.error(f'image {image_cfg.get_image()} not found, pull from docker hub.')
        client.images.pull(
            image_cfg.repository,
            image_cfg.tag,
            auth_config={
                'username': get_settings().docker.USERNAME,
                'password': get_settings().docker.PASSWORD,
            },
        )
        time.sleep(1)

    client.containers.run(
        image_cfg.get_image(),
        commands,
        mounts=mounts,
        user=get_current_user_identity(),
        auto_remove=auto_remove,
    )


def run_podman(
    image_cfg: ImageConfig, commands: list[str], mounts: list[Mount], auto_remove: bool = True
) -> str:
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
                        'password': get_settings().docker.PASSWORD,
                    },
                    progress_bar=True,
                )

            result = client.containers.run(
                image_cfg.get_image(), commands, stderr=True, auto_remove=auto_remove, mounts=mounts
            )
            return result.decode('utf-8')

    return ''
