import os
import time

from loguru import logger

from app.utils.config import ImageConfig


def get_current_user_identity() -> str:
    """获取当前用户的UID:GID

    Returns:
        str: 格式为 "UID:GID" 的字符串
    """
    uid = os.getuid()
    gid = os.getgid()
    return f'{uid}:{gid}'


async def run_container_async(image: str, commands: list[str], volumes_binds: list[str]) -> None:
    """docker异步运行封装

    Args:
        image: 镜像
        commands: 指令列表
        volumes_binds: 宿主目录映射列表
    """
    import aiodocker

    async with aiodocker.Docker() as docker:
        await docker.pull(image)

        container_config = {
            'Image': image,
            'Cmd': commands,
            'HostConfig': {'Binds': volumes_binds},
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


def run_container(
    image_cfg: ImageConfig, commands: list[str], volumes_binds: list[str], auto_remove: bool = True
):
    import docker
    from docker.errors import ImageNotFound

    client = docker.from_env()
    try:
        client.containers.run(
            image_cfg.get_image(),
            commands,
            volumes=volumes_binds,
            user=get_current_user_identity(),
            auto_remove=auto_remove,
        )
    except ImageNotFound:
        logger.error(f'image {image_cfg.get_image()} not found, pull from docker hub.')
        client.images.pull(image_cfg.REPOSITORY, image_cfg.TAG)
        time.sleep(1)
        client.containers.run(
            image_cfg.get_image(),
            commands,
            volumes=volumes_binds,
            user=get_current_user_identity(),
            auto_remove=auto_remove,
        )

