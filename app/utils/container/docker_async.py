import aiodocker
from docker.types import Mount

from app.core.config import get_settings
from app.models.tools import ImageConfig

from app.utils.system import get_current_user_identity


async def arun_docker_background(image: str, commands: list[str], mounts: list[Mount]) -> None:
    """docker异步运行封装

    Args:
        image: 镜像
        commands: 指令列表
        mounts: 宿主目录映射列表
    """

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


async def get_help_doc_async(image_cfg: ImageConfig, help_command: str) -> str:
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
