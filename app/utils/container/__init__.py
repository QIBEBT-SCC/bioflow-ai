from docker.types import Mount
from loguru import logger

from app.core.config import get_settings
from app.models.tools import ImageConfig


def run_command(image_cfg: ImageConfig, mounts: list[Mount], command: str):
    logger.debug(f'Running  command "{command}" by image {image_cfg.get_image()}')
    if get_settings().docker.CONTAINER == 'podman':
        from app.utils.container.podman import run_podman

        status, output = run_podman(
            image_cfg,
            command,
            mounts,
        )
    else:
        from app.utils.container.docker import run_docker

        status, output = run_docker(
            image_cfg,
            command,
            mounts,
        )

    return status, output


__all__ = ['run_command']
