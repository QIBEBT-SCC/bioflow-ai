from docker.types import Mount
from loguru import logger

from app.core.config import get_settings
from app.models.tools import ImageConfig


def run_command(
    image_cfg: ImageConfig,
    mounts: list[Mount],
    command: str,
    *,
    ensure_ascii: bool = True,
):
    logger.debug(f'Running  command "{command}" by image {image_cfg.get_image()}')
    if get_settings().docker.CONTAINER == 'podman':
        from app.utils.container.podman import run_podman  # noqa: PLC0415

        status, output = run_podman(
            image_cfg,
            command,
            mounts,
        )
    else:
        from app.utils.container.docker import run_docker  # noqa: PLC0415

        status, output = run_docker(image_cfg, command, mounts, ensure_ascii=ensure_ascii)

    return status, output


__all__ = ['run_command']
