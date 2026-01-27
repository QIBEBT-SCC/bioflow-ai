from docker.types import Mount

from app.core.config import get_settings
from app.models.tools import ImageConfig


def run_command(image_cfg: ImageConfig, mounts: list[Mount], command: str):
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
