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
) -> tuple[int, str]:
    """Execute shell commands within a specified container environment.

    Args:
        image_cfg: An image configuration object containing the name of the image to run and related metadata.
        mounts: A list of mount configurations defining file/directory
        command: The full command string to be executed inside the container.
        ensure_ascii: Whether to ensure the output is ASCII-encoded. Defaults to True.

    Returns:
        A tuple containing two elements:
            - status (int): The exit status code after command execution (0 typically indicates success).
            - output (str): The standard output and standard error content after command execution.
    """
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

        status, output = run_docker(image_cfg, command, mounts, ensure_ascii=ensure_ascii)

    return status, output


def run_command_background(  # noqa: ANN201
    image_cfg: ImageConfig,
    command: str,
    mounts: list[Mount],
    working_dir: str,
    interval: float = 1.0,
):
    """后台运行容器命令, 实时 yield 资源占用统计, 自动分发到 podman 或 docker."""
    if get_settings().docker.CONTAINER == 'podman':
        from app.utils.container.podman import run_podman_background

        yield from run_podman_background(image_cfg, command, mounts, interval=interval, workdir=working_dir)
    else:
        from app.utils.container.docker import run_docker_background

        yield from run_docker_background(image_cfg, command, mounts, working_dir, interval=interval)


def remove_container(container_id: str) -> None:
    """移除指定容器."""
    if get_settings().docker.CONTAINER == 'docker':
        from app.utils.container.docker import remove_container

        remove_container(container_id)
    else:
        from app.utils.container.podman import remove_container

        remove_container(container_id)


__all__ = ['remove_container', 'run_command', 'run_command_background']
