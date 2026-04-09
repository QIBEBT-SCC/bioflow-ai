from docker.types import Mount
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.tools import ImageConfig


class ContainerMount(BaseModel):
    target: str = Field(description='Target path in container, a suitably named subdirectory under /data/')
    source: str = Field(description='Source path on host')
    read_only: bool


def create_container(image: ImageConfig, mounts: list[ContainerMount]) -> str:
    """Create container by image."""
    mount_configs = [{'type': 'bind', 'target': m.target, 'source': m.source, 'read_only': m.read_only} for m in mounts]

    if get_settings().docker.CONTAINER == 'podman':
        from app.utils.container.podman import create_container

        # Podman specific relabeling
        for m in mount_configs:
            m['relabel'] = 'z'

        container_id = create_container(image, mount_configs)
    else:
        from app.utils.container.docker import create_container

        final_mounts = [
            Mount(
                target=m['target'],
                source=m['source'],
                type='bind',
                read_only=m.get('read_only', False),
            )
            for m in mount_configs
        ]
        container_id = create_container(image, final_mounts)

    return container_id


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


def run_command_in_container(container_id: str, command: str, workdir: str | None = None) -> tuple[int, str]:
    """在已经创建的容器中运行命令."""
    if get_settings().docker.CONTAINER == 'docker':
        from app.utils.container.docker import run_docker_by_id

        code, out = run_docker_by_id(container_id, command, workdir)
    else:
        from app.utils.container.podman import run_podman_by_id

        code, out = run_podman_by_id(container_id, command, workdir)

    return code, out


def remove_container(container_id: str) -> None:
    """移除指定容器."""
    if get_settings().docker.CONTAINER == 'docker':
        from app.utils.container.docker import remove_container

        remove_container(container_id)
    else:
        from app.utils.container.podman import remove_container

        remove_container(container_id)


__all__ = [
    'ContainerMount',
    'create_container',
    'remove_container',
    'run_command',
    'run_command_background',
    'run_command_in_container',
]
