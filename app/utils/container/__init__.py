from docker.types import Mount
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.schemas.tool_config import ImageConfig


class ContainerMount(BaseModel):
    target: str = Field(description='Target path in container, a suitably named subdirectory under /data/')
    source: str = Field(description='Source path on host')
    read_only: bool


def _convert_to_podman_mount(mounts: list[ContainerMount]) -> list[dict]:
    return [
        {'type': 'bind', 'target': m.target, 'source': m.source, 'read_only': m.read_only, 'relabel': 'z'}
        for m in mounts
    ]


def _convert_to_docker_mount(mounts: list[ContainerMount]) -> list[Mount]:
    return [Mount(target=m.target, source=m.source, type='bind', read_only=m.read_only) for m in mounts]


def create_container(image: ImageConfig, mounts: list[ContainerMount]) -> str:
    """Create container by image."""
    if get_settings().docker.CONTAINER == 'podman':
        from app.utils.container.podman import create_container

        container_id = create_container(image, _convert_to_podman_mount(mounts))
    else:
        from app.utils.container.docker import create_container

        container_id = create_container(image, _convert_to_docker_mount(mounts))

    return container_id


def run_command(
    image_cfg: ImageConfig,
    mounts: list[ContainerMount],
    command: str,
    workdir: str | None = None,
    *,
    ensure_ascii: bool = True,
    user: str | None = None,
) -> tuple[int, str]:
    """Execute shell commands within a specified container environment.

    Args:
        image_cfg: An image configuration object containing the name of the image to run and related metadata.
        mounts: A list of mount configurations defining file/directory
        command: The full command string to be executed inside the container.
        workdir: The working directory of command execution.
        ensure_ascii: Whether to ensure the output is ASCII-encoded. Defaults to True.
        user: User identity passed to the container runtime, typically ``UID:GID``.

    Returns:
        A tuple containing two elements:
            - status (int): The exit status code after command execution (0 typically indicates success).
            - output (str): The standard output and standard error content after command execution.
    """
    if get_settings().docker.CONTAINER == 'podman':
        from app.utils.container.podman import run_podman

        status, output = run_podman(image_cfg, command, _convert_to_podman_mount(mounts), workdir, user=user)
    else:
        from app.utils.container.docker import run_docker

        status, output = run_docker(
            image_cfg,
            command,
            _convert_to_docker_mount(mounts),
            workdir,
            ensure_ascii=ensure_ascii,
            user=user,
        )

    return status, output


def run_command_background(  # noqa: ANN201
    image_cfg: ImageConfig,
    command: str,
    mounts: list[ContainerMount],
    working_dir: str,
    interval: float = 1.0,
    *,
    user: str | None = None,
):
    """后台运行容器命令, 实时 yield 资源占用统计, 自动分发到 podman 或 docker."""
    if get_settings().docker.CONTAINER == 'podman':
        from app.utils.container.podman import run_podman_background

        yield from run_podman_background(
            image_cfg,
            command,
            _convert_to_podman_mount(mounts),
            interval=interval,
            workdir=working_dir,
            user=user,
        )
    else:
        from app.utils.container.docker import run_docker_background

        yield from run_docker_background(
            image_cfg,
            command,
            _convert_to_docker_mount(mounts),
            working_dir,
            interval=interval,
            user=user,
        )


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
