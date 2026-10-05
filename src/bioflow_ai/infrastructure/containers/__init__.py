import asyncio
from contextlib import suppress

from docker.types import Mount

from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.containers._general import (
    CONTAINER_ROOT_USER,
    ContainerMount,
    ContainerProcessLaunch,
    ContainerProcessSpec,
    ContainerTmpfs,
    ContainerUser,
    build_process_arguments,
    resolve_container_user,
)
from bioflow_ai.schemas.tool_config import ImageConfig


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
        from bioflow_ai.infrastructure.containers.podman import create_container

        container_id = create_container(image, _convert_to_podman_mount(mounts))
    else:
        from bioflow_ai.infrastructure.containers.docker import create_container

        container_id = create_container(image, _convert_to_docker_mount(mounts))  # ty: ignore[invalid-argument-type]

    return container_id


def build_container_process_launch(spec: ContainerProcessSpec) -> ContainerProcessLaunch:
    """Build a native subprocess invocation for the configured container runtime."""
    runtime = get_settings().docker.CONTAINER
    runtime_arguments: tuple[str, ...] = ()
    if runtime == 'podman' and spec.podman_keep_id is not None:
        uid, gid = spec.podman_keep_id
        runtime_arguments = (f'--userns=keep-id:uid={uid},gid={gid}',)
    arguments, host_environment = build_process_arguments(spec, runtime, runtime_arguments=runtime_arguments)
    return ContainerProcessLaunch(argv=[runtime, *arguments], host_environment=host_environment)


async def remove_container_process(identifier: str, timeout: float = 10) -> None:
    """Force-remove a container through the configured native runtime."""
    runtime = get_settings().docker.CONTAINER
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            runtime,
            'rm',
            '--force',
            identifier,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except FileNotFoundError:
        return
    except TimeoutError:
        if process is not None:
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=3)


def run_command(
    image_cfg: ImageConfig,
    mounts: list[ContainerMount],
    command: str,
    workdir: str | None = None,
    *,
    ensure_ascii: bool = True,
    user: str | None = CONTAINER_ROOT_USER,
) -> tuple[int, str]:
    """Execute shell commands within a specified container environment.

    Args:
        image_cfg: An image configuration object containing the name of the image to run and related metadata.
        mounts: A list of mount configurations defining file/directory
        command: The full command string to be executed inside the container.
        workdir: The working directory of command execution.
        ensure_ascii: Whether to ensure the output is ASCII-encoded. Defaults to True.
        user: User identity inside the container. Defaults to container root; the rootless runtime maps it to the host user that owns the daemon.

    Returns:
        A tuple containing two elements:
            - status (int): The exit status code after command execution (0 typically indicates success).
            - output (str): The standard output and standard error content after command execution.
    """
    if get_settings().docker.CONTAINER == 'podman':
        from bioflow_ai.infrastructure.containers.podman import run_podman

        status, output = run_podman(image_cfg, command, _convert_to_podman_mount(mounts), workdir, user=user)
    else:
        from bioflow_ai.infrastructure.containers.docker import run_docker

        status, output = run_docker(
            image_cfg,
            command,
            _convert_to_docker_mount(mounts),
            workdir,
            ensure_ascii=ensure_ascii,
            user=user,
        )

    return status, output


async def run_command_async(
    image_cfg: ImageConfig,
    mounts: list[ContainerMount],
    command: str,
    workdir: str | None = None,
    *,
    ensure_ascii: bool = True,
    user: str | None = CONTAINER_ROOT_USER,
) -> tuple[int, str]:
    """Run a blocking container command without blocking the asyncio event loop.

    Docker and Podman SDK calls are synchronous, including image pulls and the
    wait for command completion. Running them in a worker thread keeps unrelated
    tasks such as Redis lock renewal and agent heartbeats schedulable.

    Cancellation waits for the worker to finish so the runtime-specific
    ``finally`` block can remove the container before cancellation propagates.
    """
    worker = asyncio.create_task(
        asyncio.to_thread(
            run_command,
            image_cfg,
            mounts,
            command,
            workdir,
            ensure_ascii=ensure_ascii,
            user=user,
        ),
        name='container-command',
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        if not worker.cancelled():
            worker.exception()
        raise


def run_command_background(  # noqa: ANN201
    image_cfg: ImageConfig,
    command: str,
    mounts: list[ContainerMount],
    working_dir: str,
    interval: float = 1.0,
    *,
    user: str | None = CONTAINER_ROOT_USER,
):
    """后台运行容器命令, 实时 yield 资源占用统计, 自动分发到 podman 或 docker."""
    if get_settings().docker.CONTAINER == 'podman':
        from bioflow_ai.infrastructure.containers.podman import run_podman_background

        yield from run_podman_background(
            image_cfg,
            command,
            _convert_to_podman_mount(mounts),
            interval=interval,
            workdir=working_dir,
            user=user,
        )
    else:
        from bioflow_ai.infrastructure.containers.docker import run_docker_background

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
        from bioflow_ai.infrastructure.containers.docker import run_docker_by_id

        code, out = run_docker_by_id(container_id, command, workdir)
    else:
        from bioflow_ai.infrastructure.containers.podman import run_podman_by_id

        code, out = run_podman_by_id(container_id, command, workdir)

    return code, out


def remove_container(container_id: str) -> None:
    """移除指定容器."""
    if get_settings().docker.CONTAINER == 'docker':
        from bioflow_ai.infrastructure.containers.docker import remove_container

        remove_container(container_id)
    else:
        from bioflow_ai.infrastructure.containers.podman import remove_container

        remove_container(container_id)


__all__ = [
    'CONTAINER_ROOT_USER',
    'ContainerMount',
    'ContainerProcessLaunch',
    'ContainerProcessSpec',
    'ContainerTmpfs',
    'ContainerUser',
    'build_container_process_launch',
    'create_container',
    'remove_container',
    'remove_container_process',
    'resolve_container_user',
    'run_command',
    'run_command_async',
    'run_command_background',
    'run_command_in_container',
]
