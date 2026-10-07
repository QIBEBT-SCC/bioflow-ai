"""Bounded shell inspection of a read-only authoring workspace."""

import asyncio
import os
from contextlib import suppress
from pathlib import Path, PurePosixPath
from uuid import uuid4

from anyio import Path as AsyncPath

from bioflow_ai.infrastructure.containers import (
    ContainerMount,
    ContainerProcessSpec,
    ContainerTmpfs,
    build_container_process_launch,
    remove_container_process,
    resolve_container_user,
)
from bioflow_ai.schemas.tool_config import ImageConfig

UTILS_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-utils:1.2.0')
MAX_OUTPUT_BYTES = 16_000
MAX_COMMAND_TIMEOUT = 120


async def _collect_output(process: asyncio.subprocess.Process) -> tuple[int, str]:
    captured = bytearray()
    truncated = False
    if process.stdout is None:
        msg = 'Container output pipe was not created.'
        raise RuntimeError(msg)
    while chunk := await process.stdout.read(65_536):
        remaining = MAX_OUTPUT_BYTES - len(captured)
        captured.extend(chunk[:remaining])
        truncated |= len(chunk) > remaining
    code = await process.wait()
    output = captured.decode('utf-8', errors='replace')
    if truncated:
        output += '\n[Output truncated; narrow the command with rg, sed, head, or tail.]'
    return code, output


async def run_workspace_command(
    workspace: Path, command: str, workdir: str = '/workspace', timeout: int = 30
) -> tuple[int, str]:
    """Run an inspection command with no writable host mounts and bounded resources."""
    directory = PurePosixPath(workdir)
    if not directory.is_absolute() or not directory.is_relative_to('/workspace') or '..' in directory.parts:
        msg = 'workdir must be an absolute path inside /workspace.'
        raise ValueError(msg)
    if not 1 <= timeout <= MAX_COMMAND_TIMEOUT:
        msg = 'timeout must be between 1 and 120 seconds.'
        raise ValueError(msg)
    root = await AsyncPath(workspace).resolve(strict=True)
    if not await root.is_dir():
        msg = 'The workspace must be a directory.'
        raise ValueError(msg)
    identity = resolve_container_user(1000, 1000)
    name = f'bioflow-code-read-{uuid4().hex}'
    spec = ContainerProcessSpec(
        image=UTILS_IMAGE,
        command=('/bin/bash', '-c', command),
        name=name,
        mounts=(ContainerMount(target='/workspace', source=str(root), read_only=True),),
        tmpfs=(ContainerTmpfs('/tmp', ('rw', 'nosuid', 'nodev', 'size=64m', 'mode=1777')),),  # noqa: S108
        environment={'HOME': '/tmp'},  # noqa: S108
        workdir=str(directory),
        user=identity.identity,
        podman_keep_id=identity.podman_keep_id,
        read_only=True,
        drop_all_capabilities=True,
        no_new_privileges=True,
        pids_limit=64,
        memory_limit='256m',
        cpu_limit=1,
    )
    launch = build_container_process_launch(spec)
    process = await asyncio.create_subprocess_exec(
        *launch.argv,
        env={**os.environ, **launch.host_environment},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        return await asyncio.wait_for(_collect_output(process), timeout=timeout)
    except (TimeoutError, asyncio.CancelledError):
        await asyncio.shield(remove_container_process(name))
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        raise
