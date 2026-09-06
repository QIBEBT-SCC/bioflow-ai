import os
import re
from dataclasses import dataclass, field
from ipaddress import ip_address
from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.schemas.tool_config import ImageConfig

CONTAINER_ROOT_USER = '0:0'
LOOPBACK_NO_PROXY = 'localhost,127.0.0.1,::1'
ContainerRuntime = Literal['docker', 'podman']


class ContainerMount(BaseModel):
    """One host path mounted into a container."""

    target: str = Field(description='Target path in the container')
    source: str = Field(description='Source path on the host')
    read_only: bool


@dataclass(frozen=True)
class ContainerTmpfs:
    """One temporary filesystem mounted into a container."""

    target: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContainerProcessSpec:
    """Runtime-neutral description of a native container subprocess."""

    image: ImageConfig
    command: tuple[str, ...]
    name: str
    mounts: tuple[ContainerMount, ...] = ()
    tmpfs: tuple[ContainerTmpfs, ...] = ()
    environment: dict[str, str] = field(default_factory=dict, repr=False)
    host_environment: dict[str, str] = field(default_factory=dict, repr=False)
    workdir: str | None = None
    user: str | None = None
    podman_keep_id: tuple[int, int] | None = None
    interactive: bool = False
    read_only: bool = False
    drop_all_capabilities: bool = False
    no_new_privileges: bool = False
    pids_limit: int | None = None
    memory_limit: str | None = None
    cpu_limit: float | None = None
    use_proxy: bool = False


@dataclass(frozen=True)
class ContainerProcessLaunch:
    """Native command and host environment used to start a container subprocess."""

    argv: list[str]
    host_environment: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class ContainerUser:
    """Container identity plus runtime-specific host ownership requirements."""

    identity: str
    host_owner: int | None
    podman_keep_id: tuple[int, int] | None


def resolve_container_user(uid: int, gid: int) -> ContainerUser:
    """Resolve a fixed non-root identity for the configured container runtime."""
    if get_settings().docker.CONTAINER == 'podman':
        return ContainerUser(identity=f'{uid}:{gid}', host_owner=None, podman_keep_id=(uid, gid))
    host_uid = os.getuid()
    host_gid = os.getgid()
    if host_uid != 0:
        return ContainerUser(identity=f'{host_uid}:{host_gid}', host_owner=None, podman_keep_id=None)
    return ContainerUser(identity=f'{uid}:{gid}', host_owner=uid, podman_keep_id=None)


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == 'localhost':
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _process_environment(spec: ContainerProcessSpec, runtime: ContainerRuntime) -> tuple[dict[str, str], bool]:
    environment = dict(spec.environment)
    if not spec.use_proxy:
        return environment, False

    network = get_settings().network
    host = network.PROXY_HOST
    uses_host_alias = _is_loopback_host(host)
    if uses_host_alias:
        host = 'host.docker.internal' if runtime == 'docker' else 'host.containers.internal'
    proxy_url = f'{network.PROXY_TYPE}://{host}:{network.PROXY_PORT}'
    environment.update(
        HTTP_PROXY=proxy_url,
        HTTPS_PROXY=proxy_url,
        ALL_PROXY=proxy_url,
        NO_PROXY=LOOPBACK_NO_PROXY,
        no_proxy=LOOPBACK_NO_PROXY,
    )
    return environment, uses_host_alias


def build_process_arguments(
    spec: ContainerProcessSpec,
    runtime: ContainerRuntime,
    *,
    runtime_arguments: tuple[str, ...] = (),
) -> tuple[list[str], dict[str, str]]:
    """Build arguments shared by native Docker and Podman subprocesses."""
    environment, uses_host_alias = _process_environment(spec, runtime)
    arguments = ['run', *runtime_arguments, '--rm']
    if spec.interactive:
        arguments.append('--interactive')
    arguments.extend(['--name', spec.name])
    if spec.read_only:
        arguments.append('--read-only')
    if spec.user is not None:
        arguments.extend(['--user', spec.user])
    if spec.drop_all_capabilities:
        arguments.extend(['--cap-drop', 'ALL'])
    if spec.no_new_privileges:
        arguments.extend(['--security-opt', 'no-new-privileges'])
    if spec.pids_limit is not None:
        arguments.extend(['--pids-limit', str(spec.pids_limit)])
    if spec.memory_limit is not None:
        arguments.extend(['--memory', spec.memory_limit])
    if spec.cpu_limit is not None:
        arguments.extend(['--cpus', str(spec.cpu_limit)])
    for tmpfs in spec.tmpfs:
        value = tmpfs.target
        if tmpfs.options:
            value += ':' + ','.join(tmpfs.options)
        arguments.extend(['--tmpfs', value])
    for mount in spec.mounts:
        access = 'ro' if mount.read_only else 'rw'
        arguments.extend(['--mount', f'type=bind,src={mount.source},dst={mount.target},{access}'])
    if spec.workdir is not None:
        arguments.extend(['--workdir', spec.workdir])
    if runtime == 'docker' and uses_host_alias:
        arguments.extend(['--add-host', 'host.docker.internal:host-gateway'])
    for key, value in environment.items():
        arguments.extend(['--env', f'{key}={value}'])
    for key in spec.host_environment:
        arguments.extend(['--env', key])
    arguments.extend([spec.image.get_image(), *spec.command])
    return arguments, dict(spec.host_environment)


def _strip_ansi_escape_sequences(text: str) -> str:
    """Remove ANSI escape sequences from text.

    ANSI escape sequences are used for terminal color and format control. When command output is captured,
    these sequences appear as garbled characters (such as [1;34m, [m, etc.).

    Args:
        text: Text containing ANSI escape sequences

    Returns:
        Text with ANSI escape sequences removed
    """
    # ANSI 转义序列的正则表达式模式
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)
