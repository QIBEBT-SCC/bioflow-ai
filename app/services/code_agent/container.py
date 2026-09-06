"""Native Docker and Podman command construction for isolated ACP adapters."""

import json
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from typing import Literal

from app.core.config import get_settings
from app.schemas.coding_agent import CodingAgentProvider
from app.services.code_agent.account import container_user
from app.services.code_agent.settings import read_settings

CODEX_IMAGE: str = 'docker.io/aye1032/codex-acp:1.10.0-bioflow.2'
OPENCODE_IMAGE: str = 'docker.io/aye1032/opencode-acp:1.18.29-bioflow'


@dataclass(frozen=True)
class ContainerLaunch:
    """One fully resolved, shell-free container launch."""

    argv: list[str]
    safe_environment: dict[str, str] = field(repr=False)


def _is_loopback_host(host: str) -> bool:
    """Return whether a configured proxy host refers to the current network namespace."""
    if host.casefold() == 'localhost':
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _proxy_url(runtime: Literal['docker', 'podman']) -> tuple[str, bool]:
    """Translate a host-loopback proxy into the native container host alias."""
    network = get_settings().network
    host = network.PROXY_HOST
    uses_host_alias = _is_loopback_host(host)
    if uses_host_alias:
        host = 'host.docker.internal' if runtime == 'docker' else 'host.containers.internal'
    return f'{network.PROXY_TYPE}://{host}:{network.PROXY_PORT}', uses_host_alias


def _common_arguments(
    name: str,
    workspace: Path,
    auth_home: Path,
    runtime: Literal['docker', 'podman'],
    provider: CodingAgentProvider = 'codex',
) -> tuple[list[str], dict[str, str]]:
    config = get_settings().code_agent
    workspace_path = str(workspace.resolve(strict=True))
    resolved_workspace = workspace.resolve(strict=True)
    resolved_auth = auth_home.resolve(strict=True)
    if (
        resolved_auth == resolved_workspace
        or resolved_auth in resolved_workspace.parents
        or resolved_workspace in resolved_auth.parents
    ):
        message = 'Agent credentials and the workspace must be separate directories'
        raise ValueError(message)
    auth_path = str(resolved_auth)
    user, _owner = container_user()
    agent_home = '/home/codex' if provider == 'codex' else '/home/agent'
    credential_target = {
        'codex': '/home/codex/.codex',
        'opencode': '/credentials',
    }[provider]
    arguments = [
        'run',
        '--rm',
        '--interactive',
        '--name',
        name,
        '--read-only',
        '--user',
        user,
        '--cap-drop',
        'ALL',
        '--security-opt',
        'no-new-privileges',
        '--pids-limit',
        str(config.PID_LIMIT),
        '--memory',
        config.MEMORY_LIMIT,
        '--cpus',
        str(config.CPU_LIMIT),
        '--tmpfs',
        '/tmp:rw,noexec,nosuid,size=64m',  # noqa: S108 - this is a container-local tmpfs mount.
        '--tmpfs',
        f'{agent_home}:rw,nosuid,size=256m,uid={user.split(":")[0]},gid={user.split(":")[1]}',
        '--mount',
        f'type=bind,src={workspace_path},dst=/workspace,rw',
        '--workdir',
        '/workspace',
        '--env',
        f'HOME={agent_home}',
    ]
    environment: dict[str, str] = {}
    if provider == 'codex':
        arguments.extend(['--mount', f'type=bind,src={auth_path},dst={credential_target},rw'])
        arguments.extend(['--env', 'CODEX_HOME=/home/codex/.codex', '--env', 'NO_BROWSER=1', '--env', 'CODEX_CONFIG'])
        environment['CODEX_CONFIG'] = json.dumps(read_settings().codex_config())
    else:
        arguments.extend(['--mount', f'type=bind,src={auth_path}/auth.json,dst={credential_target}/auth.json,ro'])
        if (resolved_auth / 'opencode.json').is_file():
            arguments.extend(
                [
                    '--mount',
                    f'type=bind,src={auth_path}/opencode.json,dst=/etc/opencode.json,ro',
                    '--env',
                    'OPENCODE_CONFIG=/etc/opencode.json',
                ]
            )
    network = get_settings().network
    if network.USE_PROXY:
        proxy_url, uses_host_alias = _proxy_url(runtime)
        if runtime == 'docker' and uses_host_alias:
            arguments.extend(['--add-host', 'host.docker.internal:host-gateway'])
        arguments.extend(['--env', 'HTTP_PROXY', '--env', 'HTTPS_PROXY', '--env', 'ALL_PROXY'])
        environment.update(HTTP_PROXY=proxy_url, HTTPS_PROXY=proxy_url, ALL_PROXY=proxy_url)
    commands = {
        'codex': [CODEX_IMAGE, 'codex-acp'],
        'opencode': [OPENCODE_IMAGE, 'opencode', 'acp'],
    }
    arguments.extend(commands[provider])
    return arguments, environment


def build_docker_launch(
    name: str, workspace: Path, auth_home: Path, provider: CodingAgentProvider = 'codex'
) -> ContainerLaunch:
    """Build a Docker-native runner invocation."""
    arguments, environment = _common_arguments(name, workspace, auth_home, 'docker', provider)
    return ContainerLaunch(argv=['docker', *arguments], safe_environment=environment)


def build_podman_launch(
    name: str, workspace: Path, auth_home: Path, provider: CodingAgentProvider = 'codex'
) -> ContainerLaunch:
    """Build a rootless Podman-native runner invocation."""
    arguments, environment = _common_arguments(name, workspace, auth_home, 'podman', provider)
    arguments[1:1] = ['--userns=keep-id:uid=65532,gid=65532']
    return ContainerLaunch(argv=['podman', *arguments], safe_environment=environment)


def build_container_launch(
    name: str, workspace: Path, auth_home: Path, provider: CodingAgentProvider = 'codex'
) -> ContainerLaunch:
    """Select the configured native container runtime without compatibility shims."""
    if get_settings().docker.CONTAINER == 'podman':
        return build_podman_launch(name, workspace, auth_home, provider)
    return build_docker_launch(name, workspace, auth_home, provider)
