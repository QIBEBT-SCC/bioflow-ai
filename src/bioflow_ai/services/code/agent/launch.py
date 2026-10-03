"""Provider-specific ACP container process configuration."""

import json
from pathlib import Path

from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.containers import (
    ContainerMount,
    ContainerProcessLaunch,
    ContainerProcessSpec,
    ContainerTmpfs,
    build_container_process_launch,
    resolve_container_user,
)
from bioflow_ai.schemas.coding_agent import CodingAgentProvider
from bioflow_ai.schemas.tool_config import ImageConfig
from bioflow_ai.services.code.agent.settings import read_settings

CODEX_IMAGE = ImageConfig.from_str('docker.io/aye1032/codex-acp:1.10.0-bioflow.2')
OPENCODE_IMAGE = ImageConfig.from_str('docker.io/aye1032/opencode-acp:1.18.29-bioflow')


def build_agent_launch(
    name: str,
    workspace: Path,
    auth_home: Path,
    provider: CodingAgentProvider = 'codex',
) -> ContainerProcessLaunch:
    """Build the configured runtime invocation for one ACP provider."""
    workspace_path = workspace.resolve(strict=True)
    resolved_auth = auth_home.resolve(strict=True)
    if (
        resolved_auth == workspace_path
        or resolved_auth in workspace_path.parents
        or workspace_path in resolved_auth.parents
    ):
        message = 'Agent credentials and the workspace must be separate directories'
        raise ValueError(message)

    identity = resolve_container_user(65532, 65532)
    agent_home = '/home/codex' if provider == 'codex' else '/home/agent'
    mounts = [ContainerMount(source=str(workspace_path), target='/workspace', read_only=False)]
    environment = {'HOME': agent_home}
    if provider == 'codex':
        mounts.append(ContainerMount(source=str(resolved_auth), target='/home/codex/.codex', read_only=False))
        environment.update(
            CODEX_HOME='/home/codex/.codex',
            NO_BROWSER='1',
            CODEX_CONFIG=json.dumps(read_settings().codex_config()),
        )
    else:
        mounts.append(
            ContainerMount(
                source=str(resolved_auth / 'auth.json'),
                target='/credentials/auth.json',
                read_only=True,
            )
        )
        if (resolved_auth / 'opencode.json').is_file():
            mounts.append(
                ContainerMount(
                    source=str(resolved_auth / 'opencode.json'),
                    target='/etc/opencode.json',
                    read_only=True,
                )
            )
            environment['OPENCODE_CONFIG'] = '/etc/opencode.json'

    config = get_settings().code_agent
    specs = {
        'codex': (CODEX_IMAGE, ('codex-acp',)),
        'opencode': (OPENCODE_IMAGE, ('opencode', 'acp')),
    }
    image, command = specs[provider]
    return build_container_process_launch(
        ContainerProcessSpec(
            image=image,
            command=command,
            name=name,
            mounts=tuple(mounts),
            tmpfs=(
                ContainerTmpfs('/tmp', ('rw', 'noexec', 'nosuid', 'size=64m')),  # noqa: S108
                ContainerTmpfs(agent_home, ('rw', 'nosuid', 'size=256m', 'mode=1777')),
            ),
            environment=environment,
            workdir='/workspace',
            user=identity.identity,
            podman_keep_id=identity.podman_keep_id,
            interactive=True,
            read_only=True,
            drop_all_capabilities=True,
            no_new_privileges=True,
            pids_limit=config.PID_LIMIT,
            memory_limit=config.MEMORY_LIMIT,
            cpu_limit=config.CPU_LIMIT,
            use_proxy=get_settings().network.USE_PROXY,
        )
    )
