"""Shared CODE workspace initialization for authoring and workflow execution."""

import shlex
from pathlib import Path
from shutil import copyfile, rmtree

from bioflow_ai.core.config import get_settings
from bioflow_ai.core.paths import BUILTIN_PRESETS_DIR
from bioflow_ai.infrastructure.containers import ContainerMount, run_command, run_command_async
from bioflow_ai.infrastructure.containers.mount_paths import build_uv_mounts
from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.schemas.tool_config import ImageConfig

BASH_CODE_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-utils:1.2.0')
PYTHON_CODE_IMAGE = ImageConfig.from_str('ghcr.io/astral-sh/uv:python3.13-bookworm-slim')
R_CODE_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-r-dev:4.4.3-1.0.0')

CODE_IMAGES = {
    CodeNodeType.BASH: BASH_CODE_IMAGE,
    CodeNodeType.PYTHON: PYTHON_CODE_IMAGE,
    CodeNodeType.R: R_CODE_IMAGE,
}
SCRIPT_FILENAMES = {
    CodeNodeType.BASH: 'script.sh',
    CodeNodeType.PYTHON: 'main.py',
    CodeNodeType.R: 'script.R',
}
_LANGUAGE_DIRECTORIES = {CodeNodeType.BASH: 'bash', CodeNodeType.PYTHON: 'python', CodeNodeType.R: 'r'}


def reset_code_workspace(workspace: Path) -> None:
    """Remove the previous execution workspace and create an empty Step directory."""
    if workspace.exists():
        rmtree(workspace)
    workspace.mkdir(parents=True)


def build_workspace_mounts(
    workspace: Path,
    node_type: CodeNodeType,
    *,
    extra_mounts: list[ContainerMount] | None = None,
) -> list[ContainerMount]:
    """Mount a workspace and language caches, then any caller-supplied inputs and outputs."""
    mounts = [ContainerMount(source=str(workspace), target='/workspace', read_only=False)]
    if node_type is CodeNodeType.PYTHON:
        mounts.extend(build_uv_mounts())
    mounts.extend(extra_mounts or [])
    return mounts


def _prepare_workspace(
    workspace: Path,
    node_type: CodeNodeType,
    dependencies: list[str],
    source: str,
    extra_mounts: list[ContainerMount] | None,
) -> tuple[list[ContainerMount], str | None]:
    (workspace / 'output').mkdir(mode=0o777, exist_ok=True)
    agents = workspace / 'AGENTS.md'
    copyfile(BUILTIN_PRESETS_DIR / 'code' / _LANGUAGE_DIRECTORIES[node_type] / 'AGENTS.md', agents)
    agents.chmod(0o666)
    if source or node_type is not CodeNodeType.PYTHON:
        script = workspace / SCRIPT_FILENAMES[node_type]
        script.write_text(source, encoding='utf-8')
        script.chmod(0o666)
    mounts = build_workspace_mounts(workspace, node_type, extra_mounts=extra_mounts)

    if node_type is not CodeNodeType.PYTHON:
        return mounts, None

    index_url = str(get_settings().code.PYTHON_INDEX_URL)
    add_command = ['uv', 'add', '--no-sync', '--default-index', index_url]
    commands = [
        'uv init --app --no-package --vcs none --python "3.13"',
        shlex.join([*add_command, '--', 'click', 'loguru']),
        shlex.join([*add_command, '--dev', '--', 'ruff', 'ty', 'pytest']),
    ]
    if dependencies:
        commands.append(shlex.join([*add_command, '--', *dependencies]))
    commands.append(shlex.join(['uv', 'sync', '--default-index', index_url]))
    return mounts, ' && '.join(commands)


def _finish_workspace(
    workspace: Path,
    node_type: CodeNodeType,
    result: tuple[int, str] | None,
) -> None:
    log_path = workspace / '00_init.log'
    if result is not None:
        status, output = result
        log_path.write_text(output, encoding='utf-8')
        if status != 0:
            message = f'{node_type.name.title()} workspace initialization failed (exit {status}):\n{output}'
            raise RuntimeError(message)
    else:
        log_path.write_text(f'Initialized {node_type.name.title()} workspace.\n', encoding='utf-8')
    script = workspace / SCRIPT_FILENAMES[node_type]
    script.chmod(0o666)


def initialize_code_workspace(
    workspace: Path,
    node_type: CodeNodeType,
    dependencies: list[str] | None = None,
    *,
    source: str = '',
    extra_mounts: list[ContainerMount] | None = None,
) -> list[ContainerMount]:
    """Initialize files and the language environment, returning the mounts used for execution."""
    mounts, command = _prepare_workspace(workspace, node_type, dependencies or [], source, extra_mounts)
    result = None
    if command is not None:
        result = run_command(CODE_IMAGES[node_type], mounts, command, workdir='/workspace')
    _finish_workspace(workspace, node_type, result)
    return mounts


async def initialize_code_workspace_async(
    workspace: Path,
    node_type: CodeNodeType,
    dependencies: list[str] | None = None,
    *,
    source: str = '',
    extra_mounts: list[ContainerMount] | None = None,
) -> list[ContainerMount]:
    """Initialize the same workspace without blocking the coding agent's event loop."""
    mounts, command = _prepare_workspace(workspace, node_type, dependencies or [], source, extra_mounts)
    result = None
    if command is not None:
        result = await run_command_async(CODE_IMAGES[node_type], mounts, command, workdir='/workspace')
    _finish_workspace(workspace, node_type, result)
    return mounts
