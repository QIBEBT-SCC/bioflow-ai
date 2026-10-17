"""Workspace tools for the project-prepare agent: read-only /app/data plus the read-write /work project dir."""

import asyncio
import posixpath
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import Field

from bioflow_ai.infrastructure.containers import ContainerMount, run_command_async
from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.states import ProjectPrepareState
from bioflow_ai.llm.tools.apply_patch import OPERATION_HEADERS, apply_workspace_patch, parse_patch
from bioflow_ai.llm.tools.sample import _truncate_output
from bioflow_ai.llm.utils.patch import MAX_PATCH_CHARS, PatchError
from bioflow_ai.schemas.tool_config import ImageConfig

DATA_DIR = '/app/data'
WORK_DIR = '/work'
MOVE_HEADER = '*** Move to: '
RUN_IMAGE = 'docker.io/aye1032/biokit-seq:1.0.0'


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(f'{root}/')


def _resolve_container_path(path: str, cwd: str) -> str:
    """Resolve a container path against ``cwd`` and normalize ``.``/``..`` without touching the host."""
    if '\\' in path or '\x00' in path:
        msg = f'Invalid path: {path!r}'
        raise PatchError(msg)
    return posixpath.normpath(posixpath.join(cwd, path))


def _resolve_cwd(workdir: str | None, default_dir: str) -> str:
    """Resolve an explicit per-call working dir; it must lie inside /work or /app/data."""
    if not workdir:
        return default_dir
    cwd = _resolve_container_path(workdir, default_dir)
    if not (_under(cwd, WORK_DIR) or _under(cwd, DATA_DIR)):
        msg = f'Working dir must be inside {WORK_DIR} or {DATA_DIR}: {workdir}'
        raise PatchError(msg)
    return cwd


def _work_relative(path: str, cwd: str) -> str:
    """Map a patch path to a path relative to /work, rejecting /app/data and anything outside /work."""
    resolved = _resolve_container_path(path, cwd)
    if _under(resolved, DATA_DIR):
        msg = f'{DATA_DIR} is read-only; write project files under {WORK_DIR}: {path}'
        raise PatchError(msg)
    if not _under(resolved, WORK_DIR) or resolved == WORK_DIR:
        msg = f'Path must be a file inside {WORK_DIR}: {path}'
        raise PatchError(msg)
    return posixpath.relpath(resolved, WORK_DIR)


def apply_project_patch(patch: str, workdir: Path, cwd: str) -> str:
    """Apply a patch whose paths are relative to ``cwd`` (or absolute container paths) to the project dir.

    Paths in the operation headers are rewritten to /work-relative form, then the patch goes through the shared
    workspace implementation, which verifies every hunk and rejects symlinks and escapes before writing.
    """
    parse_patch(patch)  # Report structural errors against the paths the model actually wrote.
    lines = patch.replace('\r\n', '\n').split('\n')
    for index, line in enumerate(lines):
        header = next((prefix for prefix in (*OPERATION_HEADERS, MOVE_HEADER) if line.startswith(prefix)), None)
        if header is not None:
            lines[index] = header + _work_relative(line[len(header) :], cwd)
    workdir.mkdir(parents=True, exist_ok=True)
    result = apply_workspace_patch('\n'.join(lines), workdir)
    # Report changed files with their container paths.
    head, _, changes = result.partition('\n')
    return '\n'.join([head, *(f'{change[:2]}{WORK_DIR}/{change[2:]}' for change in changes.splitlines())])


@tool(parse_docstring=True)
async def set_base_dir(
    runtime: ToolRuntime[ProjectContext, ProjectPrepareState],
    base_path: str | None = None,
) -> Command | str:
    """Manage the /app/data mount and switch the default working dir of the file tools.

    With `base_path`: mount that host directory read-only at /app/data and make /app/data the default dir.
    Without arguments: make /work the default dir; any existing /app/data mount is kept unchanged.
    /work (the project dir) stays mounted either way.

    Args:
        base_path: Host directory to mount at /app/data. Omit to switch the default dir to /work.
    """
    if base_path:
        if not await asyncio.to_thread(Path(base_path).is_dir):
            return f'Directory does not exist on the host: {base_path}. Check the path, or ask the user to confirm it.'
        update = {'base_path': base_path, 'default_dir': DATA_DIR}
        content = f'Mounted {base_path} read-only at {DATA_DIR}. Default dir is now {DATA_DIR}; {WORK_DIR} unchanged.'
    else:
        update = {'default_dir': WORK_DIR}
        mounted = runtime.state.get('base_path')
        kept = f'{DATA_DIR} still mounts {mounted}.' if mounted else f'{DATA_DIR} is not mounted.'
        content = f'Default dir is now {WORK_DIR}. {kept}'
    return Command(update={**update, 'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})


@tool('run_command', parse_docstring=True)
async def project_run_command(
    command: str,
    runtime: ToolRuntime[ProjectContext, ProjectPrepareState],
    workdir: str | None = None,
) -> str:
    """Run a read-only shell command in a container.

    /app/data (the data dir set by `set_base_dir`) and /work (the project dir) are both mounted read-only.
    Relative paths resolve against the current default dir unless `workdir` is given.
    Available commands include seqkit, fd, rg, jq, yq, mlr, awk, grep, find, head and file.

    Args:
        command: Shell command to run in the container.
        workdir: Working dir for this call only, inside /work or /app/data. Defaults to the current default dir.
    """
    try:
        cwd = _resolve_cwd(workdir, runtime.state.get('default_dir', WORK_DIR))
    except PatchError as exc:
        return str(exc)

    project_dir = Path(runtime.context.workdir)
    await asyncio.to_thread(project_dir.mkdir, parents=True, exist_ok=True)
    mounts = [ContainerMount(source=str(project_dir), target=WORK_DIR, read_only=True)]
    if base_dir := runtime.state.get('base_path'):
        if not await asyncio.to_thread(Path(base_dir).is_dir):
            return (
                f'The data dir {base_dir} no longer exists. Call `set_base_dir` with the correct path, '
                'or ask the user to confirm it.'
            )
        mounts.append(ContainerMount(source=base_dir, target=DATA_DIR, read_only=True))
    elif _under(cwd, DATA_DIR):
        return f'{DATA_DIR} is not mounted. Call `set_base_dir` with the data path first.'

    code, result = await run_command_async(ImageConfig.from_str(RUN_IMAGE), mounts, command, cwd)
    run_status = 'successfully' if code == 0 else 'failed'
    if clean_out := result.strip():
        return (
            _truncate_output(clean_out)
            if code == 0
            else f'Command ran {run_status}.\n\n......\n{_truncate_output(clean_out)}'
        )
    return f'Command ran {run_status}. No other output.'


@tool('apply_patch', parse_docstring=True)
async def project_apply_patch(
    patch: Annotated[str, Field(max_length=MAX_PATCH_CHARS)],
    runtime: ToolRuntime[ProjectContext, ProjectPrepareState],
    workdir: str | None = None,
) -> str:
    """Create, edit, move or delete UTF-8 files in the project dir /work using a Codex-format patch.

    Patch format and path rules:
    - Start with `*** Begin Patch` and finish with `*** End Patch`, each on its own line.
    - Add: `*** Add File: path`, followed by content lines prefixed with `+`.
    - Delete: `*** Delete File: path`, without a content block.
    - Update: `*** Update File: path`, followed by `@@` context hunks.
    - Inside a hunk, prefix unchanged lines with a space, removals with `-`, and additions with `+`.
    - Move: put `*** Move to: new_path` immediately after the Update header, before its hunks.
    - To anchor the final hunk at the end of a file, finish it with `*** End of File`.
    - Do not use unified-diff line numbers. Combine hunks for the same file.
    - Relative paths resolve against `workdir`, or the current default dir when it is omitted; absolute paths must
      be under /work. /app/data is read-only.
    - All hunks are verified before writing. Filesystem failures may leave partial changes, which are reported.
    - Minimal patch (send its contents without the code fences):
      ```text
      *** Begin Patch
      *** Add File: /work/PLAN.md
      +# Plan
      *** End Patch
      ```

    Args:
        patch: Complete raw patch text, without Markdown fences or shell commands.
        workdir: Working dir for this call only, e.g. /work. Defaults to the current default dir.
    """
    try:
        cwd = _resolve_cwd(workdir, runtime.state.get('default_dir', WORK_DIR))
        return await asyncio.to_thread(apply_project_patch, patch, Path(runtime.context.workdir), cwd)
    except (PatchError, OSError) as exc:
        return f'Patch rejected: {exc}'
