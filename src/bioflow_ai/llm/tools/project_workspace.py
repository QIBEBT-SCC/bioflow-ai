"""Workspace tools for the project-planner agent: read-only /app/data plus the read-write /work project dir."""

import asyncio
from pathlib import Path, PurePosixPath

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.states import ProjectPlannerState
from bioflow_ai.llm.tools.apply_patch import apply_workspace_patch
from bioflow_ai.llm.tools.container import PROJECT_DATA_DIR, PROJECT_WORK_DIR, resolve_project_cwd


async def apply_project_patch(patch: str, workdir: Path, cwd: str) -> str:
    """Apply the original patch in the host directory corresponding to ``cwd``."""
    directory = workdir / PurePosixPath(cwd).relative_to(PROJECT_WORK_DIR)
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    result = await apply_workspace_patch(patch, directory)
    # Report changed files with their container paths.
    head, _, changes = result.partition('\n')
    return '\n'.join([head, *(f'{change[:2]}{cwd}/{change[2:]}' for change in changes.splitlines())])


@tool(parse_docstring=True)
async def set_base_dir(
    runtime: ToolRuntime[ProjectContext, ProjectPlannerState],
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
        update = {'base_path': base_path, 'default_dir': PROJECT_DATA_DIR}
        content = f'Mounted {base_path} read-only at {PROJECT_DATA_DIR}. Default dir is now {PROJECT_DATA_DIR}; {PROJECT_WORK_DIR} unchanged.'
    else:
        update = {'default_dir': PROJECT_WORK_DIR}
        mounted = runtime.state.get('base_path')
        kept = f'{PROJECT_DATA_DIR} still mounts {mounted}.' if mounted else f'{PROJECT_DATA_DIR} is not mounted.'
        content = f'Default dir is now {PROJECT_WORK_DIR}. {kept}'
    return Command(update={**update, 'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})


@tool('apply_patch', parse_docstring=True)
async def project_apply_patch(
    patch: str,
    runtime: ToolRuntime[ProjectContext, ProjectPlannerState],
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
    - All file paths must be relative to `workdir`, or the current default dir when it is omitted. Do not use absolute
      paths in the patch. The working directory must be inside /work; /app/data is read-only.
    - Files are applied in patch order. An error can leave earlier file changes in place; inspect files before retrying.
    - Minimal patch (send its contents without the code fences):
      ```text
      *** Begin Patch
      *** Add File: PLAN.md
      +# Plan
      *** End Patch
      ```

    Args:
        patch: Complete raw patch text, without Markdown fences or shell commands.
        workdir: Working dir for this call only, e.g. /work. Defaults to the current default dir.
    """
    try:
        cwd = resolve_project_cwd(workdir, runtime.state.get('default_dir', PROJECT_WORK_DIR))
        return await apply_project_patch(patch, Path(runtime.context.workdir), cwd)
    except (OSError, RuntimeError, ValueError) as exc:
        return f'Patch failed: {exc}\nPlease check that the paths are correct and relative to the working directory.'
