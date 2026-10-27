"""Codex-format patch tools backed by apply-patch-py."""

from pathlib import Path

from apply_patch_py import apply_patch
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from bioflow_ai.llm.context import CodingContext


async def apply_workspace_patch(patch: str, workspace: Path) -> str:
    """Apply the original patch through the library and report changed files."""
    affected = await apply_patch(patch, workdir=workspace)
    changes = [
        f'{operation} {path}'
        for operation, paths in (('A', affected.added), ('M', affected.modified), ('D', affected.deleted))
        for path in paths
    ]
    return 'Success. Updated the following files:\n' + '\n'.join(changes)


@tool('apply_patch', parse_docstring=True)
async def coding_apply_patch(
    patch: str,
    runtime: ToolRuntime[CodingContext],
) -> str:
    """Edit UTF-8 workspace files using a Codex-format patch.

    Patch format and workspace rules:
    - Start with `*** Begin Patch` and finish with `*** End Patch`, each on its own line.
    - Add: `*** Add File: path`, followed by content lines prefixed with `+`.
    - Delete: `*** Delete File: path`, without a content block.
    - Update: `*** Update File: path`, followed by `@@` context hunks.
    - Inside a hunk, prefix unchanged lines with a space, removals with `-`, and additions with `+`.
    - Move: put `*** Move to: new_path` immediately after the Update header, before its hunks.
    - To anchor the final hunk at the end of a file, finish it with `*** End of File`.
    - Do not use unified-diff line numbers. Combine hunks for the same file.
    - File paths must be relative to `/workspace`, such as `script.py` or `output/example.txt`; do not use absolute paths.
    - Files are applied in patch order. An error can leave earlier file changes in place; inspect files before retrying.
    - Minimal patch (send its contents without the code fences):
      ```text
      *** Begin Patch
      *** Add File: output/example.txt
      +example
      *** End Patch
      ```

    Args:
        patch: Complete raw patch text, without Markdown fences or shell commands.
    """
    try:
        return await apply_workspace_patch(patch, runtime.context.workspace_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return f'Patch failed: {exc}\nPlease check that the paths are correct and relative to the working directory.'
