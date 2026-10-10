"""Controlled Codex-format patch tool and its workspace implementation."""

import asyncio
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import Field

from bioflow_ai.llm.context import CodingContext
from bioflow_ai.llm.utils.patch import (
    MAX_PATCH_CHARS,
    PatchChunk,
    PatchError,
    derive_new_contents,
    parse_patch_chunks,
)
from bioflow_ai.schemas.code import CodeNodeType

OPERATION_HEADERS = ('*** Add File: ', '*** Delete File: ', '*** Update File: ')


@dataclass
class FilePatch:
    """A parsed file operation."""

    operation: str
    path: str
    contents: str = ''
    move_path: str | None = None
    chunks: list[PatchChunk] = field(default_factory=list)


@dataclass
class PreparedChange:
    """A file change whose paths and source text have already been verified."""

    operation: str
    source: Path
    target: Path
    contents: str | None
    mode: int = 0o666


def parse_patch(patch: str) -> list[FilePatch]:
    """Parse Add, Delete, Update, Move, and context hunks without executing commands."""
    if len(patch) > MAX_PATCH_CHARS:
        msg = 'Patch exceeds the maximum size.'
        raise PatchError(msg)
    lines = patch.replace('\r\n', '\n').strip('\n').split('\n')
    if len(lines) <= 2 or lines[0] != '*** Begin Patch' or lines[-1] != '*** End Patch':  # noqa: PLR2004
        msg = 'Expected *** Begin Patch and *** End Patch around at least one file operation.'
        raise PatchError(msg)
    operations: list[FilePatch] = []
    index = 1
    while index < len(lines) - 1:
        header = next((prefix for prefix in OPERATION_HEADERS if lines[index].startswith(prefix)), None)
        if header is None:
            msg = f'Invalid file operation on patch line {index + 1}: {lines[index]}'
            raise PatchError(msg)
        path = lines[index][len(header) :]
        if not path.strip():
            msg = 'File paths must not be empty.'
            raise PatchError(msg)
        operation = FilePatch(header.split()[1], path)
        operations.append(operation)
        index += 1
        body: list[str] = []
        while index < len(lines) - 1 and not lines[index].startswith(OPERATION_HEADERS):
            body.append(lines[index])
            index += 1
        if operation.operation == 'Add':
            if not body or any(not line.startswith('+') for line in body):
                msg = f'Add File {path}: every content line must start with +.'
                raise PatchError(msg)
            operation.contents = '\n'.join(line[1:] for line in body) + '\n'
        elif operation.operation == 'Delete':
            if body:
                msg = f'Delete File {path} cannot contain a content block.'
                raise PatchError(msg)
        else:
            if body and body[0].startswith('*** Move to: '):
                operation.move_path = body.pop(0)[len('*** Move to: ') :]
                if not operation.move_path.strip():
                    msg = 'Move destinations must not be empty.'
                    raise PatchError(msg)
            operation.chunks = parse_patch_chunks(body, path)
    return operations


def resolve_patch_path(workspace: Path, filename: str, allowed_files: frozenset[str] | None = None) -> Path:
    """Resolve a virtual /workspace path, rejecting escapes and symlinks at every level."""
    virtual = PurePosixPath(filename)
    if virtual.is_absolute():
        try:
            virtual = virtual.relative_to('/workspace')
        except ValueError as exc:
            msg = 'Absolute paths must be under /workspace.'
            raise PatchError(msg) from exc
    if not virtual.parts or '..' in virtual.parts or '\\' in filename or '\x00' in filename:
        msg = f'Invalid workspace file path: {filename!r}'
        raise PatchError(msg)
    if (
        allowed_files is not None
        and virtual.as_posix() not in allowed_files
        and not (virtual.parent != PurePosixPath('.') and virtual.parts[0] == 'output')
    ):
        msg = f'File is not editable for this CODE language: {filename}'
        raise PatchError(msg)
    root = workspace.resolve(strict=True)
    if not root.is_dir():
        msg = 'The workspace must be a directory.'
        raise PatchError(msg)
    path = root
    target = root.joinpath(*virtual.parts)
    for part in virtual.parts:
        path /= part
        if path.is_symlink():
            msg = f'Symlink paths are not editable: {filename}'
            raise PatchError(msg)
        if path != target and path.exists() and not path.is_dir():
            msg = f'Parent path must be a directory: {filename}'
            raise PatchError(msg)
    if not path.resolve().is_relative_to(root) or path == root:
        msg = f'Path leaves the workspace: {filename}'
        raise PatchError(msg)
    if path.exists() and not path.is_file():
        msg = f'Patch target must be a regular file: {filename}'
        raise PatchError(msg)
    return path


def _read_source(path: Path) -> str:
    try:
        return path.read_bytes().decode('utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        msg = f'Cannot read UTF-8 source {path.name}: {exc}'
        raise PatchError(msg) from exc


def _prepare_changes(
    operations: list[FilePatch], workspace: Path, allowed_files: frozenset[str] | None
) -> list[PreparedChange]:
    prepared: list[PreparedChange] = []
    occupied: set[Path] = set()
    for operation in operations:
        source = resolve_patch_path(workspace, operation.path, allowed_files)
        target = resolve_patch_path(workspace, operation.move_path, allowed_files) if operation.move_path else source
        paths = {source, target}
        if paths & occupied:
            msg = f'Multiple operations target {operation.path}. Combine hunks into one Update File.'
            raise PatchError(msg)
        occupied.update(paths)
        mode = stat.S_IMODE(source.stat().st_mode) if source.exists() else 0o666
        contents: str | None = operation.contents
        if operation.operation != 'Add':
            original = _read_source(source)
            contents = (
                derive_new_contents(original, operation.chunks, operation.path)
                if operation.operation == 'Update'
                else None
            )
        if contents is not None:
            try:
                contents.encode('utf-8')
            except UnicodeEncodeError as exc:
                msg = f'Patch contents must be valid UTF-8: {operation.path}'
                raise PatchError(msg) from exc
        prepared.append(PreparedChange(operation.operation, source, target, contents, mode))
    return prepared


def _write_file(path: Path, contents: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents.encode('utf-8'))
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_workspace_patch(patch: str, workspace: Path, *, allowed_files: frozenset[str] | None = None) -> str:
    """Verify every operation before writing, then return changes including any partial failure."""
    try:
        changes = _prepare_changes(parse_patch(patch), workspace, allowed_files)
    except OSError as exc:
        msg = f'Patch verification failed: {exc}'
        raise PatchError(msg) from exc
    applied: list[str] = []
    root = workspace.resolve()
    try:
        for change in changes:
            # Recheck immediately before each mutation, including newly created parent directories.
            resolve_patch_path(root, change.source.relative_to(root).as_posix(), allowed_files)
            resolve_patch_path(root, change.target.relative_to(root).as_posix(), allowed_files)
            if change.operation == 'Delete':
                change.source.unlink()
                applied.append(f'D {change.source.relative_to(root)}')
            else:
                _write_file(change.target, change.contents or '', change.mode)
                marker = 'A' if change.operation == 'Add' else 'M'
                applied.append(f'{marker} {change.target.relative_to(root)}')
                if change.source != change.target:
                    change.source.unlink()
    except (OSError, PatchError) as exc:
        committed = '\n'.join(applied) or '(none)'
        msg = f'Patch application failed: {exc}\nAlready applied changes:\n{committed}'
        raise PatchError(msg) from exc
    return 'Success. Updated the following files:\n' + '\n'.join(applied)


@tool('apply_patch', parse_docstring=True)
async def coding_apply_patch(
    patch: Annotated[str, Field(max_length=MAX_PATCH_CHARS)], runtime: ToolRuntime[CodingContext]
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
    - Paths must be relative to `/workspace` or absolute beneath it.
    - Only the target script, its allowed `dependencies.txt`, and `output/` test files are editable.
    - All hunks are verified before writing. Filesystem failures may leave partial changes, which are reported.
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
    allowed_files = {
        CodeNodeType.BASH: {'script.sh'},
        CodeNodeType.PYTHON: {'script.py', 'dependencies.txt'},
        CodeNodeType.R: {'script.R', 'dependencies.txt'},
    }[runtime.context.node_type]
    worker = asyncio.create_task(
        asyncio.to_thread(
            apply_workspace_patch,
            patch,
            runtime.context.workspace_path,
            allowed_files=frozenset(allowed_files),
        ),
        name='coding-apply-patch',
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # The owner may delete its workspace after cancellation. Finish any in-flight
        # filesystem mutation before propagating cancellation to that owner.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise
    except (PatchError, OSError) as exc:
        return f'Patch rejected: {exc}'
