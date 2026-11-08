"""Source formatting for reusable CODE definitions."""

import shlex
from pathlib import Path
from tempfile import TemporaryDirectory

from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.containers import run_command_async
from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.services.code.workspace import CODE_IMAGES, build_workspace_mounts

FORMAT_FILENAME = 'script.py'
# Ruff's default style, ignoring any configuration; without a cache nothing is left in the workspace. Output is
# only read on failure, and ``--quiet`` would hide parse errors.
RUFF_COMMANDS = (
    ('check', '--isolated', '--no-cache', '--select', 'I', '--fix-only', FORMAT_FILENAME),
    ('format', '--isolated', '--no-cache', FORMAT_FILENAME),
)


class CodeFormatError(ValueError):
    """Raised when source cannot be formatted, such as on a syntax error."""


async def format_source(node_type: CodeNodeType, source: str) -> str:
    """Sort imports and format Python source with Ruff in the Python CODE image.

    Ruff runs through ``uv tool run`` against the shared uv cache, as the coding workspace installs it. Import
    sorting leaves unparsable source unchanged, so syntax errors surface from the format pass.

    Raises:
        CodeFormatError: The node type has no formatter, or Ruff rejected the source.
    """
    if node_type is not CodeNodeType.PYTHON:
        msg = f'Formatting is not supported for {node_type.value} code.'
        raise CodeFormatError(msg)
    index_url = str(get_settings().code.PYTHON_INDEX_URL)
    command = ' && '.join(
        shlex.join(['uv', 'tool', 'run', '--quiet', '--default-index', index_url, 'ruff', *arguments])
        for arguments in RUFF_COMMANDS
    )
    with TemporaryDirectory(prefix='bioflow-format-') as directory:
        workspace = Path(directory)
        script = workspace / FORMAT_FILENAME
        script.write_text(source, encoding='utf-8')
        script.chmod(0o666)
        status, output = await run_command_async(
            CODE_IMAGES[node_type],
            build_workspace_mounts(workspace, node_type),
            command,
            workdir='/workspace',
            ensure_ascii=False,
        )
        if status != 0:
            msg = output.strip().removeprefix('error: ') or f'Ruff exited with status {status}.'
            raise CodeFormatError(msg)
        return script.read_text(encoding='utf-8')
