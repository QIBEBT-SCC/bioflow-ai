# ruff: noqa: EM101
"""Container-backed, read-only evidence inspection for workflow diagnosis V2."""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime  # noqa: TC002
from sqlalchemy.orm import Session

import app.crud.project_file_mapping as project_file_mapping_crud
import app.crud.resource.sample as sample_crud
import app.crud.run_instance as run_instance_crud
from app.core.session import engine
from app.schemas.tool_config import ImageConfig
from app.utils.container import ContainerMount, run_command
from llm.context import WorkflowDiagnosisContext  # noqa: TC001
from llm.utils.formatters import format_json

EVIDENCE_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-seq:1.0.0')
EVIDENCE_MOUNT_PATH = '/app/data/input'
EVIDENCE_WORKDIR = '/app/data'
MAX_COMMAND_CHARS = 8_000
COMMAND_TIMEOUT_SECONDS = 60
COMMAND_KILL_GRACE_SECONDS = 5
MAX_CAPTURE_BYTES = 64 * 1024
MAX_RETURN_CHARS = 40_000
RETURN_HEAD_CHARS = 30_000
_CAPTURE_FILE_BLOCKS = math.ceil(MAX_CAPTURE_BYTES / 512)
_WRAPPER_METADATA = re.compile(
    r'^__BIOFLOW_DIAG_STATUS__=(?P<status>\d+)\n'
    r'^__BIOFLOW_DIAG_CAPTURE_BYTES__=(?P<bytes>\d+)\n'
    r'^__BIOFLOW_DIAG_CAPTURE_TRUNCATED__=(?P<truncated>[01])\n?',
    re.MULTILINE,
)


class EvidenceSourceError(ValueError):
    """A recoverable error while resolving an evidence source."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _evidence_error(code: str, message: str) -> EvidenceSourceError:
    return EvidenceSourceError(code, message)


@dataclass(frozen=True)
class ResolvedEvidence:
    """A validated host resource that may be mounted into the evidence container."""

    source: str
    path: Path
    kind: Literal['file', 'directory']


def _error(code: str, message: str) -> str:
    return format_json({'ok': False, 'error': {'code': code, 'message': message}}, max_chars=None)


def _validated_path(raw_path: str | None, source: str) -> tuple[Path, Literal['file', 'directory']]:
    if not raw_path:
        raise _evidence_error('missing_path', f'Evidence source `{source}` has no filesystem path.')
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise _evidence_error('path_not_found', f'Evidence source `{source}` no longer exists.') from error
    if path.is_file():
        return path, 'file'
    if path.is_dir():
        return path, 'directory'
    raise _evidence_error('unsupported_path', f'Evidence source `{source}` is not a regular file or directory.')


def _current_run(session: Session, context: WorkflowDiagnosisContext) -> Any:
    run = run_instance_crud.get(session, context.source_run_uid)
    if run is None:
        raise _evidence_error('run_not_found', 'The workflow execution no longer exists.')
    if run.owner_id != context.user_id:
        raise _evidence_error('access_denied', 'The workflow execution does not belong to the current user.')
    if run.project_id != context.project_id:
        raise _evidence_error('project_mismatch', 'The workflow execution does not belong to the current project.')
    return run


def resolve_evidence_source(source: str, context: WorkflowDiagnosisContext) -> ResolvedEvidence:
    """Resolve one logical evidence reference without accepting arbitrary host paths."""
    with Session(engine) as session:
        run = _current_run(session, context)

        if source == 'run:current':
            path, kind = _validated_path(run.base_dir, source)
            if kind != 'directory':
                raise _evidence_error('invalid_run_path', 'The current run path is not a directory.')
            return ResolvedEvidence(source=source, path=path, kind=kind)

        if source.startswith('sample:'):
            tag = source.removeprefix('sample:')
            if not tag:
                raise _evidence_error('invalid_source', 'A sample source must include a tag.')
            if run.sample_uid is None:
                raise _evidence_error(
                    'sample_unavailable',
                    'The current workflow execution is project-level and has no bound sample.',
                )
            sample = sample_crud.get_with_files(session, run.sample_uid)
            if sample is None:
                raise _evidence_error('sample_not_found', 'The sample bound to the current run no longer exists.')
            if sample.owner_id != context.user_id or sample.project_id != context.project_id:
                raise _evidence_error('access_denied', 'The current run sample is outside the allowed project scope.')
            sample_file = next((item for item in sample.sample_files if item.tag == tag), None)
            if sample_file is None:
                raise _evidence_error('reference_not_found', f'Sample reference `{source}` was not found.')
            path, kind = _validated_path(sample_file.file_path, source)
            return ResolvedEvidence(source=source, path=path, kind=kind)

        if source.startswith('proj:'):
            keyword = source.removeprefix('proj:')
            if not keyword:
                raise _evidence_error('invalid_source', 'A project source must include a keyword.')
            mapping = project_file_mapping_crud.get_by_keyword(session, context.project_id, keyword)
            if mapping is None:
                raise _evidence_error('reference_not_found', f'Project reference `{source}` was not found.')
            if mapping.project_id != context.project_id:
                raise _evidence_error('project_mismatch', 'The project reference is outside the allowed project scope.')
            path, kind = _validated_path(mapping.file_path, source)
            return ResolvedEvidence(source=source, path=path, kind=kind)

    raise _evidence_error(
        'invalid_source',
        'source must be exactly `run:current`, `sample:<tag>`, or `proj:<keyword>`.',
    )


def build_bounded_command(command: str) -> str:
    """Wrap an inspection command with time and captured-output limits inside the container."""
    stripped = command.strip()
    if not stripped:
        raise _evidence_error('empty_command', 'The evidence inspection command cannot be empty.')
    if len(stripped) > MAX_COMMAND_CHARS:
        raise _evidence_error(
            'command_too_long',
            f'The evidence inspection command exceeds {MAX_COMMAND_CHARS} characters.',
        )

    quoted_command = shlex.quote(stripped)
    return (
        'output_file=$(mktemp)\n'
        '(\n'
        f'  ulimit -f {_CAPTURE_FILE_BLOCKS}\n'
        f'  timeout -k {COMMAND_KILL_GRACE_SECONDS}s {COMMAND_TIMEOUT_SECONDS}s /bin/sh -c {quoted_command}\n'
        ') >"$output_file" 2>&1\n'
        'command_status=$?\n'
        'capture_bytes=$(wc -c <"$output_file" | tr -d " ")\n'
        f'if [ "$capture_bytes" -ge {MAX_CAPTURE_BYTES} ]; then capture_truncated=1; else capture_truncated=0; fi\n'
        'printf "__BIOFLOW_DIAG_STATUS__=%s\\n" "$command_status"\n'
        'printf "__BIOFLOW_DIAG_CAPTURE_BYTES__=%s\\n" "$capture_bytes"\n'
        'printf "__BIOFLOW_DIAG_CAPTURE_TRUNCATED__=%s\\n" "$capture_truncated"\n'
        f'head -c {RETURN_HEAD_CHARS} "$output_file"\n'
        f'if [ "$capture_bytes" -gt {RETURN_HEAD_CHARS} ]; then\n'
        '  printf "\\n... [container output tail] ...\\n"\n'
        f'  tail -c {MAX_RETURN_CHARS - RETURN_HEAD_CHARS} "$output_file"\n'
        'fi\n'
        'rm -f "$output_file"\n'
        'exit "$command_status"'
    )


def _parse_command_output(output: str, exit_code: int) -> tuple[str, int, int | None, bool]:
    match = _WRAPPER_METADATA.match(output)
    capture_bytes: int | None = None
    capture_truncated = False
    if match is not None:
        exit_code = int(match.group('status'))
        capture_bytes = int(match.group('bytes'))
        capture_truncated = match.group('truncated') == '1'
        output = output[match.end() :]

    output = output.strip()
    python_truncated = len(output) > MAX_RETURN_CHARS
    if python_truncated:
        tail_chars = MAX_RETURN_CHARS - RETURN_HEAD_CHARS
        omitted = len(output) - MAX_RETURN_CHARS
        output = (
            f'{output[:RETURN_HEAD_CHARS]}\n'
            f'... [truncated {omitted} characters] ...\n'
            f'{output[-tail_chars:]}'
        )
    return output, exit_code, capture_bytes, capture_truncated or python_truncated


@tool('run_command', parse_docstring=True)
def run_evidence_command(
    source: str,
    command: str,
    runtime: ToolRuntime[WorkflowDiagnosisContext],
) -> str:
    """Run a bounded shell inspection against one read-only workflow evidence source.

    The source is mounted at ``/app/data/input``. It may be a file or a directory, so
    inspect it with ``file /app/data/input`` when the type is not already known.

    Args:
        source: Exactly ``run:current``, ``sample:<tag>``, or ``proj:<keyword>``.
        command: Shell inspection command using ``/app/data/input`` as the evidence path.
    """
    try:
        evidence = resolve_evidence_source(source, runtime.context)
        bounded_command = build_bounded_command(command)
    except EvidenceSourceError as error:
        return _error(error.code, str(error))

    runtime.stream_writer({'info': f'Inspecting read-only evidence: {source}'})
    mount = ContainerMount(source=str(evidence.path), target=EVIDENCE_MOUNT_PATH, read_only=True)
    try:
        exit_code, raw_output = run_command(
            EVIDENCE_IMAGE,
            [mount],
            bounded_command,
            workdir=EVIDENCE_WORKDIR,
            ensure_ascii=False,
        )
    except Exception as error:  # Container failures are recoverable evidence gaps for the agent.
        return _error('container_error', f'{type(error).__name__}: {error}')

    output, exit_code, capture_bytes, truncated = _parse_command_output(raw_output, exit_code)
    return format_json(
        {
            'ok': exit_code == 0,
            'source': evidence.source,
            'mount_path': EVIDENCE_MOUNT_PATH,
            'source_type': evidence.kind,
            'exit_code': exit_code,
            'timed_out': exit_code in {124, 137},
            'truncated': truncated,
            'captured_bytes': capture_bytes,
            'output': output,
        },
        max_chars=None,
    )
