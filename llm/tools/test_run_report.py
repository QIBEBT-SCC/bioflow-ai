"""Structured report tool for bioinformatics tool test runs."""

import re
from pathlib import Path, PurePosixPath
from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.config import get_settings
from llm.context import TestRunContext


def resolve_test_run_path(container_path: str, test_run_id: str) -> Path:
    """Translate an absolute /data/output path into the corresponding test-run host path."""
    path = PurePosixPath(container_path)
    if not path.is_absolute() or '..' in path.parts:
        err_msg = f'Path must be absolute and cannot contain parent traversal: {container_path}'
        raise ValueError(err_msg)

    try:
        relative_path = path.relative_to('/data/output')
    except ValueError as error:
        err_msg = f'Path must be located under /data/output: {container_path}'
        raise ValueError(err_msg) from error

    storage_root = Path(get_settings().persistent.storage.STORAGE_PATH).expanduser().resolve()
    test_run_root = (storage_root / 'testrun' / test_run_id).resolve()
    host_path = (test_run_root / Path(*relative_path.parts)).resolve()
    try:
        host_path.relative_to(test_run_root)
    except ValueError as error:
        err_msg = f'Path resolves outside the test-run directory: {container_path}'
        raise ValueError(err_msg) from error
    return host_path


class TestRunFile(BaseModel):
    """A concrete input or output observed during a successful test run."""

    name: str = Field(description='A short, unique logical name, such as r1, reference, result, or log.')
    path: str = Field(description='Absolute path inside the container, under /data/output.')
    file_type: Literal['file', 'folder'] = Field(description='Whether the path is a regular file or a directory.')
    description: str = Field(description='A concise English description of the file or directory.')


class TestRunParameter(BaseModel):
    """A parameter used by the successful test-run command."""

    parameter: str = Field(description='The exact option or positional argument used in the command.')
    description: str = Field(description='A concise English explanation of the parameter.')


class TestRunReport(BaseModel):
    """Machine-readable facts collected from a successful tool test run."""

    working_dir: str = Field(description='Absolute container working directory under /data/output.')
    help_command: str = Field(description='Bare command that prints help for the wrapped command or subcommand.')
    running_command: str = Field(description='The exact successful command, including log redirection.')
    parameters: list[TestRunParameter] = Field(description='All parameters used in running_command.')
    input_files: list[TestRunFile] = Field(
        description='All existing files or directories consumed by running_command. Use stable logical names.'
    )
    output_files: list[TestRunFile] = Field(
        description='All files or directories produced by running_command, including its log.'
    )
    note: str | None = Field(default=None, description='Optional additional execution notes in English.')

    def to_markdown(self) -> str:
        """Render the structured report for downstream LLM context and human display."""
        parameter_lines = '\n'.join(
            f'- `{parameter.parameter}`: {parameter.description}' for parameter in self.parameters
        )
        input_lines = '\n'.join(
            f'- `{file.name}`: `{file.path}` [{file.file_type}] {file.description}' for file in self.input_files
        )
        output_lines = '\n'.join(
            f'- `{file.name}`: `{file.path}` [{file.file_type}] {file.description}' for file in self.output_files
        )
        sections = [
            '## Test Run Report',
            f'working dir: `{self.working_dir}`',
            f'help command:\n```bash\n{self.help_command}\n```',
            f'running command:\n```bash\n{self.running_command}\n```',
            f'**Command Explain**\n{parameter_lines or "- None"}',
            f'**Input Files**\n{input_lines or "- None"}',
            f'**Output Files**\n{output_lines or "- None"}',
        ]
        if self.note:
            sections.append(f'**Note**\n{self.note}')
        return '\n\n'.join(sections)


def _validate_report_file(file: TestRunFile, test_run_id: str, label: str) -> tuple[Path | None, list[str]]:
    errors: list[str] = []
    try:
        host_path = resolve_test_run_path(file.path, test_run_id)
    except ValueError as error:
        return None, [f'{label} `{file.name}`: {error}']

    if not host_path.exists():
        errors.append(f'{label} `{file.name}` does not exist: {file.path} (host: {host_path})')
    elif file.file_type == 'file' and not host_path.is_file():
        errors.append(f'{label} `{file.name}` is declared as a file but is not a file: {file.path}')
    elif file.file_type == 'folder' and not host_path.is_dir():
        errors.append(f'{label} `{file.name}` is declared as a folder but is not a directory: {file.path}')

    return host_path, errors


def validate_test_run_report(report: TestRunReport, test_run_id: str) -> list[str]:
    """Return deterministic validation errors for a structured test-run report."""
    errors: list[str] = []
    try:
        host_working_dir = resolve_test_run_path(report.working_dir, test_run_id)
    except ValueError as error:
        errors.append(f'working_dir: {error}')
        host_working_dir = None
    else:
        if not host_working_dir.is_dir():
            errors.append(
                f'working_dir does not exist or is not a directory: {report.working_dir} (host: {host_working_dir})'
            )

    if not report.help_command.strip():
        errors.append('help_command cannot be empty.')
    elif re.search(r'(^|\s)(docker|podman|sudo)(\s|$)|[|;&<>]', report.help_command):
        errors.append(
            'help_command must be a bare command without container prefixes, redirection, or shell operators.'
        )

    if not report.running_command.strip():
        errors.append('running_command cannot be empty.')
    if not report.output_files:
        errors.append('output_files cannot be empty.')
    elif not any(file.name == 'log' for file in report.output_files):
        errors.append('output_files must contain the execution log with the logical name `log`.')

    all_files = [('Input file', file) for file in report.input_files]
    all_files.extend(('Output file', file) for file in report.output_files)
    names = [file.name for _, file in all_files]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        errors.append(f'File logical names must be unique: {", ".join(duplicate_names)}')

    bookkeeping_files = {'00_running_command.txt', 'task_log.txt'}
    for label, file in all_files:
        host_path, file_errors = _validate_report_file(file, test_run_id, label)
        errors.extend(file_errors)
        if label == 'Output file' and PurePosixPath(file.path).name in bookkeeping_files:
            errors.append(f'Output file `{file.name}` must not include bookkeeping file {file.path}.')
        if label == 'Output file' and host_working_dir is not None and host_path is not None:
            try:
                host_path.relative_to(host_working_dir)
            except ValueError:
                errors.append(f'Output file `{file.name}` must be located under working_dir: {file.path}')

    return errors


@tool('write_tool_report')
async def write_test_run_report(report: TestRunReport, runtime: ToolRuntime[TestRunContext]) -> Command:
    """Submit the final structured report after the tool command has run successfully.

    Every path must be an absolute container path under ``/data/output``. Input files must be the concrete files or
    directories consumed by ``running_command``. Output files must include every result produced by that command,
    including the execution log. Do not include bookkeeping files such as ``00_running_command.txt`` or
    ``task_log.txt``.
    """
    validation_errors = validate_test_run_report(report, runtime.context.test_run_id)
    if validation_errors:
        error_list = '\n- '.join(validation_errors)
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=(
                            'The structured test-run report was not saved because validation failed. '
                            f'Check the following issues and call this tool again:\n- {error_list}'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    return_msg = (
        'Updated the structured test-run report.'
        if runtime.state.get('test_run_report')
        else 'The structured test-run report has been saved.'
    )
    return Command(
        update={
            'test_run_report': report,
            'updated': True,
            'messages': [ToolMessage(content=return_msg, tool_call_id=runtime.tool_call_id)],
        }
    )
