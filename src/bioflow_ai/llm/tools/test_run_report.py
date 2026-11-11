"""Structured report tool for bioinformatics tool test runs."""

import re
from pathlib import Path, PurePosixPath
from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.containers import ContainerMount
from bioflow_ai.infrastructure.containers.mount_paths import (
    PROJECT_DATA_ROOTS,
    build_project_data_mount_plan,
    resolve_registered_database_input_path,
)
from bioflow_ai.llm.context import TestRunContext


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


def resolve_test_run_input_path(
    container_path: str,
    test_run_id: str,
    project_id: int | None = None,
    database_mounts: list[ContainerMount] | None = None,
) -> Path:
    """Resolve an input from output storage, project data, or a registered database mount."""
    path = PurePosixPath(container_path)
    if not path.is_absolute() or '..' in path.parts:
        err_msg = f'Path must be absolute and cannot contain parent traversal: {container_path}'
        raise ValueError(err_msg)

    try:
        path.relative_to('/data/output')
    except ValueError:
        pass
    else:
        return resolve_test_run_path(container_path, test_run_id)

    if project_id is not None and any(_is_same_or_child(path, PurePosixPath(root)) for root in PROJECT_DATA_ROOTS):
        return build_project_data_mount_plan(project_id).resolve_input_path(container_path)

    if database_mounts:
        return resolve_registered_database_input_path(container_path, database_mounts)

    allowed_roots = ', '.join(('/data/output', *PROJECT_DATA_ROOTS))
    err_msg = (
        f'Input path must be located under one of the managed roots ({allowed_roots}) or an external mount '
        f'of a registered biological database used by a successful run_command: {container_path}'
    )
    raise ValueError(err_msg)


def _is_same_or_child(path: PurePosixPath, root: PurePosixPath) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class TestRunFile(BaseModel):
    """A concrete input or output observed during a successful test run."""

    name: str = Field(description='A short, unique logical name, such as r1, reference, result, or log.')
    path: str = Field(description='Absolute path inside the container, under a managed test-run input root.')
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
    running_command: str = Field(
        description='Exactly one successful invocation of the target tool, with optional result/log redirection. Exclude preparation, postprocessing, and other tool calls.'
    )
    parameters: list[TestRunParameter] = Field(description='All parameters used in running_command.')
    input_files: list[TestRunFile] = Field(
        description='All existing files or directories consumed by running_command. Use stable logical names.'
    )
    output_files: list[TestRunFile] = Field(
        description='All files or directories produced by running_command, including its log.'
    )
    database_mounts: list[ContainerMount] = Field(
        default_factory=list,
        description='Database input mount mappings recorded automatically from successful run_command calls. Omit this field when submitting a report.',
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
        if self.database_mounts:
            mount_lines = '\n'.join(f'- `{mount.target}`: host `{mount.source}`' for mount in self.database_mounts)
            sections.append(f'**Registered Database Mounts**\n{mount_lines}')
        return '\n\n'.join(sections)


def _validate_report_file(
    file: TestRunFile,
    test_run_id: str,
    label: str,
    project_id: int | None,
    database_mounts: list[ContainerMount],
) -> tuple[Path | None, list[str]]:
    errors: list[str] = []
    try:
        host_path = (
            resolve_test_run_input_path(file.path, test_run_id, project_id, database_mounts)
            if label == 'Input file'
            else resolve_test_run_path(file.path, test_run_id)
        )
    except ValueError as error:
        return None, [f'{label} `{file.name}`: {error}']

    try:
        if not host_path.exists():
            errors.append(f'{label} `{file.name}` does not exist: {file.path} (host: {host_path})')
        elif file.file_type == 'file' and not host_path.is_file():
            errors.append(f'{label} `{file.name}` is declared as a file but is not a file: {file.path}')
        elif file.file_type == 'folder' and not host_path.is_dir():
            errors.append(f'{label} `{file.name}` is declared as a folder but is not a directory: {file.path}')
    except OSError as error:
        errors.append(f'{label} `{file.name}` cannot be accessed: {file.path} (host: {host_path}): {error}')

    return host_path, errors


def validate_test_run_report(report: TestRunReport, test_run_id: str, project_id: int | None = None) -> list[str]:
    """Return deterministic validation errors for a structured test-run report."""
    errors: list[str] = []
    try:
        host_working_dir = resolve_test_run_path(report.working_dir, test_run_id)
    except ValueError as error:
        errors.append(f'working_dir: {error}')
        host_working_dir = None
    else:
        try:
            is_working_dir = host_working_dir.is_dir()
        except OSError as error:
            errors.append(f'working_dir cannot be accessed: {report.working_dir} (host: {host_working_dir}): {error}')
        else:
            if not is_working_dir:
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
        host_path, file_errors = _validate_report_file(file, test_run_id, label, project_id, report.database_mounts)
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
    """Submit the final structured report after the tool command has run successfully."""
    recorded_mounts = [ContainerMount.model_validate(mount) for mount in runtime.state.get('test_run_mounts', [])]
    database_mounts = [
        mount
        for mount in recorded_mounts
        if any(_is_same_or_child(PurePosixPath(file.path), PurePosixPath(mount.target)) for file in report.input_files)
    ]
    report = report.model_copy(update={'database_mounts': database_mounts})
    validation_errors = validate_test_run_report(
        report,
        runtime.context.test_run_id,
        runtime.context.project_id,
    )
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
