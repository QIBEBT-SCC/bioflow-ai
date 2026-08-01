"""Execution validation tool for generated Docker tool configurations."""

import uuid
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger

from app.schemas.tool_config import FileMount, ImageConfig
from app.services.tool_execution import (
    DockerToolExecution,
    DockerToolExecutionConfig,
    DockerToolExecutionResult,
    ResolvedToolInput,
    execute_docker_tool_with_result,
    prepare_docker_tool_execution,
)
from app.utils.container import ContainerMount
from llm.context import ToolGeneratorContext
from llm.tools.test_run_report import TestRunReport, resolve_test_run_path

REQUIRED_CONFIG_FIELDS = (
    'image',
    'test_run_report',
    'name',
    'description',
    'file_mounts',
    'command_template',
    'dynamic_params',
    'immutable_static_params',
    'modifiable_static_params',
)
DIAGNOSTIC_TAIL_CHARS = 4000
MAX_LISTED_FILES = 50


def _validation_failed(message: str, tool_call_id: str | None) -> Command:
    """Return a failed validation result without ending the agent loop."""
    return Command(
        update={
            'validated': False,
            'updated': False,
            'should_end': False,
            'messages': [
                ToolMessage(content=f'Tool configuration validation failed: {message}', tool_call_id=tool_call_id)
            ],
        }
    )


def _build_validation_inputs(
    report: TestRunReport,
    file_mounts: list[FileMount],
    test_run_id: str,
) -> dict[str, ResolvedToolInput]:
    """Bind generated input handles to concrete files recorded by the test-run report."""
    report_inputs = {file.name: file for file in report.input_files}
    generated_inputs = [file for file in file_mounts if file.file_type == 'INPUT']
    generated_names = {file.name for file in generated_inputs}
    missing_names = sorted(generated_names - report_inputs.keys())
    if missing_names:
        err_msg = (
            'Generated input handle(s) have no matching test-run input: '
            f'{", ".join(missing_names)}. INPUT FileMount.name must reuse the logical name from the report.'
        )
        raise ValueError(err_msg)

    unused_names = sorted(report_inputs.keys() - generated_names)
    if unused_names:
        err_msg = f'Test-run input(s) were not represented by an INPUT FileMount: {", ".join(unused_names)}.'
        raise ValueError(err_msg)

    return {
        name: ResolvedToolInput(value=str(resolve_test_run_path(report_file.path, test_run_id)))
        for name, report_file in report_inputs.items()
    }


def _tail_text(path: Path) -> str:
    """Read a bounded tail from a diagnostic text file."""
    if not path.is_file():
        return '(log file was not created)'
    try:
        content = path.read_text(errors='replace')
    except OSError as error:
        return f'(failed to read log: {error})'
    return content[-DIAGNOSTIC_TAIL_CHARS:] or '(log file is empty)'


def _list_generated_files(output_dir: Path) -> str:
    """List a bounded set of paths that actually appeared in the validation directory."""
    if not output_dir.exists():
        return '(validation output directory was not created)'
    paths = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob('*'))
    if not paths:
        return '(no files were generated)'
    displayed = paths[:MAX_LISTED_FILES]
    if len(paths) > MAX_LISTED_FILES:
        displayed.append(f'... and {len(paths) - MAX_LISTED_FILES} more')
    return '\n'.join(displayed)


def _format_execution_failure(
    execution_result: DockerToolExecutionResult,
    command: str,
    volume_binds: list[ContainerMount],
    output_dir: Path,
) -> str:
    """Build actionable feedback for the configuration-generating agent."""
    console_tail = execution_result.console_output[-DIAGNOSTIC_TAIL_CHARS:] or '(container output is empty)'
    log_path = execution_result.output.log
    log_tail = _tail_text(Path(log_path)) if log_path is not None else '(no log output was declared)'
    missing_outputs = (
        '\n'.join(str(path) for path in execution_result.missing_outputs) or '(no declared outputs are missing)'
    )
    mounts = '\n'.join(f'{bind.source} -> {bind.target}' for bind in volume_binds) or '(no mounts)'
    generated_files = _list_generated_files(output_dir)
    return (
        f'Command exited with code {execution_result.exit_code}.\n'
        f'Command:\n{command}\n\n'
        f'Mounts:\n{mounts}\n\n'
        f'Container output tail:\n{console_tail}\n\n'
        f'Declared log tail:\n{log_tail}\n\n'
        f'Missing declared outputs:\n{missing_outputs}\n\n'
        f'Files actually generated:\n{generated_files}'
    )


def _format_execution_exception(error: Exception, execution: DockerToolExecution) -> str:
    """Expose the prepared command and mounts when container validation raises."""
    mounts = '\n'.join(f'{bind.source} -> {bind.target}' for bind in execution.volume_binds) or '(no mounts)'
    return (
        f'Container validation could not start or complete: {type(error).__name__}: {error}\n\n'
        f'Command:\n{execution.command}\n\n'
        f'Mounts:\n{mounts}'
    )


@tool('validate_tool_config')
def validate_tool_config(runtime: ToolRuntime[ToolGeneratorContext]) -> Command:
    """Run the generated tool configuration with the real test-run inputs.

    Call this only after file mounts, command details, name, and description have all been configured. If validation
    fails, inspect the returned command, mount, or missing-output error, update the relevant configuration, and call
    this tool again. A successful validation ends the configuration task automatically.
    """
    missing_fields = [field for field in REQUIRED_CONFIG_FIELDS if field not in runtime.state]
    if missing_fields:
        return _validation_failed(
            f'Configuration is incomplete; missing state field(s): {", ".join(missing_fields)}.',
            runtime.tool_call_id,
        )

    execution = None
    try:
        report = TestRunReport.model_validate(runtime.state['test_run_report'])
        file_mounts = runtime.state['file_mounts']
        inputs = _build_validation_inputs(report, file_mounts, runtime.context.test_run_id)
        config = DockerToolExecutionConfig(
            image=ImageConfig.model_validate(runtime.state['image']),
            command_template=runtime.state['command_template'],
            dynamic_params=runtime.state['dynamic_params'],
            immutable_static_params=runtime.state['immutable_static_params'],
            modifiable_static_params=runtime.state['modifiable_static_params'],
            file_mounts=file_mounts,
        )
        test_run_root = resolve_test_run_path('/data/output', runtime.context.test_run_id)
        output_dir = test_run_root / 'validation' / uuid.uuid4().hex
        execution = prepare_docker_tool_execution(
            config=config,
            inputs=inputs,
            output_dir=str(output_dir),
            sample_name='validation',
            params=config.modifiable_static_params or '',
        )
        execution_result = execute_docker_tool_with_result(execution)
        if not execution_result.succeeded:
            return _validation_failed(
                _format_execution_failure(
                    execution_result,
                    execution.command,
                    execution.volume_binds,
                    output_dir,
                ),
                runtime.tool_call_id,
            )
    except Exception as error:
        logger.exception('Generated tool configuration validation failed')
        detail = _format_execution_exception(error, execution) if execution is not None else str(error)
        return _validation_failed(detail, runtime.tool_call_id)

    output_paths = ', '.join(str(Path(path)) for path in execution_result.output.all_files().values())
    return Command(
        update={
            'validated': True,
            'updated': True,
            'should_end': True,
            'messages': [
                ToolMessage(
                    content=(
                        'Tool configuration validation passed. '
                        f'Command: {execution.command}\nValidated outputs: {output_paths}'
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
