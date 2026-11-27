import json
import shlex
from pathlib import Path

from loguru import logger

from bioflow_ai.infrastructure.containers import ContainerMount, run_command
from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.schemas.react_flow import TaskInput
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.services.code.workspace import (
    BASH_CODE_IMAGE,
    PYTHON_CODE_IMAGE,
    R_CODE_IMAGE,
    initialize_code_workspace,
    reset_code_workspace,
)
from bioflow_ai.services.workflow.execution_context import get_node_output, get_prefix

PYTHON_CODE_MAX_DEPENDENCIES = 32
PYTHON_CODE_MAX_DEPENDENCY_LENGTH = 200


def normalize_python_dependencies(dependencies: list[str] | str | None) -> list[str]:
    """Normalize node dependency data into distinct PEP 508 requirement strings."""
    if dependencies is None:
        return []

    if isinstance(dependencies, str):
        normalized = dependencies.splitlines()
    elif isinstance(dependencies, list):
        normalized = dependencies
    else:
        msg = 'Python code dependencies must be a list of strings or a newline-separated string'
        raise TypeError(msg)

    return normalized


def _parse_bound_code_input(
    kwarg: TaskInput,
    language: str,
    seen_parameters: set[str],
    *,
    expect_path: bool,
) -> tuple[str, str, str | None]:
    """Decode one bound CODE input and enforce unique option names across handles."""
    json_str, sample_name = get_node_output(kwarg.source_uid, kwarg.source_k)
    bound_input = json.loads(json_str)
    if not isinstance(bound_input, dict) or len(bound_input) != 1:
        msg = f'Each {language} code input must come from one bind_param node'
        raise ValueError(msg)

    parameter, raw_value = next(iter(bound_input.items()))
    if not isinstance(parameter, str) or not parameter.startswith('-'):
        msg = f'{language} code input parameter must start with "-": {parameter}'
        raise ValueError(msg)
    if parameter in seen_parameters:
        msg = f'Duplicate {language} code input parameter: {parameter}'
        raise ValueError(msg)
    if not isinstance(raw_value, str):
        kind = 'file or directory path' if expect_path else 'text value'
        msg = f'{language} code input {parameter} must resolve to a {kind}'
        raise TypeError(msg)

    seen_parameters.add(parameter)
    return parameter, raw_value, sample_name


def _resolve_container_code_inputs(
    inputs: dict[str, TaskInput | list[TaskInput]],
    language: str,
) -> tuple[list[str], list[ContainerMount], list[str | None]]:
    """Resolve bind_param outputs into container arguments and read-only mounts."""
    raw_file_inputs = inputs.get('input_files', [])
    file_kwargs = raw_file_inputs if isinstance(raw_file_inputs, list) else [raw_file_inputs]
    raw_value_inputs = inputs.get('input_values', [])
    value_kwargs = raw_value_inputs if isinstance(raw_value_inputs, list) else [raw_value_inputs]

    arguments: list[str] = []
    mounts: list[ContainerMount] = []
    sample_names: list[str | None] = []
    seen_parameters: set[str] = set()

    for index, kwarg in enumerate(file_kwargs, start=1):
        parameter, file_path_value, sample_name = _parse_bound_code_input(
            kwarg,
            language,
            seen_parameters,
            expect_path=True,
        )

        host_path = Path(file_path_value).absolute()
        if not host_path.exists():
            raise FileNotFoundError(host_path)

        suffix = ''.join(host_path.suffixes) if host_path.is_file() else ''
        container_path = f'/input_{index:03d}{suffix}'
        mounts.append(ContainerMount(source=str(host_path), target=container_path, read_only=True))
        arguments.extend([parameter, container_path])
        sample_names.append(sample_name)

    for kwarg in value_kwargs:
        parameter, text_value, sample_name = _parse_bound_code_input(
            kwarg,
            language,
            seen_parameters,
            expect_path=False,
        )
        arguments.extend([parameter, text_value])
        sample_names.append(sample_name)

    return arguments, mounts, sample_names


def execute_bash(
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    code: str,
    output_dir: str,
) -> ToolOutput:
    """Initialize and run Bash directly in the persistent Step workspace."""
    workspace = Path(output_dir).absolute()
    arguments, input_mounts, sample_names = _resolve_container_code_inputs(inputs, 'Bash')
    reset_code_workspace(workspace)
    mounts = initialize_code_workspace(workspace, CodeNodeType.BASH, source=code, extra_mounts=input_mounts)
    command = shlex.join(['bash', '/workspace/script.sh', *arguments])
    logger.debug(f'bash executor: {command}')
    status, console_output = run_command(
        BASH_CODE_IMAGE,
        mounts,
        command,
        workdir='/workspace',
        ensure_ascii=False,
    )
    log_path = workspace / '11_run.log'
    log_path.write_text(console_output, encoding='utf-8')

    if status != 0:
        err_msg = f'Bash script failed (exit {status}):\n{console_output}'
        raise RuntimeError(err_msg)

    return ToolOutput(
        result={'output_folder': workspace / 'output'},
        log=log_path,
        sample_name=get_prefix(sample_names),
    )


def execute_python(
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    code: str,
    dependencies: list[str] | str | None,
    output_dir: str,
) -> ToolOutput:
    """Initialize and run Python directly in the persistent Step workspace."""
    workspace = Path(output_dir).absolute()

    arguments, input_mounts, sample_names = _resolve_container_code_inputs(inputs, 'Python')
    normalized_dependencies = normalize_python_dependencies(dependencies)

    reset_code_workspace(workspace)
    try:
        mounts = initialize_code_workspace(
            workspace,
            CodeNodeType.PYTHON,
            normalized_dependencies,
            source=code,
            extra_mounts=input_mounts,
        )
    except RuntimeError as error:
        (workspace / '11_run.log').write_text(str(error), encoding='utf-8')
        raise
    command = shlex.join(['uv', 'run', '--no-sync', '/workspace/main.py', *arguments])
    logger.debug(f'python executor: {command}')
    status, console_output = run_command(
        PYTHON_CODE_IMAGE,
        mounts,
        command,
        workdir='/workspace',
        ensure_ascii=False,
    )
    log_path = workspace / '11_run.log'
    log_path.write_text(console_output, encoding='utf-8')

    if status != 0:
        err_msg = f'Python script failed (exit {status}):\n{console_output}'
        raise RuntimeError(err_msg)

    return ToolOutput(
        result={'output_folder': workspace / 'output'},
        log=log_path,
        sample_name=get_prefix(sample_names),
    )


def execute_r(
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    code: str,
    output_dir: str,
) -> ToolOutput:
    """Initialize and run R directly in the persistent Step workspace."""
    workspace = Path(output_dir).absolute()
    arguments, input_mounts, sample_names = _resolve_container_code_inputs(inputs, 'R')

    reset_code_workspace(workspace)
    mounts = initialize_code_workspace(
        workspace,
        CodeNodeType.R,
        source=f'{code.rstrip()}\n',
        extra_mounts=input_mounts,
    )
    command = shlex.join(['Rscript', '--vanilla', '/workspace/script.R', *arguments])
    logger.debug(f'r executor: {command}')
    status, console_output = run_command(
        R_CODE_IMAGE,
        mounts,
        command,
        workdir='/workspace',
        ensure_ascii=False,
    )

    log_path = workspace / '11_run.log'
    log_path.write_text(console_output, encoding='utf-8')
    if status != 0:
        err_msg = f'R script failed (exit {status}):\n{console_output}'
        raise RuntimeError(err_msg)

    return ToolOutput(
        result={'output_folder': workspace / 'output'},
        log=log_path,
        sample_name=get_prefix(sample_names),
    )
