import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import tomli_w
from celery import Task
from loguru import logger

from app.models.workflow import ToolOutput
from app.schemas.react_flow import TaskInput
from app.schemas.tool_config import ImageConfig
from app.tasks.base.base_task import BaseBioTask, get_prefix, get_tool_output
from app.utils.container import ContainerMount, run_command
from script.worker import celery_app

PYTHON_CODE_IMAGE = ImageConfig.from_str('ghcr.io/astral-sh/uv:python3.13-bookworm-slim')
PYTHON_CODE_REQUIRES_PYTHON = '==3.13.*'
PYTHON_CODE_CLI_DEPENDENCY = 'click>=8.1,<9'
PYTHON_CODE_MAX_DEPENDENCIES = 32
PYTHON_CODE_MAX_DEPENDENCY_LENGTH = 200
PYTHON_PACKAGE_NAME_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*')


def normalize_python_dependencies(dependencies: Any) -> list[str]:
    """Normalize node dependency data into distinct PEP 508 requirement strings."""
    if dependencies is None:
        return []

    if isinstance(dependencies, str):
        raw_dependencies = dependencies.splitlines()
    elif isinstance(dependencies, list):
        raw_dependencies = dependencies
    else:
        msg = 'Python code dependencies must be a list of strings or a newline-separated string'
        raise TypeError(msg)

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_dependency in raw_dependencies:
        if not isinstance(raw_dependency, str):
            msg = 'Each Python code dependency must be a string'
            raise TypeError(msg)

        dependency = raw_dependency.strip()
        if not dependency:
            continue
        if len(dependency) > PYTHON_CODE_MAX_DEPENDENCY_LENGTH:
            msg = f'Python code dependencies must not exceed {PYTHON_CODE_MAX_DEPENDENCY_LENGTH} characters each'
            raise ValueError(msg)
        if any(character in dependency for character in ('\x00', '\n', '\r')):
            msg = 'Python code dependencies must be single-line requirement strings'
            raise ValueError(msg)
        if dependency.startswith('-'):
            msg = f'Python code dependency is not a package requirement: {dependency}'
            raise ValueError(msg)
        if dependency not in seen:
            normalized.append(dependency)
            seen.add(dependency)

    if len(normalized) > PYTHON_CODE_MAX_DEPENDENCIES:
        msg = f'Python code nodes support at most {PYTHON_CODE_MAX_DEPENDENCIES} direct dependencies'
        raise ValueError(msg)

    return normalized


def render_python_script(code: str, dependencies: Any) -> str:
    """Add backend-controlled PEP 723 metadata to user-authored Python code."""
    if not code.strip():
        msg = 'Python code must not be empty'
        raise ValueError(msg)

    user_dependencies = normalize_python_dependencies(dependencies)
    runtime_dependencies = [
        PYTHON_CODE_CLI_DEPENDENCY,
        *(
            dependency
            for dependency in user_dependencies
            if (match := PYTHON_PACKAGE_NAME_PATTERN.match(dependency)) is None
            or match.group().lower().replace('_', '-').replace('.', '-') != 'click'
        ),
    ]
    metadata = tomli_w.dumps(
        {
            'requires-python': PYTHON_CODE_REQUIRES_PYTHON,
            'dependencies': runtime_dependencies,
        }
    ).rstrip()
    commented_metadata = '\n'.join(f'# {line}' if line else '#' for line in metadata.splitlines())
    return f'# /// script\n{commented_metadata}\n# ///\n\n{code.rstrip()}\n'


def _resolve_python_inputs(
    inputs: dict[str, TaskInput | list[TaskInput]],
) -> tuple[list[str], list[ContainerMount], list[str | None]]:
    """Resolve bind_param outputs into container arguments and read-only mounts."""
    raw_inputs = inputs.get('input_files', [])
    kwargs = raw_inputs if isinstance(raw_inputs, list) else [raw_inputs]

    arguments: list[str] = []
    mounts: list[ContainerMount] = []
    sample_names: list[str | None] = []
    seen_parameters: set[str] = set()

    for index, kwarg in enumerate(kwargs, start=1):
        json_str, sample_name = get_tool_output(kwarg.source_uid, kwarg.source_k)
        bound_input = json.loads(json_str)
        if not isinstance(bound_input, dict) or len(bound_input) != 1:
            msg = 'Each Python code input must come from one bind_param node'
            raise ValueError(msg)

        parameter, file_path_value = next(iter(bound_input.items()))
        if not isinstance(parameter, str) or not parameter.startswith('-'):
            msg = f'Python code input parameter must start with "-": {parameter}'
            raise ValueError(msg)
        if parameter in seen_parameters:
            msg = f'Duplicate Python code input parameter: {parameter}'
            raise ValueError(msg)
        if not isinstance(file_path_value, str):
            msg = f'Python code input {parameter} must resolve to a file or directory path'
            raise TypeError(msg)

        host_path = Path(file_path_value).absolute()
        if not host_path.exists():
            raise FileNotFoundError(host_path)

        suffix = ''.join(host_path.suffixes) if host_path.is_file() else ''
        container_path = f'/input_{index:03d}{suffix}'
        mounts.append(ContainerMount(source=str(host_path), target=container_path, read_only=True))
        arguments.extend([parameter, container_path])
        sample_names.append(sample_name)
        seen_parameters.add(parameter)

    return arguments, mounts, sample_names


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def bash_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    code: str,
    output_dir: str,
) -> ToolOutput:
    """直接用 subprocess 执行 bash 脚本.

    输入来自 bind_param 节点, 每个输入是 JSON 字符串 {param: file_path}.
    输出为脚本工作目录 (folder_path).
    """
    # 1. 收集所有 bind_param 输入, 解析 {param: file_path}
    raw_inputs = inputs.get('input_files', [])
    if not isinstance(raw_inputs, list):
        raw_inputs = [raw_inputs]

    param_args: dict[str, str] = {}
    input_sample_names = []
    for kwarg in raw_inputs:
        json_str, sample_name = get_tool_output(kwarg.source_uid, kwarg.source_k)
        param_args.update(json.loads(json_str))
        input_sample_names.append(sample_name)

    # 2. 准备输出目录并写入脚本
    output_path = Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    script_path = output_path / '_script.sh'
    script_path.write_text(code)

    # 3. 构建命令: bash _script.sh -param /path/to/file ...
    args = [str(script_path)]
    for param, file_path in param_args.items():
        args.extend([param, file_path])

    logger.debug(f'bash_task: bash {" ".join(args)}')

    result = subprocess.run(
        ['/usr/bin/bash', *args],
        cwd=output_path,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        err_msg = f'bash script failed (exit {result.returncode}):\n{result.stderr}'
        raise RuntimeError(err_msg)

    if input_sample_names:
        common_sample_name = get_prefix(input_sample_names)
        return ToolOutput(result={'output_folder': output_path}, sample_name=common_sample_name)

    return ToolOutput(result={'output_folder': output_path})


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def python_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    code: str,
    dependencies: list[str] | str | None,
    output_dir: str,
) -> ToolOutput:
    """Run a single-file Python code node in an isolated uv container."""
    output_path = Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    arguments, input_mounts, sample_names = _resolve_python_inputs(inputs)
    script_content = render_python_script(code, dependencies)

    with tempfile.TemporaryDirectory(prefix='bioflow-python-code-') as staging_dir:
        script_path = Path(staging_dir) / 'main.py'
        script_path.write_text(script_content, encoding='utf-8')

        mounts = [
            ContainerMount(source=staging_dir, target='/app', read_only=True),
            *input_mounts,
            ContainerMount(source=str(output_path), target='/data/output', read_only=False),
        ]
        command = shlex.join(['uv', 'run', '--script', '/app/main.py', *arguments])
        logger.debug(f'python_task: {command}')
        status, console_output = run_command(
            PYTHON_CODE_IMAGE,
            mounts,
            command,
            workdir='/data/output',
            ensure_ascii=False,
        )

    if status != 0:
        err_msg = f'Python script failed (exit {status}):\n{console_output}'
        raise RuntimeError(err_msg)

    return ToolOutput(
        result={'output_folder': output_path},
        sample_name=get_prefix(sample_names),
    )
