import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import tomli_w
from loguru import logger

from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.containers import ContainerMount, run_command
from bioflow_ai.infrastructure.containers.mount_paths import build_uv_mounts
from bioflow_ai.schemas.react_flow import TaskInput
from bioflow_ai.schemas.tool_config import ImageConfig
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.services.workflow.execution_context import get_node_output, get_prefix

BASH_CODE_IMAGE = ImageConfig.from_str('')
PYTHON_CODE_IMAGE = ImageConfig.from_str('ghcr.io/astral-sh/uv:python3.13-bookworm-slim')
R_CODE_IMAGE = ImageConfig.from_str('r-base:4.4.3')
R_CODE_MAX_DEPENDENCIES = 32
R_CODE_MAX_DEPENDENCY_LENGTH = 100
R_PACKAGE_NAME_PATTERN = re.compile(r'^[A-Za-z](?:[A-Za-z0-9.]*[A-Za-z0-9])?$')
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
    runtime_dependencies = list(user_dependencies)
    if not any(
        (match := PYTHON_PACKAGE_NAME_PATTERN.match(dependency)) is not None
        and match.group().lower().replace('_', '-').replace('.', '-') == 'click'
        for dependency in runtime_dependencies
    ):
        runtime_dependencies.insert(0, PYTHON_CODE_CLI_DEPENDENCY)
    metadata = tomli_w.dumps(
        {
            'requires-python': PYTHON_CODE_REQUIRES_PYTHON,
            'dependencies': runtime_dependencies,
        }
    ).rstrip()
    commented_metadata = '\n'.join(f'# {line}' if line else '#' for line in metadata.splitlines())
    return f'# /// script\n{commented_metadata}\n# ///\n\n{code.rstrip()}\n'


def normalize_r_dependencies(dependencies: Any) -> list[str]:
    """Normalize R dependency data into distinct CRAN package names."""
    if dependencies is None:
        return []
    if isinstance(dependencies, str):
        raw_dependencies = dependencies.splitlines()
    elif isinstance(dependencies, list):
        raw_dependencies = dependencies
    else:
        msg = 'R code dependencies must be a list of package names or a newline-separated string'
        raise TypeError(msg)

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_dependency in raw_dependencies:
        if not isinstance(raw_dependency, str):
            msg = 'Each R code dependency must be a string'
            raise TypeError(msg)
        dependency = raw_dependency.strip()
        if not dependency:
            continue
        if len(dependency) > R_CODE_MAX_DEPENDENCY_LENGTH:
            msg = f'R code dependencies must not exceed {R_CODE_MAX_DEPENDENCY_LENGTH} characters each'
            raise ValueError(msg)
        if R_PACKAGE_NAME_PATTERN.fullmatch(dependency) is None:
            msg = f'R code dependency must be a CRAN package name: {dependency}'
            raise ValueError(msg)
        if dependency not in seen:
            normalized.append(dependency)
            seen.add(dependency)

    if len(normalized) > R_CODE_MAX_DEPENDENCIES:
        msg = f'R code nodes support at most {R_CODE_MAX_DEPENDENCIES} direct dependencies'
        raise ValueError(msg)
    return normalized


def render_r_dependency_script(
    dependencies: Any,
    repository: str,
) -> str:
    """Render a backend-controlled installer for R 4.4.3-compatible CRAN packages."""
    packages = normalize_r_dependencies(dependencies)
    package_vector = ', '.join(json.dumps(package) for package in packages)
    repository_literal = json.dumps(repository)
    return (
        'library_dir <- Sys.getenv("R_LIBS_USER")\n'
        'if (!nzchar(library_dir)) stop("R_LIBS_USER is not configured")\n'
        'dir.create(library_dir, recursive = TRUE, showWarnings = FALSE)\n'
        '.libPaths(c(library_dir, .libPaths()))\n'
        f'options(repos = c(CRAN = {repository_literal}))\n'
        f'packages <- c({package_vector})\n'
        'missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]\n'
        'if (length(missing)) install.packages(missing, lib = library_dir, dependencies = NA)\n'
        'unavailable <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]\n'
        'if (length(unavailable)) stop(sprintf("Failed to install CRAN packages: %s", '
        'paste(unavailable, collapse = ", ")))\n'
    )


def _resolve_container_code_inputs(
    inputs: dict[str, TaskInput | list[TaskInput]],
    language: str,
) -> tuple[list[str], list[ContainerMount], list[str | None]]:
    """Resolve bind_param outputs into container arguments and read-only mounts."""
    raw_inputs = inputs.get('input_files', [])
    kwargs = raw_inputs if isinstance(raw_inputs, list) else [raw_inputs]

    arguments: list[str] = []
    mounts: list[ContainerMount] = []
    sample_names: list[str | None] = []
    seen_parameters: set[str] = set()

    for index, kwarg in enumerate(kwargs, start=1):
        json_str, sample_name = get_node_output(kwarg.source_uid, kwarg.source_k)
        bound_input = json.loads(json_str)
        if not isinstance(bound_input, dict) or len(bound_input) != 1:
            msg = f'Each {language} code input must come from one bind_param node'
            raise ValueError(msg)

        parameter, file_path_value = next(iter(bound_input.items()))
        if not isinstance(parameter, str) or not parameter.startswith('-'):
            msg = f'{language} code input parameter must start with "-": {parameter}'
            raise ValueError(msg)
        if parameter in seen_parameters:
            msg = f'Duplicate {language} code input parameter: {parameter}'
            raise ValueError(msg)
        if not isinstance(file_path_value, str):
            msg = f'{language} code input {parameter} must resolve to a file or directory path'
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


def execute_bash(
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    code: str,
    output_dir: str,
) -> ToolOutput:
    """Run Bash from a temporary workspace and persist files written directly under output/."""
    # 1. 收集所有 bind_param 输入, 解析 {param: file_path}
    raw_inputs = inputs.get('input_files', [])
    if not isinstance(raw_inputs, list):
        raw_inputs = [raw_inputs]

    param_args: dict[str, str] = {}
    input_sample_names = []
    for kwarg in raw_inputs:
        json_str, sample_name = get_node_output(kwarg.source_uid, kwarg.source_k)
        param_args.update(json.loads(json_str))
        input_sample_names.append(sample_name)

    # 2. 准备输出目录
    output_path = Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='bioflow-bash-code-') as staging_dir:
        workspace = Path(staging_dir)
        (workspace / 'output').symlink_to(output_path, target_is_directory=True)
        script_path = workspace / 'script.sh'
        script_path.write_text(code, encoding='utf-8')

        args = [str(script_path)]
        for param, file_path in param_args.items():
            args.extend([param, file_path])

        logger.debug(f'bash executor: bash {" ".join(args)}')

        result = subprocess.run(
            ['/usr/bin/bash', *args],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
    log_path = output_path / '11_run.log'
    log_path.write_text(
        f'[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}',
        encoding='utf-8',
    )

    if result.returncode != 0:
        err_msg = f'bash script failed (exit {result.returncode}):\n{result.stderr}'
        raise RuntimeError(err_msg)

    if input_sample_names:
        common_sample_name = get_prefix(input_sample_names)
        return ToolOutput(result={'output_folder': output_path}, log=log_path, sample_name=common_sample_name)

    return ToolOutput(result={'output_folder': output_path}, log=log_path)


def execute_python(
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    code: str,
    dependencies: list[str] | str | None,
    output_dir: str,
) -> ToolOutput:
    """Run Python from /workspace and persist files written directly under output/."""
    output_path = Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    arguments, input_mounts, sample_names = _resolve_container_code_inputs(inputs, 'Python')
    script_content = render_python_script(code, dependencies)

    with tempfile.TemporaryDirectory(prefix='bioflow-python-code-') as staging_dir:
        workspace = Path(staging_dir)
        (workspace / 'output').mkdir()
        script_path = workspace / 'main.py'
        script_path.write_text(script_content, encoding='utf-8')

        mounts = [
            ContainerMount(source=staging_dir, target='/workspace', read_only=False),
            *build_uv_mounts(),
            *input_mounts,
            ContainerMount(source=str(output_path), target='/workspace/output', read_only=False),
        ]
        index_url = str(get_settings().code.PYTHON_INDEX_URL)
        command = shlex.join(
            [
                'uv',
                'run',
                '--default-index',
                index_url,
                '--script',
                '/workspace/main.py',
                *arguments,
            ]
        )
        logger.debug(f'python executor: {command}')
        status, console_output = run_command(
            PYTHON_CODE_IMAGE,
            mounts,
            command,
            workdir='/workspace',
            ensure_ascii=False,
        )
    log_path = output_path / '11_run.log'
    log_path.write_text(console_output, encoding='utf-8')

    if status != 0:
        err_msg = f'Python script failed (exit {status}):\n{console_output}'
        raise RuntimeError(err_msg)

    return ToolOutput(
        result={'output_folder': output_path},
        log=log_path,
        sample_name=get_prefix(sample_names),
    )


def execute_r(
    *,
    inputs: dict[str, TaskInput | list[TaskInput]],
    code: str,
    dependencies: list[str] | str | None,
    output_dir: str,
) -> ToolOutput:
    """Run R from /workspace and persist files written directly under output/."""
    if not code.strip():
        msg = 'R code must not be empty'
        raise ValueError(msg)
    if '\x00' in code:
        msg = 'R code must not contain NUL bytes'
        raise ValueError(msg)

    output_path = Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    arguments, input_mounts, sample_names = _resolve_container_code_inputs(inputs, 'R')
    normalized_dependencies = normalize_r_dependencies(dependencies)

    with tempfile.TemporaryDirectory(prefix='bioflow-r-code-') as staging_dir:
        workspace = Path(staging_dir)
        (workspace / 'output').mkdir()
        script_path = workspace / 'script.R'
        script_path.write_text(f'{code.rstrip()}\n', encoding='utf-8')
        if normalized_dependencies:
            dependency_script = workspace / 'install_dependencies.R'
            repository = str(get_settings().code.CRAN_REPOSITORY)
            dependency_script.write_text(
                render_r_dependency_script(normalized_dependencies, repository),
                encoding='utf-8',
            )

        mounts = [
            ContainerMount(source=staging_dir, target='/workspace', read_only=False),
            *input_mounts,
            ContainerMount(source=str(output_path), target='/workspace/output', read_only=False),
        ]
        run_script_command = shlex.join(['Rscript', '--vanilla', '/workspace/script.R', *arguments])
        if normalized_dependencies:
            library_dir = '/data/r-library'
            install_command = shlex.join(['Rscript', '--vanilla', '/workspace/install_dependencies.R'])
            command = (
                f'mkdir -p {shlex.quote(library_dir)} && '
                f'export R_LIBS_USER={shlex.quote(library_dir)} && '
                f'{install_command} && {run_script_command}'
            )
        else:
            command = run_script_command
        logger.debug(f'r executor: {command}')
        status, console_output = run_command(
            R_CODE_IMAGE,
            mounts,
            command,
            workdir='/workspace',
            ensure_ascii=False,
        )

    log_path = output_path / '11_run.log'
    log_path.write_text(console_output, encoding='utf-8')
    if status != 0:
        err_msg = f'R script failed (exit {status}):\n{console_output}'
        raise RuntimeError(err_msg)

    return ToolOutput(
        result={'output_folder': output_path},
        log=log_path,
        sample_name=get_prefix(sample_names),
    )
