"""Containerized tool execution independent of Celery and persistence."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from typing import Any

from loguru import logger

from app.models.tools import FileMount, ImageConfig, ParamDefine
from app.models.workflow import ToolOutput
from app.utils.container import ContainerMount, run_command, run_command_background
from app.utils.input_format import PartialFormatter, build_command_template

MOUNTLIST_PREFIX = 'mountlist:'


@dataclass(frozen=True)
class ResolvedToolInput:
    """A tool input already resolved to a host path or literal value."""

    value: str
    sample_name: str | None = None


@dataclass(frozen=True)
class DockerToolExecutionConfig:
    """The executable subset of a generated Docker tool definition."""

    image: ImageConfig
    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str | None
    modifiable_static_params: str | None
    file_mounts: list[FileMount]


@dataclass(frozen=True)
class DockerToolExecution:
    """A fully prepared Docker tool invocation."""

    image: ImageConfig
    command: str
    volume_binds: list[ContainerMount]
    expected_output: ToolOutput


@dataclass(frozen=True)
class DockerToolExecutionResult:
    """Observable result of a synchronous prepared-tool execution."""

    output: ToolOutput
    exit_code: int
    console_output: str
    missing_outputs: list[Path]

    @property
    def succeeded(self) -> bool:
        """Return whether the command exited cleanly and produced every declared output."""
        return self.exit_code == 0 and not self.missing_outputs


def _resolve_mount_path(
    mount_paths: dict[str, Path],
    param_inputs: dict[str, str],
    conflicted_paths: set[str],
    container_mount_path: str,
    host_mount_path: Path,
) -> str:
    """Assign a unique container path when host directory mounts conflict."""
    abs_host = host_mount_path.absolute()
    existing = mount_paths.get(container_mount_path)

    if existing is None and container_mount_path not in conflicted_paths:
        return container_mount_path

    if existing == abs_host:
        return container_mount_path

    if existing is not None and container_mount_path not in conflicted_paths:
        conflicted_paths.add(container_mount_path)
        old_host = mount_paths.pop(container_mount_path)
        old_sub = f'{container_mount_path}/{old_host.name}'
        mount_paths[old_sub] = old_host
        for key, value in param_inputs.items():
            if value == container_mount_path or value.startswith(container_mount_path + '/'):
                relative = value[len(container_mount_path) :]
                param_inputs[key] = old_sub + relative

    candidate = f'{container_mount_path}/{host_mount_path.name}'
    counter = 1
    while candidate in mount_paths and mount_paths[candidate] != abs_host:
        candidate = f'{container_mount_path}/{host_mount_path.name}_{counter}'
        counter += 1

    logger.warning(f'挂载冲突: {container_mount_path} 已分裂, 将 {abs_host} 映射到 {candidate}')
    return candidate


def _load_mountlist_manifest(manifest_ref: str) -> dict[str, Any]:
    """Load a mount-list manifest produced by the collect_mount_dirs processor."""
    manifest_path = Path(manifest_ref.removeprefix(MOUNTLIST_PREFIX)).absolute()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text())
    if manifest.get('type') != 'mount_dir_list':
        err_msg = f'Invalid mount-list manifest type: {manifest.get("type")}'
        raise ValueError(err_msg)
    if not isinstance(manifest.get('entries'), list):
        err_msg = 'Invalid mount-list manifest: entries must be a list'
        raise TypeError(err_msg)
    if not manifest.get('root_dir'):
        err_msg = 'Invalid mount-list manifest: missing root_dir'
        raise ValueError(err_msg)

    return manifest


def _add_mountlist_mounts(
    manifest_ref: str,
    file: FileMount,
    mount_paths: dict[str, Path],
    param_inputs: dict[str, str],
    conflicted_paths: set[str],
) -> str:
    """Expand a mount-list manifest into a logical container input directory."""
    manifest = _load_mountlist_manifest(manifest_ref)
    host_root = Path(manifest['root_dir']).absolute()
    host_root.mkdir(parents=True, exist_ok=True)

    container_root = _resolve_mount_path(mount_paths, param_inputs, conflicted_paths, file.mount_path, host_root)
    if container_root not in mount_paths:
        mount_paths[container_root] = host_root

    seen_mount_names: set[str] = set()
    for index, entry in enumerate(manifest['entries'], start=1):
        source_path = Path(entry['source_path']).absolute()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not source_path.is_dir():
            raise NotADirectoryError(source_path)

        mount_name = str(entry.get('mount_name') or f'dir_{index:03d}')
        if mount_name in seen_mount_names:
            mount_name = f'{index:03d}_{mount_name}'
        seen_mount_names.add(mount_name)

        container_path = f'{container_root}/{mount_name}'
        if container_path in mount_paths and mount_paths[container_path] != source_path:
            err_msg = f'Mount-list target conflict: {container_path}'
            raise ValueError(err_msg)

        mount_paths[container_path] = source_path

    return container_root


def build_tool_mounts(
    file_mounts: list[FileMount],
    host_output_dir: str,
    sample_name: str,
    inputs: dict[str, ResolvedToolInput],
) -> tuple[dict[str, str], dict[str, Path], ToolOutput]:
    """Build container mounts and expected outputs from resolved tool inputs."""
    param_inputs: dict[str, str] = {}
    mount_paths: dict[str, Path] = {}
    expected_output = ToolOutput()
    conflicted_paths: set[str] = set()
    source_sample_names: list[str | None] = []

    input_files = [file for file in file_mounts if file.file_type == 'INPUT']
    output_files = [file for file in file_mounts if file.file_type == 'OUTPUT']

    for file in input_files:
        resolved_input = inputs[file.name]
        if file.file_path.startswith('value:'):
            param_inputs[file.name] = resolved_input.value
            continue

        file_path = resolved_input.value
        if file_path.startswith(MOUNTLIST_PREFIX):
            source_sample_names.append(resolved_input.sample_name)
            param_inputs[file.name] = _add_mountlist_mounts(
                file_path,
                file,
                mount_paths,
                param_inputs,
                conflicted_paths,
            )
            continue

        host_file_path = Path(file_path)
        source_sample_names.append(resolved_input.sample_name)

        if host_file_path.is_dir():
            host_mount_path = host_file_path
            actual = _resolve_mount_path(
                mount_paths,
                param_inputs,
                conflicted_paths,
                file.mount_path,
                host_mount_path,
            )
            container_file_path = actual
        else:
            host_mount_path = host_file_path.parent
            actual = _resolve_mount_path(
                mount_paths,
                param_inputs,
                conflicted_paths,
                file.mount_path,
                host_mount_path,
            )
            container_file_path = f'{actual}/{host_file_path.name}'

        if actual not in mount_paths:
            mount_paths[actual] = host_mount_path.absolute()

        param_inputs[file.name] = container_file_path

    common_sample_name = _get_common_sample_name(source_sample_names)
    output_sample_name = common_sample_name or sample_name
    expected_output.sample_name = output_sample_name

    for file in output_files:
        try:
            host_file_path = Path(file.file_path.format(output_dir=host_output_dir, sample_name=output_sample_name))
        except KeyError as error:
            err_msg = f'Unknown variable {error} in output path template: {file.file_path}'
            raise ValueError(err_msg) from error

        host_mount_path = Path(host_output_dir)
        actual = _resolve_mount_path(mount_paths, param_inputs, conflicted_paths, file.mount_path, host_mount_path)
        container_file_path = f'{actual}/{host_file_path.name}'
        if actual not in mount_paths:
            mount_paths[actual] = host_mount_path.absolute()

        if file.is_log:
            expected_output.log = host_file_path.absolute()
        elif file.is_report:
            expected_output.reports[file.name] = host_file_path.absolute()
        else:
            expected_output.result[file.name] = host_file_path.absolute()

        param_inputs[file.name] = container_file_path

    return param_inputs, mount_paths, expected_output


def _get_common_sample_name(sample_names: list[str | None]) -> str | None:
    """Return the common sample name while preserving the existing task behavior."""
    valid_names = [name for name in sample_names if name is not None]
    if not valid_names:
        return None

    common_characters = takewhile(
        lambda characters: all(char == characters[0] for char in characters),
        zip(*valid_names, strict=False),
    )
    return ''.join(characters[0] for characters in common_characters).rstrip('.-_rR')


def prepare_docker_tool_execution(
    config: DockerToolExecutionConfig,
    inputs: dict[str, ResolvedToolInput],
    output_dir: str,
    sample_name: str,
    params: str = '',
) -> DockerToolExecution:
    """Prepare a Docker invocation without accessing Celery or persistent state."""
    host_output_dir = Path(output_dir).absolute()
    host_output_dir.mkdir(parents=True, exist_ok=True)

    param_inputs, mount_paths, expected_output = build_tool_mounts(
        config.file_mounts,
        output_dir,
        sample_name,
        inputs,
    )
    final_template = build_command_template(
        command_template=config.command_template,
        dynamic_params=config.dynamic_params,
        immutable_static_params=config.immutable_static_params,
        modifiable_static_params=params,
        file_mounts=config.file_mounts,
    )
    command = PartialFormatter().format(
        final_template,
        output_dir='/data/output',
        sample_name=expected_output.sample_name,
        **param_inputs,
    )
    volume_binds = [
        ContainerMount(source=str(host_path), target=container_path, read_only=False)
        for container_path, host_path in mount_paths.items()
    ]
    volume_binds.append(ContainerMount(source='/tmp', target='/tmp', read_only=False))  # noqa: S108

    return DockerToolExecution(
        image=config.image,
        command=command,
        volume_binds=volume_binds,
        expected_output=expected_output,
    )


def execute_docker_tool(
    execution: DockerToolExecution,
    monitor: Callable[[dict[str, Any] | None], None] | None = None,
) -> ToolOutput:
    """Execute a prepared Docker tool and verify all declared outputs."""
    if execution.expected_output.all_exist():
        return execution.expected_output

    for stats in run_command_background(
        execution.image,
        execution.command,
        execution.volume_binds,
        working_dir='/data/output',
    ):
        if monitor is not None:
            monitor(stats)

    missing_outputs = [str(path) for path in execution.expected_output.all_files().values() if not Path(path).exists()]
    if missing_outputs:
        err_msg = f'部分输出文件未生成: {missing_outputs}'
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    return execution.expected_output


def execute_docker_tool_with_result(execution: DockerToolExecution) -> DockerToolExecutionResult:
    """Execute a prepared tool synchronously and preserve diagnostics for validation."""
    exit_code, console_output = run_command(
        execution.image,
        execution.volume_binds,
        execution.command,
        workdir='/data/output',
        ensure_ascii=False,
    )
    missing_outputs = [Path(path) for path in execution.expected_output.all_files().values() if not Path(path).exists()]
    return DockerToolExecutionResult(
        output=execution.expected_output,
        exit_code=exit_code,
        console_output=console_output,
        missing_outputs=missing_outputs,
    )
