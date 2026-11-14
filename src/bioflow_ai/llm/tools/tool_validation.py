"""Read-only inspection and node execution during tool generation."""

import json
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.orm import Session

import bioflow_ai.crud.tools.tool as tool_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.context import ToolGeneratorContext
from bioflow_ai.llm.tools.test_run_report import TestRunReport, resolve_test_run_input_path, resolve_test_run_path
from bioflow_ai.llm.tools.tool_catalog import InvalidToolDefinitionError, SearchToolSchema, search_catalog_nodes
from bioflow_ai.schemas.tool_config import ToolType
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.services.tools.execution import (
    DockerToolExecution,
    DockerToolExecutionConfig,
    DockerToolExecutionResult,
    ResolvedToolInput,
    execute_docker_tool_with_result,
    prepare_docker_tool_execution,
)
from bioflow_ai.services.tools.passthrough import copy_validation_paths, isolate_validation_inputs
from bioflow_ai.services.workflow.output_validation import output_path_exists
from bioflow_ai.services.workflow.validation.passthrough import validate_managed_paths
from bioflow_ai.utils.formatters import search_nodes_formatter
from bioflow_ai.utils.uid_mapper import BidirectionalMapper, merge_mappers, restore_mapper

MAX_LISTED_FILES = 50
DIAGNOSTIC_TAIL_CHARS = 4000
REQUIRED_DRAFT_FIELDS = (*DockerToolExecutionConfig.model_fields, 'name', 'description')


class NodeOutputBinding(BaseModel):
    """Reference to an output from an earlier execution in this validation run."""

    model_config = ConfigDict(extra='forbid')

    run_id: str
    output: str
    relative_path: str = '.'


def _message(
    runtime: ToolRuntime[ToolGeneratorContext],
    content: str,
    *,
    error: bool = False,
    artifact: dict[str, Any] | None = None,
) -> ToolMessage:
    return ToolMessage(
        content=content, tool_call_id=runtime.tool_call_id, status='error' if error else 'success', artifact=artifact
    )


@tool(parse_docstring=True)
def search_node(patterns: list[str], runtime: ToolRuntime[ToolGeneratorContext]) -> Command:
    """Search existing Tool nodes by their complete command.

    Args:
        patterns: Case-insensitive patterns. Every pattern must match, e.g. ["configManta", "--bam"].
    """
    try:
        query = SearchToolSchema(patterns=patterns)
        nodes, matched_uids = search_catalog_nodes(query.patterns)
    except (ValidationError, InvalidToolDefinitionError) as error:
        return Command(update={'messages': [_message(runtime, str(error), error=True)]})

    mapper = merge_mappers(runtime.state.get('id_mapper'), BidirectionalMapper(datas=nodes))
    count = len(matched_uids)
    content = search_nodes_formatter(nodes, mapper)
    if count > len(nodes):
        content += f'\nShowing {len(nodes)} of {count} matches. Add patterns to narrow the search.'
    return Command(
        update={'id_mapper': mapper, 'messages': [_message(runtime, content, artifact={'result_count': count})]}
    )


def _load_node(node_id: int, runtime: ToolRuntime[ToolGeneratorContext]) -> dict[str, Any]:
    uid = restore_mapper(runtime.state.get('id_mapper')).get_uid_by_index(node_id)
    if not uid:
        message = f'Unknown node ID {node_id}. Use search_node first.'
        raise ValueError(message)
    with Session(engine) as session:
        node = tool_crud.get_with_image(session, uid)
        if node is None or node.image is None:
            message = f'Node {node_id} no longer exists or has no image.'
            raise ValueError(message)
        config = DockerToolExecutionConfig(
            image=node.image.image,
            tool_type=node.tool_type,
            command_template=node.command_template,
            dynamic_params=node.dynamic_params,
            immutable_static_params=node.immutable_static_params,
            modifiable_static_params=node.modifiable_static_params,
            file_mounts=node.file_mounts,
        )
        return {
            'node_id': node_id,
            'uid': str(node.uid),
            'name': node.name,
            'description': node.description,
            **config.model_dump(mode='json'),
        }


@tool(parse_docstring=True)
def view_node(node_id: int, runtime: ToolRuntime[ToolGeneratorContext]) -> ToolMessage:
    """Read the raw node definition JSON, including its image, mounts and command parameters.

    Args:
        node_id: The ID returned by search_node.
    """
    try:
        return _message(runtime, json.dumps(_load_node(node_id, runtime), ensure_ascii=False))
    except ValueError as error:
        return _message(runtime, str(error), error=True)


def _relative_path(root: Path, value: str) -> Path:
    base = root.resolve()
    relative = PurePosixPath(value)
    if relative.is_absolute() or '..' in relative.parts:
        message = f'Use a relative path without parent traversal: {value}'
        raise ValueError(message)
    resolved = (base / relative).resolve()
    if not resolved.is_relative_to(base):
        message = f'Path escapes its directory: {value}'
        raise ValueError(message)
    return resolved


@tool('inspect_files', parse_docstring=True)
def inspect_validation_files(
    command: Literal['ls', 'stat'],
    runtime: ToolRuntime[ToolGeneratorContext],
    root: Literal['trial', 'validation'] = 'trial',
    path: str = '.',
) -> ToolMessage:
    """Inspect files with read-only ls or stat operations. Paths are relative to the chosen root.

    Args:
        command: List directory entries with ls or inspect one file with stat.
        root: trial for original test-run files or validation for files from execute_node.
        path: Relative path within that root.
    """
    base = (
        resolve_test_run_path('/data/output', runtime.context.test_run_id)
        if root == 'trial'
        else runtime.context.validation_root.resolve()
    )
    try:
        target = _relative_path(base, path)
        if command == 'stat':
            info = target.stat()
            result: dict[str, Any] = {'path': path, 'is_dir': target.is_dir(), 'size': info.st_size}
        else:
            entries = list(islice(target.iterdir(), MAX_LISTED_FILES + 1))
            result = {
                'entries': sorted(entry.name for entry in entries[:MAX_LISTED_FILES]),
                'truncated': len(entries) > MAX_LISTED_FILES,
            }
        return _message(runtime, json.dumps(result, ensure_ascii=False))
    except (ValueError, OSError) as error:
        return _message(runtime, str(error), error=True)


def _resolve_bindings(
    config: DockerToolExecutionConfig,
    input_bindings: dict[str, str | NodeOutputBinding],
    runtime: ToolRuntime[ToolGeneratorContext],
    input_copy_root: Path,
) -> dict[str, ResolvedToolInput]:
    mounts = {file.name: file for file in config.file_mounts if file.file_type == 'INPUT'}
    missing = sorted(mounts.keys() - input_bindings.keys())
    extra = sorted(input_bindings.keys() - mounts.keys())
    if missing or extra:
        message = f'Bind exactly the INPUT handles. Missing: {missing}. Unknown: {extra}.'
        raise ValueError(message)

    report = TestRunReport.model_validate(runtime.state['test_run_report'])
    root = runtime.context.validation_root.resolve()
    trial_root = resolve_test_run_path('/data/output', runtime.context.test_run_id)
    runs = runtime.state.get('validation_runs', {})
    resolved: dict[str, ResolvedToolInput] = {}
    to_copy: dict[str, Path] = {}
    for name, binding in input_bindings.items():
        file = mounts[name]
        if file.file_path.startswith('value:'):
            if not isinstance(binding, str):
                message = f'Input {name!r} requires a literal string.'
                raise ValueError(message)
            resolved[name] = ResolvedToolInput(value=binding)
            continue
        if isinstance(binding, NodeOutputBinding):
            if binding.run_id not in runs:
                message = f'Unknown execution {binding.run_id!r}. Use an output from this validation run.'
                raise ValueError(message)
            output = ToolOutput.model_validate(runs[binding.run_id])
            value = output.all_files().get(binding.output)
            if value is None:
                message = f'Execution {binding.run_id!r} has no available output {binding.output!r}.'
                raise ValueError(message)
            path = Path(value).resolve()
            if binding.relative_path != '.':
                if not path.is_dir():
                    message = f'Output {binding.output!r} is not a directory.'
                    raise ValueError(message)
                path = _relative_path(path, binding.relative_path)
            if not path.is_relative_to(root):
                message = f'Output {binding.output!r} escapes the validation directory.'
                raise ValueError(message)
            resolved[name] = ResolvedToolInput(
                value=str(path),
                workspace=output.workspace if config.tool_type == ToolType.DOCKER_PASSTHROUGH else None,
                source_uid=binding.run_id,
                sample_name=output.sample_name,
            )
        else:
            path = (
                resolve_test_run_input_path(
                    binding, runtime.context.test_run_id, runtime.context.project_id, report.database_mounts
                )
                if PurePosixPath(binding).is_absolute()
                else _relative_path(trial_root, binding)
            )
            if file.is_passthrough:
                # Only test-run data may be staged for a writable handle.
                if not path.is_relative_to(trial_root):
                    message = f'Writable input {name!r} must come from the trial directory or an earlier execution.'
                    raise ValueError(message)
                if path == trial_root:
                    message = f'Bind {name!r} to a specific trial file or directory, rather than the entire trial root.'
                    raise ValueError(message)
                validate_managed_paths([path], trial_root)
                to_copy[name] = path
            else:
                resolved[name] = ResolvedToolInput(value=str(path))
        if not output_path_exists(path):
            message = f'Input {name!r} does not exist: {path}'
            raise FileNotFoundError(message)

    resolved.update(
        {
            name: ResolvedToolInput(value=str(path))
            for name, path in copy_validation_paths(to_copy, input_copy_root).items()
        }
    )
    return resolved


def _protect_trial_data(execution: DockerToolExecution, root: Path) -> DockerToolExecution:
    """Keep every bind outside this execution attempt read-only, including upstream runs."""
    return execution.model_copy(
        update={
            'volume_binds': [
                mount
                if Path(mount.source).resolve().is_relative_to(root)
                else mount.model_copy(update={'read_only': True})
                for mount in execution.volume_binds
            ],
        }
    )


def _check_mount_targets(config: DockerToolExecutionConfig) -> None:
    for file in config.file_mounts:
        if file.file_type == 'INPUT' and file.file_path.startswith('value:'):
            continue
        target = PurePosixPath(file.mount_path)
        if not target.is_relative_to('/data') or target == PurePosixPath('/data') or '..' in target.parts:
            message = f'Mount {file.name!r} must target a path below /data: {file.mount_path}'
            raise ValueError(message)


def _expected_draft_outputs(
    config: DockerToolExecutionConfig,
    execution: DockerToolExecution,
    root: Path,
    output_members: dict[str, list[str]],
) -> dict[str, list[Path]]:
    """Resolve explicit draft output paths and expand any declared filename-prefix members."""
    outputs = {file.name: file for file in config.file_mounts if file.file_type == 'OUTPUT'}
    unknown = sorted(output_members.keys() - outputs.keys())
    if unknown:
        message = f'Unknown OUTPUT handles in output_members: {unknown}'
        raise ValueError(message)
    values = execution.expected_output.all_files()
    paths: dict[str, list[Path]] = {}
    for name, file in outputs.items():
        base = Path(values['log' if file.is_log else name])
        suffixes = output_members.get(name)
        if suffixes is not None and (
            not suffixes
            or any(not suffix or suffix in {'.', '..'} or '/' in suffix or '\\' in suffix for suffix in suffixes)
        ):
            message = f'Output members for {name!r} must be nonempty filename suffixes such as ".1.bt2" or "_R1.fq".'
            raise ValueError(message)
        candidates = [base.with_name(base.name + suffix) for suffix in suffixes] if suffixes else [base]
        for path in candidates:
            actual = path.resolve()
            if not actual.is_relative_to(root) or actual != path.absolute():
                message = f'Output {name!r} must resolve inside the validation run: {path}'
                raise ValueError(message)
        paths[name] = candidates
    return paths


def _diagnostics(result: DockerToolExecutionResult, root: Path) -> str:
    log_tail = ''
    log_path = result.output.log
    if log_path is not None and log_path.resolve().is_relative_to(root) and log_path.is_file():
        with log_path.open('rb') as log_file:
            log_file.seek(max(0, log_path.stat().st_size - DIAGNOSTIC_TAIL_CHARS))
            log_tail = log_file.read(DIAGNOSTIC_TAIL_CHARS).decode(errors='replace')
    return '\n'.join(tail for tail in (result.console_output[-DIAGNOSTIC_TAIL_CHARS:], log_tail) if tail)


def _execution_failed(runtime: ToolRuntime[ToolGeneratorContext], detail: str, *, draft: bool, attempt: int) -> Command:
    update: dict[str, Any] = {
        'messages': [_message(runtime, detail, error=True)],
        'execution_count': attempt,
        'retry_times': 1,
        'last_retry_error': detail,
    }
    if draft:
        update.update(validated=False, updated=False, should_end=False)
    return Command(update=update)


@tool(parse_docstring=True)
def execute_node(
    input_bindings: dict[str, str | NodeOutputBinding],
    runtime: ToolRuntime[ToolGeneratorContext],
    node_id: int | None = None,
    output_members: dict[str, list[str]] | None = None,
) -> Command:
    """Run a catalog node to prepare inputs, or execute and validate the current draft.

    Args:
        input_bindings: Bind each INPUT handle to a trial-relative path or to an earlier output using
            {"run_id": "...", "output": "...", "relative_path": "."}. Use a managed absolute container path
            for project/reference inputs from the report and literal strings for value: inputs.
        node_id: ID from search_node. Omit to validate the current draft.
        output_members: For draft OUTPUTs that declare a filename prefix, give the expected suffixes, e.g.
            {"index": [".1.bt2", ".2.bt2"]}. Other OUTPUTs are checked by exact path.
    """
    draft = node_id is None
    root = runtime.context.validation_root.resolve()
    attempt = runtime.state.get('execution_count', 0) + 1
    run_id = str(attempt)
    attempt_root = root / run_id
    execution = None
    if draft:
        missing_fields = [field for field in REQUIRED_DRAFT_FIELDS if field not in runtime.state]
        if missing_fields:
            return _execution_failed(
                runtime,
                f'Configure the draft before executing it. Missing: {", ".join(missing_fields)}.',
                draft=True,
                attempt=attempt,
            )
    try:
        attempt_root.mkdir()
        if node_id is None:
            config = DockerToolExecutionConfig.model_validate(runtime.state)
            _check_mount_targets(config)
        else:
            config = DockerToolExecutionConfig.model_validate(_load_node(node_id, runtime))
        inputs = _resolve_bindings(config, input_bindings, runtime, attempt_root / 'trial')
        inputs = isolate_validation_inputs(config, inputs, attempt_root, root)
        # A separate output path also gives inherited workspaces a distinct log filename.
        execution = prepare_docker_tool_execution(
            config,
            inputs,
            str(attempt_root / 'output'),
            'validation',
            config.modifiable_static_params or '',
            run_uid=root.name,
            managed_root=str(root),
        )
        execution = _protect_trial_data(execution, attempt_root)
        declared = _expected_draft_outputs(config, execution, attempt_root, output_members or {}) if draft else {}
        if draft:
            runtime.stream_writer({'validate_node': execution})
        result = execute_docker_tool_with_result(execution)
        diagnostics = _diagnostics(result, root)
        if result.exit_code:
            return _execution_failed(
                runtime,
                f'Exit code {result.exit_code}\n{execution.command}\n{diagnostics}',
                draft=draft,
                attempt=attempt,
            )

        if draft:
            missing_paths = [
                path
                for file in config.file_mounts
                if file.file_type == 'OUTPUT' and file.is_required
                for path in declared[file.name]
                if not path.resolve().is_relative_to(root) or not (path.is_file() or path.is_dir())
            ]
            missing_paths.extend(
                path
                for file in config.file_mounts
                if file.is_passthrough
                for path in (Path(result.output.result[file.name]),)
                if not path.resolve().is_relative_to(root) or not (path.is_file() or path.is_dir())
            )
        else:
            missing_paths = result.missing_outputs
        if missing_paths:
            missing = [
                str(path.resolve().relative_to(root)) if path.resolve().is_relative_to(root) else str(path)
                for path in missing_paths
            ]
            return _execution_failed(
                runtime,
                f'Required outputs were not generated: {", ".join(missing)}\n{execution.command}\n{diagnostics}',
                draft=draft,
                attempt=attempt,
            )

        available = {
            name: path
            for name, path in result.output.all_files().items()
            if Path(path).resolve().is_relative_to(root) and output_path_exists(path)
        }
        available['workdir'] = execution.output_dir
        summary = {
            'exit_code': result.exit_code,
            'run_id': run_id,
            'command': execution.command,
            'outputs': {name: str(Path(path).resolve().relative_to(root)) for name, path in available.items()},
            'log_tail': diagnostics,
        }
        if draft:
            return Command(
                update={
                    'validated': True,
                    'execution_count': attempt,
                    'updated': True,
                    'should_end': True,
                    'messages': [_message(runtime, json.dumps(summary, ensure_ascii=False))],
                }
            )

        recorded = ToolOutput(
            result=available, workspace=result.output.workspace, sample_name=result.output.sample_name
        )
        return Command(
            update={
                'validation_runs': {**runtime.state.get('validation_runs', {}), run_id: recorded},
                'execution_count': attempt,
                'messages': [_message(runtime, json.dumps(summary, ensure_ascii=False))],
            }
        )
    except (ValueError, OSError) as error:
        return _execution_failed(runtime, str(error), draft=draft, attempt=attempt)
    except Exception as error:
        logger.exception('Tool node execution failed')
        command = f'\n{execution.command}' if execution is not None else ''
        return _execution_failed(runtime, f'{type(error).__name__}: {error}{command}', draft=draft, attempt=attempt)
