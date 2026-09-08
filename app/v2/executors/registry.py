"""V2 node executor registry.

The V2 scheduler never creates Celery signatures for individual node types.  It
publishes one generic task and dispatches to this registry inside the worker.
The existing task modules are used only as Python execution implementations;
their V1 lifecycle hooks and Celery canvas are not invoked.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nanoid
from celery import Task
from sqlalchemy.orm import Session

import app.crud.tools.tool as tool_crud
from app.schemas.react_flow import Node, TaskInput
from app.schemas.workflow_config import ToolOutput
from app.tasks.archive_operation import compress_task, decompress_task
from app.tasks.code import bash_task, python_task, r_task
from app.tasks.file_operation import (
    bind_param_task,
    collect_files_task,
    collect_mount_dirs_task,
    collect_sample_marks_task,
    create_text_file_task,
    join_path_task,
    project_mark_task,
    read_file_content_task,
    rename_file_task,
    sample_mark_task,
    select_file_task,
)
from app.tasks.load_tasks.load_bio_db import load_bio_db_task
from app.tasks.load_tasks.load_file import load_file_task, load_sequence_task
from app.tasks.load_tasks.load_genome import load_hg38_task, load_mm39_task, load_ncbi_genome_task
from app.tasks.load_tasks.static_value import set_value_task
from app.tasks.run_tool import docker_task
from app.tasks.summary_task.summary import downstream_summary_task
from app.utils.input_format import to_safe_path_segment

PRESENTATION_NODE_TYPES = frozenset({'note'})


class ExecutorConfigurationError(ValueError):
    """A node cannot be frozen into an executable V2 configuration."""


@dataclass(frozen=True, slots=True)
class ExecutorContext:
    """Immutable run context used while freezing a node configuration."""

    session: Session
    run_uid: uuid.UUID
    generation: int
    project_id: int | None
    sample_uid: uuid.UUID | None
    sample_name: str | None
    output_base: str
    position_index: int


@dataclass(frozen=True, slots=True)
class PreparedExecutor:
    """JSON-serializable node execution plan and public metadata."""

    task_name: str
    kwargs: dict[str, Any]
    name: str
    track_progress: bool = True
    tool_uid: uuid.UUID | None = None
    base_dir: str | None = None

    def as_json(self) -> dict[str, Any]:
        """Return the durable JSON representation stored on NodeRunV2."""
        return {
            'task_name': self.task_name,
            'kwargs': self.kwargs,
        }


def _output_dir(context: ExecutorContext, suffix: str) -> str:
    return str(Path(context.output_base) / f'{context.position_index}_{suffix}')


def prepare_node(
    node: Node,
    inputs: dict[str, dict[str, str] | list[dict[str, str]]],
    context: ExecutorContext,
) -> PreparedExecutor:
    """Validate and freeze one executable node without mutating its ID."""
    data = node.data
    node_type = node.type

    if node_type == 'tool':
        raw_tool_uid = data.get('tool_uid')
        if not raw_tool_uid:
            message = f'Node {node.id}: tool_uid is required'
            raise ExecutorConfigurationError(message)
        try:
            tool_uid = uuid.UUID(str(raw_tool_uid))
        except ValueError as error:
            message = f'Node {node.id}: invalid tool_uid'
            raise ExecutorConfigurationError(message) from error
        tool = tool_crud.get(context.session, tool_uid)
        if tool is None:
            message = f'Node {node.id}: tool {tool_uid} not found'
            raise ExecutorConfigurationError(message)
        output_dir = _output_dir(context, to_safe_path_segment(tool.name))
        return PreparedExecutor(
            task_name=docker_task.name,
            kwargs={
                'tool_uid': str(tool_uid),
                'inputs': inputs,
                'output_dir': output_dir,
                'sample_name': context.sample_name or '',
                'params': data.get('modifiable_params') or '',
            },
            name=tool.name,
            tool_uid=tool_uid,
            base_dir=output_dir,
        )

    if node_type == 'resource_file':
        return PreparedExecutor(
            task_name=load_file_task.name,
            kwargs={
                'file': data.get('file_path', ''),
                'project_id': context.project_id,
                'sample_uid': str(context.sample_uid) if context.sample_uid else None,
                'sample_name': context.sample_name,
            },
            name='read_file',
            track_progress=False,
        )

    if node_type == 'resource_text_file':
        output_dir = _output_dir(context, 'text_file')
        return PreparedExecutor(
            task_name=create_text_file_task.name,
            kwargs={
                'file_name': data.get('file_name', ''),
                'content': data.get('content', ''),
                'output_dir': output_dir,
            },
            name='create_text_file',
            track_progress=False,
            base_dir=output_dir,
        )

    if node_type == 'resource_sequence':
        return PreparedExecutor(
            task_name=load_sequence_task.name,
            kwargs={
                'r1': data.get('r1', 'sample:r1'),
                'r2': data.get('r2', 'sample:r2'),
                'project_id': context.project_id,
                'sample_uid': str(context.sample_uid) if context.sample_uid else None,
                'sample_name': context.sample_name,
            },
            name='read_file',
            track_progress=False,
        )

    if node_type == 'resource_db':
        raw_db_id = data.get('db_id')
        if raw_db_id is None:
            message = f'Node {node.id}: db_id is required'
            raise ExecutorConfigurationError(message)
        try:
            db_id = int(raw_db_id)
        except (TypeError, ValueError) as error:
            message = f'Node {node.id}: db_id is required'
            raise ExecutorConfigurationError(message) from error
        return PreparedExecutor(
            task_name=load_bio_db_task.name,
            kwargs={'db_id': db_id},
            name='read_file',
            track_progress=False,
        )

    if node_type == 'resource_ncbi_genome':
        return PreparedExecutor(
            task_name=load_ncbi_genome_task.name,
            kwargs={'inputs': inputs, 'required_index': data.get('required_index', [])},
            name='load_reference_genome',
        )

    if node_type in {'resource_GRCh38', 'resource_GRCm39'}:
        task = load_hg38_task if node_type == 'resource_GRCh38' else load_mm39_task
        return PreparedExecutor(
            task_name=task.name,
            kwargs={'required_index': data.get('required_index', [])},
            name='load_reference_genome',
        )

    if node_type == 'value_string':
        return PreparedExecutor(
            task_name=set_value_task.name,
            kwargs={'value': data.get('value', ''), 'value_type': 'string'},
            name='value',
            track_progress=False,
        )

    if node_type == 'bind_param':
        return PreparedExecutor(
            task_name=bind_param_task.name,
            kwargs={'inputs': inputs, 'parameter': data.get('parameter', '')},
            name='bind_param',
            track_progress=False,
        )

    if node_type == 'copy2folder':
        folder_name = data.get('folder_name') or nanoid.generate(size=8)
        output_dir = str(Path(context.output_base) / to_safe_path_segment(folder_name))
        return PreparedExecutor(
            task_name=collect_files_task.name,
            kwargs={'inputs': inputs, 'output_dir': output_dir},
            name='copy2folder',
            base_dir=output_dir,
        )

    if node_type in {'project_mark', 'sample_mark'}:
        task = project_mark_task if node_type == 'project_mark' else sample_mark_task
        return PreparedExecutor(
            task_name=task.name,
            kwargs={
                'inputs': inputs,
                'run_uid': str(context.run_uid),
                'mark_name': data.get('mark_name', ''),
                'description': data.get('description', ''),
            },
            name=node_type,
            track_progress=False,
        )

    if node_type == 'resource_sample_mark_collection':
        output_dir = _output_dir(context, 'sample_mark_collection')
        return PreparedExecutor(
            task_name=collect_sample_marks_task.name,
            kwargs={
                'project_id': context.project_id,
                'sample_uid': str(context.sample_uid) if context.sample_uid else None,
                'mark_names': data.get('mark_names', []),
                'require_all_samples': data.get('require_all_samples', True),
                'output_dir': output_dir,
            },
            name='sample_mark_collection',
            track_progress=False,
            base_dir=output_dir,
        )

    if node_type == 'collect_mount_dirs':
        output_dir = str(Path(context.output_base) / f'{node.id}_mount_dirs')
        return PreparedExecutor(
            task_name=collect_mount_dirs_task.name,
            kwargs={'inputs': inputs, 'output_dir': output_dir},
            name='collect_mount_dirs',
            track_progress=False,
            base_dir=output_dir,
        )

    if node_type == 'select_file_in_folder':
        return PreparedExecutor(
            task_name=select_file_task.name,
            kwargs={'inputs': inputs, 'file_name': data.get('file_name', '')},
            name='select_file',
            track_progress=False,
        )

    if node_type == 'read_file_content':
        return PreparedExecutor(
            task_name=read_file_content_task.name,
            kwargs={'inputs': inputs, 'file_name': data.get('file_name', '')},
            name='read_file_content',
            track_progress=False,
        )

    if node_type == 'join_path':
        return PreparedExecutor(
            task_name=join_path_task.name,
            kwargs={'inputs': inputs, 'relative_path': data.get('relative_path', '')},
            name='join_path',
            track_progress=False,
        )

    indexed_tasks: dict[str, tuple[Task, str, dict[str, Any], str]] = {
        'rename_file': (
            rename_file_task,
            'rename_file',
            {'inputs': inputs, 'new_file_name': data.get('new_file_name', '')},
            'rename_file',
        ),
        'compress': (
            compress_task,
            'compress',
            {
                'inputs': inputs,
                'compressed_file_name': data.get('compressed_file_name', ''),
                'compression_format': data.get('compression_format'),
            },
            'compress',
        ),
        'decompress': (decompress_task, 'decompress', {'inputs': inputs}, 'decompress'),
        'code_bash': (bash_task, 'code_bash', {'inputs': inputs, 'code': data.get('code', '')}, 'bash'),
        'code_python': (
            python_task,
            'code_python',
            {'inputs': inputs, 'code': data.get('code', ''), 'dependencies': data.get('dependencies', [])},
            'python',
        ),
        'code_R': (
            r_task,
            'code_R',
            {'inputs': inputs, 'code': data.get('code', ''), 'dependencies': data.get('dependencies', [])},
            'r',
        ),
    }
    if node_type in indexed_tasks:
        task, name, kwargs, suffix = indexed_tasks[node_type]
        output_dir = _output_dir(context, suffix)
        return PreparedExecutor(
            task_name=task.name,
            kwargs={**kwargs, 'output_dir': output_dir},
            name=name,
            base_dir=output_dir,
        )

    if node_type == 'downstream_summary':
        output_dir = str(Path(context.output_base) / 'downstream')
        return PreparedExecutor(
            task_name=downstream_summary_task.name,
            kwargs={'inputs': inputs, 'prompt': data.get('prompt', ''), 'output_dir': output_dir},
            name='summary',
            base_dir=output_dir,
        )

    message = f'Node {node.id}: unsupported V2 executor type {node_type!r}'
    raise ExecutorConfigurationError(message)


_EXECUTION_TASKS: dict[str, Task] = {
    task.name: task
    for task in (
        docker_task,
        load_file_task,
        create_text_file_task,
        load_sequence_task,
        load_bio_db_task,
        load_ncbi_genome_task,
        load_hg38_task,
        load_mm39_task,
        set_value_task,
        bind_param_task,
        collect_files_task,
        project_mark_task,
        sample_mark_task,
        collect_sample_marks_task,
        collect_mount_dirs_task,
        select_file_task,
        read_file_content_task,
        join_path_task,
        rename_file_task,
        compress_task,
        decompress_task,
        bash_task,
        python_task,
        r_task,
        downstream_summary_task,
    )
}


def supported_node_types() -> frozenset[str]:
    """Return all node types with a V2 executor."""
    return frozenset(
        {
            'tool',
            'resource_file',
            'resource_text_file',
            'resource_sequence',
            'resource_db',
            'resource_ncbi_genome',
            'resource_GRCh38',
            'resource_GRCm39',
            'value_string',
            'bind_param',
            'copy2folder',
            'project_mark',
            'sample_mark',
            'resource_sample_mark_collection',
            'collect_mount_dirs',
            'select_file_in_folder',
            'read_file_content',
            'join_path',
            'rename_file',
            'compress',
            'decompress',
            'code_bash',
            'code_python',
            'code_R',
            'downstream_summary',
        }
    )


def execute_prepared_node(node_run_uid: uuid.UUID, prepared_data: dict[str, Any]) -> ToolOutput:
    """Execute one frozen plan as a local Python call inside the generic V2 task."""
    task_name = prepared_data.get('task_name')
    if not isinstance(task_name, str):
        message = f'Invalid frozen executor name: {task_name!r}'
        raise TypeError(message)
    task = _EXECUTION_TASKS.get(task_name)
    if task is None:
        message = f'Unknown frozen executor: {task_name!r}'
        raise RuntimeError(message)

    kwargs = prepared_data.get('kwargs', {}).copy()
    if isinstance(raw_inputs := kwargs.get('inputs'), dict):
        kwargs['inputs'] = {
            key: [TaskInput.model_validate(item) for item in value]
            if isinstance(value, list)
            else TaskInput.model_validate(value)
            for key, value in raw_inputs.items()
        }

    task.push_request(id=str(node_run_uid))
    try:
        result = task.run(**kwargs)
    finally:
        task.pop_request()
    return result if isinstance(result, ToolOutput) else ToolOutput.model_validate(result)
