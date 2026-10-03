"""Node executor registry for the durable workflow runtime.

The scheduler never creates Celery signatures for individual node types. It
publishes one generic task and dispatches to this registry inside the worker.
Node implementations are plain Python callables and have no Celery lifecycle.
"""

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nanoid
from sqlalchemy.orm import Session

import bioflow_ai.crud.tools.tool as tool_crud
from bioflow_ai.schemas.react_flow import Node, TaskInput
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.services.workflow.collection import collect_file_collection
from bioflow_ai.tasks.archive_operation import compress, decompress
from bioflow_ai.tasks.code import execute_bash, execute_python, execute_r
from bioflow_ai.tasks.file_operation import (
    bind_param,
    collect_files,
    collect_mount_dirs,
    collect_sample_marks,
    create_text_file,
    join_path,
    project_mark,
    read_file_content,
    rename_file,
    sample_mark,
    select_file,
)
from bioflow_ai.tasks.load_tasks.load_bio_db import load_bio_db
from bioflow_ai.tasks.load_tasks.load_file import load_file, load_sequence
from bioflow_ai.tasks.load_tasks.load_genome import load_hg38, load_mm39, load_ncbi_genome
from bioflow_ai.tasks.load_tasks.static_value import set_value
from bioflow_ai.tasks.run_tool import execute_tool
from bioflow_ai.tasks.summary_task.summary import build_downstream_summary
from bioflow_ai.utils.file.paths import to_safe_path_segment

PRESENTATION_NODE_TYPES = frozenset({'note', 'image_preview'})


class ExecutorConfigurationError(ValueError):
    """A node cannot be frozen into an executable configuration."""


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

    executor_name: str
    kwargs: dict[str, Any]
    name: str
    track_progress: bool = True
    tool_uid: uuid.UUID | None = None
    base_dir: str | None = None

    def as_json(self) -> dict[str, Any]:
        """Return the durable JSON representation stored on a node run."""
        return {
            'executor_name': self.executor_name,
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
            executor_name='tool',
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
            executor_name='resource_file',
            kwargs={
                'file': data.get('file_path', ''),
                'project_id': context.project_id,
                'sample_uid': str(context.sample_uid) if context.sample_uid else None,
                'sample_name': context.sample_name,
            },
            name='read_file',
            track_progress=False,
        )

    if node_type == 'resource_frozen_path':
        return PreparedExecutor(
            executor_name=node_type,
            kwargs={'path': data['path'], 'sample_name': data['sample_name']},
            name='foreach_input',
            track_progress=False,
        )

    if node_type == 'resource_text_file':
        output_dir = _output_dir(context, 'text_file')
        return PreparedExecutor(
            executor_name='resource_text_file',
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
            executor_name='resource_sequence',
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
            executor_name='resource_db',
            kwargs={'db_id': db_id},
            name='read_file',
            track_progress=False,
        )

    if node_type == 'resource_ncbi_genome':
        return PreparedExecutor(
            executor_name='resource_ncbi_genome',
            kwargs={'inputs': inputs, 'required_index': data.get('required_index', [])},
            name='load_reference_genome',
        )

    if node_type in {'resource_GRCh38', 'resource_GRCm39'}:
        return PreparedExecutor(
            executor_name=node_type,
            kwargs={'required_index': data.get('required_index', [])},
            name='load_reference_genome',
        )

    if node_type == 'value_string':
        return PreparedExecutor(
            executor_name='value_string',
            kwargs={'value': data.get('value', ''), 'value_type': 'string'},
            name='value',
            track_progress=False,
        )

    if node_type == 'bind_param':
        return PreparedExecutor(
            executor_name='bind_param',
            kwargs={'inputs': inputs, 'parameter': data.get('parameter', '')},
            name='bind_param',
            track_progress=False,
        )

    if node_type == 'copy2folder':
        folder_name = data.get('folder_name') or nanoid.generate(size=8)
        output_dir = str(Path(context.output_base) / to_safe_path_segment(folder_name))
        return PreparedExecutor(
            executor_name='copy2folder',
            kwargs={'inputs': inputs, 'output_dir': output_dir},
            name='copy2folder',
            base_dir=output_dir,
        )

    if node_type in {'project_mark', 'sample_mark'}:
        return PreparedExecutor(
            executor_name=node_type,
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
            executor_name='resource_sample_mark_collection',
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
            executor_name='collect_mount_dirs',
            kwargs={'inputs': inputs, 'output_dir': output_dir},
            name='collect_mount_dirs',
            track_progress=False,
            base_dir=output_dir,
        )

    if node_type == 'select_file_in_folder':
        return PreparedExecutor(
            executor_name='select_file_in_folder',
            kwargs={'inputs': inputs, 'file_name': data.get('file_name', '')},
            name='select_file',
            track_progress=False,
        )

    if node_type == 'read_file_content':
        return PreparedExecutor(
            executor_name='read_file_content',
            kwargs={'inputs': inputs, 'file_name': data.get('file_name', '')},
            name='read_file_content',
            track_progress=False,
        )

    if node_type == 'join_path':
        return PreparedExecutor(
            executor_name='join_path',
            kwargs={'inputs': inputs, 'relative_path': data.get('relative_path', '')},
            name='join_path',
            track_progress=False,
        )

    if node_type == 'collect_file_collection':
        pattern = data.get('pattern')
        if not isinstance(pattern, str) or not pattern.strip():
            message = f'Node {node.id}: pattern must be a non-empty glob string'
            raise ExecutorConfigurationError(message)
        prefix = data.get('subfolder_prefix', 'item')
        if not isinstance(prefix, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', prefix) is None:
            message = f'Node {node.id}: subfolder prefix must be a safe directory name'
            raise ExecutorConfigurationError(message)
        if 'folder_path' not in inputs or isinstance(inputs['folder_path'], list):
            message = f'Node {node.id}: exactly one upstream folder is required'
            raise ExecutorConfigurationError(message)
        output_dir = _output_dir(context, 'file_collection')
        return PreparedExecutor(
            executor_name=node_type,
            kwargs={'inputs': inputs, 'pattern': pattern.strip(), 'output_dir': output_dir},
            name=str(data.get('name') or 'ForEach'),
            base_dir=output_dir,
        )

    if node_type == 'foreach':
        from bioflow_ai.schemas.react_flow import WorkflowDefinition

        child = WorkflowDefinition.model_validate(data.get('workflow'))
        interface = child.interface
        item_port_id = data.get('item_port_id')
        if interface is None or item_port_id not in {port.id for port in interface.inputs}:
            message = f'Node {node.id}: child workflow needs a current-file input'
            raise ExecutorConfigurationError(message)
        if 'collection' not in inputs or isinstance(inputs['collection'], list):
            message = f'Node {node.id}: exactly one ForEach collection is required'
            raise ExecutorConfigurationError(message)
        for port in interface.inputs:
            if port.id == item_port_id:
                continue
            if port.id not in inputs or isinstance(inputs[port.id], list):
                message = f'Node {node.id}: shared input {port.name!r} requires one edge'
                raise ExecutorConfigurationError(message)

        def check_child(graph: WorkflowDefinition) -> None:
            for child_node in graph.nodes:
                if child_node.type in {'foreach', 'sample_mark', 'project_mark'}:
                    message = f'Node {node.id}: {child_node.type} is unavailable inside ForEach'
                    raise ExecutorConfigurationError(message)
                if child_node.type == 'subgraph':
                    check_child(WorkflowDefinition.model_validate(child_node.data['workflow']))

        check_child(child)
        return PreparedExecutor(
            executor_name='foreach',
            kwargs={
                'inputs': inputs,
                'workflow': child.model_dump(mode='json'),
                'item_port_id': item_port_id,
                'subfolder_prefix': data['subfolder_prefix'] or 'item',
                'output_dir': _output_dir(context, 'foreach'),
            },
            name=str(data.get('name') or 'ForEach'),
            track_progress=False,
            base_dir=_output_dir(context, 'foreach'),
        )

    indexed_executors: dict[str, tuple[str, dict[str, Any], str]] = {
        'rename_file': (
            'rename_file',
            {'inputs': inputs, 'new_file_name': data.get('new_file_name', '')},
            'rename_file',
        ),
        'compress': (
            'compress',
            {
                'inputs': inputs,
                'compressed_file_name': data.get('compressed_file_name', ''),
                'compression_format': data.get('compression_format'),
            },
            'compress',
        ),
        'decompress': ('decompress', {'inputs': inputs}, 'decompress'),
        'code_bash': ('code_bash', {'inputs': inputs, 'code': data.get('code', '')}, 'bash'),
        'code_python': (
            'code_python',
            {'inputs': inputs, 'code': data.get('code', ''), 'dependencies': data.get('dependencies', [])},
            'python',
        ),
        'code_R': (
            'code_R',
            {'inputs': inputs, 'code': data.get('code', ''), 'dependencies': data.get('dependencies', [])},
            'r',
        ),
    }
    if node_type in indexed_executors:
        name, kwargs, suffix = indexed_executors[node_type]
        output_dir = _output_dir(context, suffix)
        return PreparedExecutor(
            executor_name=node_type,
            kwargs={**kwargs, 'output_dir': output_dir},
            name=name,
            base_dir=output_dir,
        )

    if node_type == 'downstream_summary':
        output_dir = str(Path(context.output_base) / 'downstream')
        return PreparedExecutor(
            executor_name='downstream_summary',
            kwargs={'inputs': inputs, 'prompt': data.get('prompt', ''), 'output_dir': output_dir},
            name='summary',
            base_dir=output_dir,
        )

    message = f'Node {node.id}: unsupported executor type {node_type!r}'
    raise ExecutorConfigurationError(message)


_EXECUTORS: dict[str, Callable[..., ToolOutput]] = {
    'tool': execute_tool,
    'resource_file': load_file,
    'resource_frozen_path': lambda *, path, sample_name: ToolOutput(
        result={'file_path': path}, sample_name=sample_name
    ),
    'resource_text_file': create_text_file,
    'resource_sequence': load_sequence,
    'resource_db': load_bio_db,
    'resource_ncbi_genome': load_ncbi_genome,
    'resource_GRCh38': load_hg38,
    'resource_GRCm39': load_mm39,
    'value_string': set_value,
    'bind_param': bind_param,
    'copy2folder': collect_files,
    'project_mark': project_mark,
    'sample_mark': sample_mark,
    'resource_sample_mark_collection': collect_sample_marks,
    'collect_mount_dirs': collect_mount_dirs,
    'select_file_in_folder': select_file,
    'read_file_content': read_file_content,
    'join_path': join_path,
    'collect_file_collection': collect_file_collection,
    'rename_file': rename_file,
    'compress': compress,
    'decompress': decompress,
    'code_bash': execute_bash,
    'code_python': execute_python,
    'code_R': execute_r,
    'downstream_summary': build_downstream_summary,
}


def supported_node_types() -> frozenset[str]:
    """Return all node types with an executor."""
    return frozenset(
        {
            'tool',
            'resource_file',
            'resource_frozen_path',
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
            'collect_file_collection',
            'foreach',
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
    """Execute one frozen plan as a local Python call inside the generic task."""
    executor_name = prepared_data.get('executor_name')
    if not isinstance(executor_name, str):
        message = f'Invalid frozen executor name: {executor_name!r}'
        raise TypeError(message)
    executor = _EXECUTORS.get(executor_name)
    if executor is None:
        message = f'Unknown frozen executor: {executor_name!r}'
        raise RuntimeError(message)

    kwargs = prepared_data.get('kwargs', {}).copy()
    if isinstance(raw_inputs := kwargs.get('inputs'), dict):
        kwargs['inputs'] = {
            key: [TaskInput.model_validate(item) for item in value]
            if isinstance(value, list)
            else TaskInput.model_validate(value)
            for key, value in raw_inputs.items()
        }

    if executor_name in {'tool', 'resource_db'}:
        kwargs['node_run_id'] = str(node_run_uid)
    result = executor(**kwargs)
    return result if isinstance(result, ToolOutput) else ToolOutput.model_validate(result)
