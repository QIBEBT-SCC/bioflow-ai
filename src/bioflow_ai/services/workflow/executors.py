"""Node executor registry for the durable workflow runtime.

The scheduler never creates Celery signatures for individual node types. It
publishes one generic task and dispatches to this registry inside the worker.
Node implementations are plain Python callables and have no Celery lifecycle.
"""

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nanoid
from pydantic import BaseModel, ConfigDict

from bioflow_ai.models.workflow import WorkflowRun
from bioflow_ai.schemas.react_flow import Node, TaskInput, WorkflowDefinition
from bioflow_ai.schemas.tools import WorkflowToolDefinition
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


class ExecutorContext(BaseModel):
    """Run-scoped inputs shared by every node frozen during one compilation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    run: WorkflowRun
    sample_name: str | None
    tool_definitions: dict[str, WorkflowToolDefinition]


class PreparedExecutor(BaseModel):
    """JSON-serializable node execution plan and public metadata."""

    model_config = ConfigDict(frozen=True)

    kwargs: dict[str, Any]
    name: str
    track_progress: bool = True
    tool_uid: uuid.UUID | None = None
    base_dir: str | None = None


def _output_dir(context: ExecutorContext, position_index: int, suffix: str) -> str:
    return str(Path(context.run.base_dir) / f'{position_index}_{suffix}')


# Nodes whose plan is their inputs, some data fields and an indexed output directory:
# node type -> (output directory suffix, {data field: default}).
_INDEXED_NODES: dict[str, tuple[str, dict[str, Any]]] = {
    'rename_file': ('rename_file', {'new_file_name': ''}),
    'compress': ('compress', {'compressed_file_name': '', 'compression_format': None}),
    'decompress': ('decompress', {}),
    'code_bash': ('bash', {'code': ''}),
    'code_python': ('python', {'code': '', 'dependencies': []}),
    'code_R': ('r', {'code': '', 'dependencies': []}),
}


def prepare_node(
    node: Node,
    inputs: dict[str, dict[str, str] | list[dict[str, str]]],
    position_index: int,
    context: ExecutorContext,
) -> PreparedExecutor:
    """Freeze a node from an already-validated workflow without changing its ID."""
    data = node.data
    node_type = node.type

    if node_type == 'tool':
        tool_uid = uuid.UUID(str(data['tool_uid']))
        tool = context.tool_definitions[str(tool_uid)]
        output_dir = _output_dir(context, position_index, to_safe_path_segment(tool.name))
        return PreparedExecutor(
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
            kwargs={
                'file': data.get('file_path', ''),
                'project_id': context.run.project_id,
                'sample_uid': str(context.run.sample_uid) if context.run.sample_uid else None,
                'sample_name': context.sample_name,
            },
            name='read_file',
            track_progress=False,
        )

    if node_type == 'resource_frozen_path':
        return PreparedExecutor(
            kwargs={'path': data['path'], 'sample_name': data['sample_name']},
            name='foreach_input',
            track_progress=False,
        )

    if node_type == 'resource_text_file':
        output_dir = _output_dir(context, position_index, 'text_file')
        return PreparedExecutor(
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
            kwargs={
                'r1': data.get('r1', 'sample:r1'),
                'r2': data.get('r2', 'sample:r2'),
                'project_id': context.run.project_id,
                'sample_uid': str(context.run.sample_uid) if context.run.sample_uid else None,
                'sample_name': context.sample_name,
            },
            name='read_file',
            track_progress=False,
        )

    if node_type == 'resource_db':
        db_id = int(data['db_id'])
        return PreparedExecutor(
            kwargs={'db_id': db_id},
            name='read_file',
            track_progress=False,
        )

    if node_type == 'resource_ncbi_genome':
        return PreparedExecutor(
            kwargs={'inputs': inputs, 'required_index': data.get('required_index', [])},
            name='load_reference_genome',
        )

    if node_type in {'resource_GRCh38', 'resource_GRCm39'}:
        return PreparedExecutor(
            kwargs={'required_index': data.get('required_index', [])},
            name='load_reference_genome',
        )

    if node_type == 'value_string':
        return PreparedExecutor(
            kwargs={
                'value': data.get('value', ''),
                'value_type': 'string',
                'sample_uid': str(context.run.sample_uid) if context.run.sample_uid else None,
            },
            name='value',
            track_progress=False,
        )

    if node_type == 'bind_param':
        return PreparedExecutor(
            kwargs={'inputs': inputs, 'parameter': data.get('parameter', '')},
            name='bind_param',
            track_progress=False,
        )

    if node_type == 'copy2folder':
        folder_name = data.get('folder_name') or nanoid.generate(size=8)
        output_dir = str(Path(context.run.base_dir) / to_safe_path_segment(folder_name))
        return PreparedExecutor(
            kwargs={'inputs': inputs, 'output_dir': output_dir},
            name='copy2folder',
            base_dir=output_dir,
        )

    if node_type in {'project_mark', 'sample_mark'}:
        return PreparedExecutor(
            kwargs={
                'inputs': inputs,
                'run_uid': str(context.run.uid),
                'mark_name': data.get('mark_name', ''),
                'description': data.get('description', ''),
            },
            name=node_type,
            track_progress=False,
        )

    if node_type == 'resource_sample_mark_collection':
        output_dir = _output_dir(context, position_index, 'sample_mark_collection')
        return PreparedExecutor(
            kwargs={
                'project_id': context.run.project_id,
                'sample_uid': str(context.run.sample_uid) if context.run.sample_uid else None,
                'mark_names': data.get('mark_names', []),
                'require_all_samples': data.get('require_all_samples', True),
                'output_dir': output_dir,
            },
            name='sample_mark_collection',
            track_progress=False,
            base_dir=output_dir,
        )

    if node_type == 'collect_mount_dirs':
        output_dir = str(Path(context.run.base_dir) / f'{node.id}_mount_dirs')
        return PreparedExecutor(
            kwargs={'inputs': inputs, 'output_dir': output_dir},
            name='collect_mount_dirs',
            track_progress=False,
            base_dir=output_dir,
        )

    if node_type == 'select_file_in_folder':
        return PreparedExecutor(
            kwargs={'inputs': inputs, 'file_name': data.get('file_name', '')},
            name='select_file',
            track_progress=False,
        )

    if node_type == 'read_file_content':
        return PreparedExecutor(
            kwargs={'inputs': inputs, 'file_name': data.get('file_name', '')},
            name='read_file_content',
            track_progress=False,
        )

    if node_type == 'join_path':
        return PreparedExecutor(
            kwargs={'inputs': inputs, 'relative_path': data.get('relative_path', '')},
            name='join_path',
            track_progress=False,
        )

    if node_type == 'collect_file_collection':
        pattern = data['pattern']
        output_dir = _output_dir(context, position_index, 'file_collection')
        return PreparedExecutor(
            kwargs={'inputs': inputs, 'pattern': pattern.strip(), 'output_dir': output_dir},
            name=str(data.get('name') or 'ForEach'),
            base_dir=output_dir,
        )

    if node_type == 'foreach':
        child = WorkflowDefinition.model_validate(data['workflow'])
        output_dir = _output_dir(context, position_index, 'foreach')
        return PreparedExecutor(
            kwargs={
                'inputs': inputs,
                'workflow': child.model_dump(mode='json'),
                'item_port_id': data['item_port_id'],
                'subfolder_prefix': data['subfolder_prefix'],
                'output_dir': output_dir,
            },
            name=str(data.get('name') or 'ForEach'),
            track_progress=False,
            base_dir=output_dir,
        )

    if node_type in _INDEXED_NODES:
        suffix, defaults = _INDEXED_NODES[node_type]
        output_dir = _output_dir(context, position_index, suffix)
        return PreparedExecutor(
            kwargs={
                'inputs': inputs,
                **{key: data.get(key, default) for key, default in defaults.items()},
                'output_dir': output_dir,
            },
            name=node_type,
            base_dir=output_dir,
        )

    if node_type == 'downstream_summary':
        output_dir = str(Path(context.run.base_dir) / 'downstream')
        return PreparedExecutor(
            kwargs={'inputs': inputs, 'prompt': data.get('prompt', ''), 'output_dir': output_dir},
            name='summary',
            base_dir=output_dir,
        )

    message = f'Node {node.id}: unsupported executor type {node_type!r}'
    raise ValueError(message)


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


def execute_prepared_node(prepared_data: dict[str, Any]) -> ToolOutput:
    """Execute one frozen plan as a local Python call inside the generic task."""
    executor = _EXECUTORS[prepared_data['executor_name']]
    kwargs = prepared_data['kwargs'].copy()
    if (raw_inputs := kwargs.get('inputs')) is not None:
        kwargs['inputs'] = {
            key: [TaskInput.model_validate(item) for item in value]
            if isinstance(value, list)
            else TaskInput.model_validate(value)
            for key, value in raw_inputs.items()
        }

    return executor(**kwargs)
