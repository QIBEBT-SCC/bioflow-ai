"""Read-only node commands, port definitions, connections and recorded results."""

from collections import defaultdict
from typing import Any
from urllib.parse import unquote

from sqlalchemy.orm import Session

from bioflow_ai.crud.tools import tool as tool_crud
from bioflow_ai.models.workflow import NodeRun
from bioflow_ai.schemas.react_flow import Node, WorkflowDefinition
from bioflow_ai.schemas.tools import Handle, ToolNodePublic
from bioflow_ai.services.tools.memory import list_tool_memories
from bioflow_ai.services.workflow.subgraphs import definition_id
from bioflow_ai.services.workflow.validation.registry import NODE_SPECS


def _handle_name(value: str, node_id: str, direction: str) -> str:
    return value.removeprefix(f'{node_id}-{direction}-')


def _source_reference(source: NodeRun, handle: str, path: tuple[str, ...]) -> dict[str, Any]:
    runtime_id = source.definition_node_id
    parts = (
        tuple(unquote(part) for part in runtime_id.removeprefix('@subgraph/').split('/'))
        if runtime_id.startswith('@subgraph/')
        else (runtime_id,)
    )
    return {
        'node_id': parts[-1],
        'handle': handle,
        **({'subgraph_path': list(parts[:-1])} if parts[:-1] != path else {}),
    }


def _input_sources(
    graph: WorkflowDefinition,
    node: Node,
    path: tuple[str, ...],
    kwargs: dict[str, Any],
    by_id: dict[str, NodeRun],
) -> dict[str, list[dict[str, Any]]]:
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph.edges:
        if edge.target == node.id:
            sources[_handle_name(edge.targetHandle, node.id, 'in')].append(
                {'node_id': edge.source, 'handle': _handle_name(edge.sourceHandle, edge.source, 'out')}
            )

    # Flattened execution references also cover inputs crossing an ordinary subgraph boundary.
    by_uid = {str(task.uid): task for task in by_id.values()}
    for name, references in kwargs.get('inputs', {}).items():
        if name in sources or node.type in {'subgraph', 'foreach'}:
            continue
        for reference in references if isinstance(references, list) else [references]:
            source = by_uid.get(reference.get('source_uid'))
            if source is not None:
                sources[name].append(_source_reference(source, reference['source_k'], path))
    return sources


def _recorded_result(
    node: Node,
    handle: str,
    path: tuple[str, ...],
    by_id: dict[str, NodeRun],
) -> str:
    task = by_id.get(definition_id((*path, node.id)))
    if task is not None:
        output = task.tool_output
        if output is None:
            return 'no_result'
        if handle == 'workdir' and task.base_dir:
            return task.base_dir
        value = output.all_files().get(handle)
        return str(value) if value is not None else 'no_result'
    if node.type == 'subgraph':
        child = WorkflowDefinition.model_validate(node.data['workflow'])
        port = next((port for port in child.interface.outputs if port.id == handle), None) if child.interface else None
        source = next((item for item in child.nodes if port and item.id == port.source.node_id), None)
        if source is not None and port is not None:
            return _recorded_result(source, port.source.handle, (*path, node.id), by_id)
    return 'no_result'


def _with_connected_handles(handles: list[Handle], names: list[str]) -> list[Handle]:
    known = {handle.name for handle in handles}
    return [*handles, *(Handle(name=name, description='') for name in dict.fromkeys(names) if name not in known)]


def describe_node(
    session: Session,
    graph: WorkflowDefinition,
    node: Node,
    path: tuple[str, ...],
    by_id: dict[str, NodeRun],
) -> dict[str, Any]:
    """Describe a node's usable ports without exposing its internal task record.

    Commands combine the tool definition with this node's parameters, preserving
    unresolved file placeholders. Results come from durable output records; a missing
    result is not a filesystem-existence check.
    """
    task = by_id.get(definition_id((*path, node.id)))
    kwargs = (task.prepared_data or {}).get('kwargs', {}) if task is not None else {}
    details: dict[str, Any] = {}
    spec = NODE_SPECS.get(node.type)
    inputs, outputs = list(spec.input_handles) if spec else [], list(spec.output_handles) if spec else []
    if node.type == 'tool':
        tool_uid = (task.tool_uid if task is not None else None) or node.data.get('tool_uid')
        tool = tool_crud.get_summary(session, tool_uid) if tool_uid else None
        if tool is not None:
            projection = ToolNodePublic(
                uid=tool.uid,
                name=tool.name,
                description=tool.description,
                tags=[],
                command_template=tool.command_template,
                dynamic_params=tool.dynamic_params or [],
                file_mounts=tool.file_mounts or [],
                immutable_static_params=tool.immutable_static_params,
                modifiable_static_params=kwargs.get('params', node.data.get('modifiable_params', '')),
            )
            inputs, outputs = projection.input_handles, projection.output_handles
            details.update(
                name=node.data.get('name') or projection.name,
                description=projection.description,
                command_template=projection.complete_command,
            )
            if memories := list_tool_memories(session, tool.uid):
                details['memories'] = [
                    {'memory_id': str(memory.uid), 'description': memory.description} for memory in memories
                ]
        else:
            details['definition_error'] = 'Tool definition is unavailable.'
    else:
        data = {key: value for key, value in node.data.items() if key not in {'name', 'workflow'} and value is not None}
        if data:
            details['data'] = data
        if node.type in {'subgraph', 'foreach'}:
            child = WorkflowDefinition.model_validate(node.data['workflow'])
            if child.interface is not None:
                inputs = [Handle(name=port.id, description=port.description) for port in child.interface.inputs]
                outputs = [Handle(name=port.id, description=port.description) for port in child.interface.outputs]
            if task is not None and task.node_type == 'foreach':
                outputs = [Handle(name='results_folder', description='ForEach results')]

    sources = _input_sources(graph, node, path, kwargs, by_id)
    inputs = _with_connected_handles(inputs, list(sources))
    outputs = _with_connected_handles(
        outputs,
        [_handle_name(edge.sourceHandle, node.id, 'out') for edge in graph.edges if edge.source == node.id],
    )
    details['input_handles'] = [
        {'name': handle.name, 'description': handle.description, 'sources': sources.get(handle.name, [])}
        for handle in inputs
    ]
    details['output_handles'] = [
        {
            'name': handle.name,
            'description': handle.description,
            'result': _recorded_result(node, handle.name, path, by_id),
        }
        for handle in outputs
    ]
    return details
