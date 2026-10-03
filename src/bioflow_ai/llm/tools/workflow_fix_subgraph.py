"""Edit an embedded subgraph with the existing workflow builder managers."""

import json
import uuid
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from sqlalchemy.orm import Session

from bioflow_ai.core.session import engine
from bioflow_ai.crud import workflow as workflow_crud
from bioflow_ai.llm.context import WorkflowFixContext
from bioflow_ai.llm.states import BuildEdge, BuildNode, WorkflowFixState
from bioflow_ai.llm.workflow_builder.loading import load_workflow_draft
from bioflow_ai.llm.workflow_builder.serialization import build_workflow_definition
from bioflow_ai.llm.workflow_builder.subgraphs import (
    INPUT_NODE_ID,
    effective_nodes,
    input_source,
    validate_editing_edges,
)
from bioflow_ai.llm.workflow_builder.validation import check_orphan_edges, validate_note_anchor
from bioflow_ai.models.workflow import Workflow
from bioflow_ai.schemas.react_flow import PortTarget, WorkflowDefinition
from bioflow_ai.schemas.workflow_config import WorkflowType
from bioflow_ai.services.workflow.compiler import validate_workflow
from bioflow_ai.services.workflow.subgraphs import validate_interface_handles


def _reply(runtime: ToolRuntime[WorkflowFixContext, WorkflowFixState], message: str, **update: Any) -> Command:
    return Command(update={'messages': [ToolMessage(content=message, tool_call_id=runtime.tool_call_id)], **update})


def _panel(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        'exist_nodes': Overwrite(
            [node if isinstance(node, BuildNode) else BuildNode.model_validate(node) for node in frame['nodes']]
        ),
        'exist_edges': Overwrite(
            [edge if isinstance(edge, BuildEdge) else BuildEdge.model_validate(edge) for edge in frame['edges']]
        ),
        'next_node_index': frame['next_node_index'],
    }


def _load(runtime: ToolRuntime[WorkflowFixContext, WorkflowFixState], node_id: str) -> Command:
    nodes = runtime.state.get('exist_nodes', [])
    node = next((item for item in nodes if item.id == node_id and item.type == 'subgraph'), None)
    if node is None:
        return _reply(runtime, f'Subgraph node {node_id!r} is not on the current panel.')
    child = WorkflowDefinition.model_validate(node.data['workflow'])
    if child.interface is None:
        return _reply(runtime, f'Subgraph {node_id} has no interface.')
    stack = list(runtime.state.get('subgraph_stack', []))
    parent_path = stack[-1]['path'] if stack else []
    original_id = runtime.state.get('original_node_id_mapping', {})
    reverse = {value: key for key, value in original_id.items()}
    path = [*parent_path, reverse.get(node_id, node_id) if not stack else node_id]
    with Session(engine) as session:
        main = session.get(Workflow, uuid.UUID(runtime.context.workflow_uid))
        if main is None:
            return _reply(runtime, 'The workflow template no longer exists.')
        loaded = load_workflow_draft(
            child,
            name=str(node.data.get('name') or 'Subgraph'),
            description=str(node.data.get('description') or ''),
            execution_scope=main.execution_scope,
            session=session,
            preserve_ids=True,
        )
        baselines = dict(runtime.state.get('subgraph_source_baselines', {}))
        source_uid = node.data.get('source_uid')
        if isinstance(source_uid, str):
            try:
                source = workflow_crud.get(session, uuid.UUID(source_uid))
            except ValueError:
                source = None
            if (
                source is not None
                and source.wf_type == WorkflowType.SUBMODULE
                and source.owner_id == runtime.context.user_id
            ):
                baselines.setdefault(source_uid, source.workflow.model_dump(mode='json'))
    ports = [port.model_dump(mode='json') for port in child.interface.inputs]
    virtual = input_source(
        [{'id': port['id'], 'name': port['name'], 'description': port['description']} for port in ports]
    )
    boundary_edges = [
        BuildEdge(
            source_node=INPUT_NODE_ID,
            source_handle=port['id'],
            target_node=target['node_id'],
            target_handle=target['handle'],
        )
        for port in ports
        for target in port['targets']
    ]
    frame = {
        'node_id': node_id,
        'path': path,
        'nodes': list(nodes),
        'edges': list(runtime.state.get('exist_edges', [])),
        'next_node_index': runtime.state.get('next_node_index', len(nodes) + 1),
        'original_child': child.model_dump(mode='json'),
        'outputs': [port.model_dump(mode='json') for port in child.interface.outputs],
    }
    return _reply(
        runtime,
        json.dumps(
            {
                'path': path,
                'nodes': [item.id for item in loaded.draft.nodes],
                'inputs': ports,
                'outputs': frame['outputs'],
            },
            ensure_ascii=False,
        ),
        exist_nodes=Overwrite([virtual, *loaded.draft.nodes]),
        exist_edges=Overwrite([*loaded.draft.edges, *boundary_edges]),
        next_node_index=loaded.draft.next_node_index,
        subgraph_stack=[*stack, frame],
        subgraph_source_baselines=baselines,
    )


def _save(runtime: ToolRuntime[WorkflowFixContext, WorkflowFixState]) -> Command:
    stack = list(runtime.state.get('subgraph_stack', []))
    if not stack:
        return _reply(runtime, 'No subgraph is loaded.')
    frame = stack[-1]
    nodes: list[BuildNode] = list(runtime.state.get('exist_nodes', []))
    edges: list[BuildEdge] = list(runtime.state.get('exist_edges', []))
    virtual = next((node for node in nodes if node.id == INPUT_NODE_ID and node.type == 'subgraph_inputs'), None)
    if virtual is None or check_orphan_edges(edges) or (problems := validate_editing_edges(nodes, edges)):
        return _reply(
            runtime, 'Cannot save subgraph: invalid input node or edges. ' + '; '.join(problems if virtual else [])
        )
    original = WorkflowDefinition.model_validate(frame['original_child'])
    if original.interface is None:
        return _reply(runtime, 'Cannot save subgraph without an interface.')
    interface = original.interface.model_copy(deep=True)
    for port in interface.inputs:
        port.targets = [
            PortTarget(node_id=edge.target_node, handle=edge.target_handle)
            for edge in edges
            if edge.source_node == INPUT_NODE_ID and edge.source_handle == port.id and edge.target_node is not None
        ]
        if not port.targets:
            return _reply(runtime, f'Input {port.id} must be connected to an internal node.')
    outputs = frame['outputs']
    for port in interface.outputs:
        choice = next(item for item in outputs if item['id'] == port.id)
        port.source.node_id = choice['source']['node_id']
        port.source.handle = choice['source']['handle']
    real_nodes = [node for node in nodes if node.id != INPUT_NODE_ID]
    if not real_nodes:
        return _reply(runtime, 'A subgraph needs at least one internal node.')
    for note in real_nodes:
        if note.type == 'note' and (error := validate_note_anchor(note.data.get('anchor_node_id'), real_nodes)):
            return _reply(runtime, f'Cannot save subgraph note {note.id}: {error}')
    original_nodes = {node.id: node for node in original.nodes}
    definition = build_workflow_definition(
        real_nodes, [edge for edge in edges if edge.source_node != INPUT_NODE_ID], interface
    )
    definition.nodes = [
        node.model_copy(update={'position': old.position, 'dragHandle': old.dragHandle, 'zIndex': old.zIndex})
        if (old := original_nodes.get(node.id)) is not None
        else node
        for node in definition.nodes
    ]
    try:
        validate_workflow(definition)
        with Session(engine) as session:
            validate_interface_handles(definition, session)
    except ValueError as error:
        return _reply(runtime, f'Cannot save subgraph: {error}')
    parent_nodes = [BuildNode.model_validate(node).model_copy(deep=True) for node in frame['nodes']]
    target = next(node for node in parent_nodes if node.id == frame['node_id'])
    target.data['workflow'] = definition.model_dump(mode='json')
    changed = definition.model_dump(mode='json') != frame['original_child']
    paths = list(runtime.state.get('changed_subgraph_paths', []))
    if changed and frame['path'] not in paths:
        paths.append(frame['path'])
    return _reply(
        runtime,
        f'Saved subgraph {"/".join(frame["path"])} to the parent panel.' if changed else 'Subgraph was unchanged.',
        **_panel({**frame, 'nodes': parent_nodes}),
        subgraph_stack=stack[:-1],
        changed_subgraph_paths=paths,
    )


@tool('fix_subgraph', parse_docstring=True)
def fix_subgraph(
    action: Literal['load', 'save', 'cancel', 'view_interface', 'set_output'],
    runtime: ToolRuntime[WorkflowFixContext, WorkflowFixState],
    node_id: str | None = None,
    port_id: str | None = None,
    source_node: str | None = None,
    source_handle: str | None = None,
) -> Command:
    """Load, edit the interface of, save, or cancel an embedded subgraph.

    Args:
        action: Operation to perform on the current subgraph panel.
        node_id: Subgraph node to load from the current panel.
        port_id: Existing output port ID to repoint.
        source_node: Internal node supplying the output port.
        source_handle: Output handle on that internal node.
    """
    if action == 'load':
        return _load(runtime, node_id or '')
    stack = list(runtime.state.get('subgraph_stack', []))
    if not stack:
        return _reply(runtime, 'No subgraph is loaded.')
    frame = stack[-1]
    if action == 'cancel':
        return _reply(runtime, 'Subgraph edits discarded.', **_panel(frame), subgraph_stack=stack[:-1])
    if action == 'view_interface':
        ports = next(node.data['ports'] for node in runtime.state.get('exist_nodes', []) if node.id == INPUT_NODE_ID)
        return _reply(runtime, json.dumps({'inputs': ports, 'outputs': frame['outputs']}, ensure_ascii=False))
    if action == 'set_output':
        try:
            nodes = {
                node.id: node
                for node in effective_nodes(runtime.state.get('exist_nodes', []), runtime.state.get('exist_edges', []))
            }
        except ValueError as error:
            return _reply(runtime, f'Cannot map output: {error}')
        if (
            not port_id
            or source_node not in nodes
            or source_node == INPUT_NODE_ID
            or source_handle not in {port.name for port in nodes[source_node].output_handles}
        ):
            return _reply(runtime, 'Select an existing output port and a valid internal node output.')
        outputs = [dict(item) for item in frame['outputs']]
        chosen = next((item for item in outputs if item['id'] == port_id), None)
        if chosen is None:
            return _reply(runtime, f'Output port {port_id!r} does not exist.')
        chosen['source'] = {'node_id': source_node, 'handle': source_handle}
        stack[-1] = {**frame, 'outputs': outputs}
        return _reply(runtime, f'Output {port_id} now maps to {source_node}.{source_handle}.', subgraph_stack=stack)
    return _save(runtime)
