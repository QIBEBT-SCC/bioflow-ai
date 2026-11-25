"""Declare and persist the boundary of one delegated subgraph draft."""

import datetime
import json
import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.workflow_builder.drafts import snapshot_active_workflow
from bioflow_ai.llm.workflow_builder.graphs import workflow_document
from bioflow_ai.llm.workflow_builder.serialization import to_execution_scope
from bioflow_ai.llm.workflow_builder.subgraphs import (
    INPUT_NODE_ID,
    build_subgraph_definition,
    input_ports,
    input_source,
)
from bioflow_ai.models.workflow import Workflow
from bioflow_ai.schemas.workflow_config import WorkflowType
from bioflow_ai.services.workflow import catalog as workflow_catalog
from bioflow_ai.services.workflow.validation import validate_workflow
from bioflow_ai.services.workflow.validation.adapters import build_graph, subgraph_graph
from bioflow_ai.services.workflow.validation.resolution import extract_draft_marks, resolve_workflow
from bioflow_ai.utils.formatters import format_validation_error, manager_help_command

if TYPE_CHECKING:
    from bioflow_ai.llm.states import BuildNode


class NamedPort(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default='', max_length=1000)


class AddInputs(BaseModel):
    inputs: list[NamedPort] = Field(min_length=1)


class RemoveInputs(BaseModel):
    input_ids: list[str] = Field(min_length=1)


class OutputChoice(NamedPort):
    id: str | None = None
    source_node: str
    source_handle: str


class SetOutputs(BaseModel):
    outputs: list[OutputChoice]


SCHEMAS: dict[str, type[BaseModel] | None] = {
    'add_subgraph_inputs': AddInputs,
    'remove_subgraph_inputs': RemoveInputs,
    'set_subgraph_outputs': SetOutputs,
    'save_subgraph': None,
}


def _message(runtime: ToolRuntime[ProjectContext], content: str, **updates: Any) -> Command:
    return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)], **updates})


@tool(parse_docstring=True)
async def subgraph_interface_manager(
    action: str, params: dict[str, Any], runtime: ToolRuntime[ProjectContext]
) -> Command:
    """Edit input and output ports, then save the delegated subgraph.

    Args:
        action: One of add_subgraph_inputs, remove_subgraph_inputs, set_subgraph_outputs, save_subgraph, or help.
        params: Parameters for the selected action.
    """
    if action == 'help':
        return manager_help_command(params, SCHEMAS, 'Inspect one subgraph interface action.', runtime.tool_call_id)
    schema = SCHEMAS.get(action)
    if action not in SCHEMAS:
        return _message(runtime, f'Unknown subgraph interface action {action!r}')
    try:
        values = schema.model_validate(params) if schema is not None else None
    except ValidationError as error:
        return _message(runtime, format_validation_error(action, error, schema))
    if isinstance(values, AddInputs):
        return _add_inputs(values, runtime)
    if isinstance(values, RemoveInputs):
        return _remove_inputs(values, runtime)
    if isinstance(values, SetOutputs):
        return _set_outputs(values, runtime)
    return await _save_subgraph(runtime)


def _add_inputs(values: AddInputs, runtime: ToolRuntime[ProjectContext]) -> Command:
    nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    ports = input_ports(nodes)
    names = {port['name'].casefold() for port in ports}
    if any(port.name.strip().casefold() in names for port in values.inputs) or len(
        {port.name.strip().casefold() for port in values.inputs}
    ) != len(values.inputs):
        return _message(runtime, 'Subgraph input names must be unique and cannot be empty.')
    used = {int(port['id'].removeprefix('input_')) for port in ports if port['id'].startswith('input_')}
    next_index = max(used, default=0) + 1
    added = []
    for item in values.inputs:
        port = {'id': f'input_{next_index}', 'name': item.name.strip(), 'description': item.description}
        added.append(port)
        ports.append(port)
        next_index += 1
    updated_nodes = [node for node in nodes if node.id != INPUT_NODE_ID]
    updated_nodes.insert(0, input_source(ports))
    result = validate_workflow(
        workflow_document(runtime.state, nodes=updated_nodes, project_id=runtime.context.project_id)
    )
    if not result.valid:
        return _message(runtime, result.message)
    return _message(
        runtime,
        json.dumps({'source_node': INPUT_NODE_ID, 'new_inputs': added}, ensure_ascii=False),
        exist_nodes=Overwrite(updated_nodes),
    )


def _remove_inputs(values: RemoveInputs, runtime: ToolRuntime[ProjectContext]) -> Command:
    ports = input_ports(runtime.state.get('exist_nodes', []))
    requested = set(values.input_ids)
    unknown = requested - {port['id'] for port in ports}
    if unknown:
        return _message(runtime, f'Unknown input IDs: {sorted(unknown)}')
    kept = [port for port in ports if port['id'] not in requested]
    nodes = [node for node in runtime.state.get('exist_nodes', []) if node.id != INPUT_NODE_ID]
    if kept:
        nodes.insert(0, input_source(kept))
    result = validate_workflow(workflow_document(runtime.state, nodes=nodes, project_id=runtime.context.project_id))
    if not result.valid:
        return _message(runtime, result.message)
    return _message(runtime, f'Removed inputs: {sorted(requested)}', exist_nodes=Overwrite(nodes))


def _set_outputs(values: SetOutputs, runtime: ToolRuntime[ProjectContext]) -> Command:
    previous = runtime.state.get('subgraph_outputs', [])
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    outputs = []
    next_index = (
        max(
            (int(output['id'].removeprefix('output_')) for output in previous if output['id'].startswith('output_')),
            default=0,
        )
        + 1
    )
    for choice in values.outputs:
        name = choice.name.strip()
        if not name or name.casefold() in seen_names:
            return _message(runtime, 'Subgraph output names must be unique and cannot be empty.')
        seen_names.add(name.casefold())
        match = next(
            (
                output
                for output in previous
                if output['source_node'] == choice.source_node and output['source_handle'] == choice.source_handle
            ),
            None,
        )
        port_id = choice.id or (match['id'] if match else f'output_{next_index}')
        if port_id in seen_ids or (choice.id and choice.id not in {output['id'] for output in previous}):
            return _message(runtime, f'Invalid or duplicate output ID {port_id!r}')
        seen_ids.add(port_id)
        if not choice.id and match is None:
            next_index += 1
        outputs.append(
            {
                'id': port_id,
                'name': name,
                'description': choice.description,
                'source_node': choice.source_node,
                'source_handle': choice.source_handle,
            }
        )
    result = validate_workflow(workflow_document(runtime.state, outputs=outputs, project_id=runtime.context.project_id))
    if not result.valid:
        return _message(runtime, result.message)
    return _message(
        runtime,
        json.dumps({'outputs': outputs}, ensure_ascii=False),
        subgraph_outputs=outputs,
        subgraph_outputs_set=True,
    )


async def _save_subgraph(runtime: ToolRuntime[ProjectContext]) -> Command:
    if runtime.state.get('subgraph_uid'):
        with Session(engine) as session:
            saved = session.get(Workflow, runtime.state['subgraph_uid'])
            if saved is None or saved.owner_id != runtime.context.user_id:
                return _message(runtime, 'Previously saved subgraph is unavailable.')
            interface = saved.workflow.interface
            return _message(
                runtime,
                json.dumps(
                    {
                        'uid': str(saved.uid),
                        'interface': interface.model_dump(mode='json') if interface is not None else None,
                    },
                    ensure_ascii=False,
                ),
            )
    drafts = snapshot_active_workflow(runtime.state)
    draft = drafts.get('subgraph')
    if draft is None or not draft.name.strip() or not draft.description.strip():
        return _message(runtime, 'Subgraph name and description are required.')
    if not runtime.state.get('subgraph_outputs_set'):
        return _message(runtime, 'Call set_subgraph_outputs before saving.')
    graph = subgraph_graph(draft.nodes, draft.edges, runtime.state.get('subgraph_outputs', []))
    draft_marks = extract_draft_marks([build_graph(item.nodes, item.edges) for item in drafts.values()])
    with Session(engine) as session:
        document = resolve_workflow(
            graph,
            session,
            fresh=True,
            execution_scope=draft.execution_scope,
            subgraph_purpose=runtime.state.get('subgraph_purpose', 'ordinary'),
            project_id=runtime.context.project_id,
            draft_marks=draft_marks,
        )
    result = validate_workflow(document)
    if not result.valid:
        return _message(runtime, f'Cannot save subgraph: {result.message}')
    definition = build_subgraph_definition(draft.nodes, draft.edges, runtime.state.get('subgraph_outputs', []))

    creation_id = runtime.state.get('subgraph_creation_id')
    if not creation_id:
        return _message(runtime, 'Missing subgraph creation ID; cannot save safely.')
    uid = uuid.uuid5(uuid.NAMESPACE_URL, f'bioflow-subgraph:{creation_id}')
    now = datetime.datetime.now(datetime.UTC)
    with Session(engine) as session:
        existing = session.get(Workflow, uid)
        if existing is not None:
            if existing.owner_id != runtime.context.user_id or existing.wf_type != WorkflowType.SUBMODULE:
                return _message(runtime, 'Subgraph creation ID conflicts with an inaccessible record.')
            saved = existing
        else:
            saved = await workflow_catalog.create_workflow(
                session,
                Workflow(
                    uid=uid,
                    name=f'{now:%y%m%d}_{draft.name}(AI)',
                    description=draft.description,
                    owner_id=runtime.context.user_id,
                    wf_type=WorkflowType.SUBMODULE,
                    execution_scope=to_execution_scope(draft.execution_scope),
                    workflow=definition,
                    public=True,
                ),
            )
    saved_interface = saved.workflow.interface
    return _message(
        runtime,
        json.dumps(
            {
                'uid': str(saved.uid),
                'interface': saved_interface.model_dump(mode='json') if saved_interface is not None else None,
            },
            ensure_ascii=False,
        ),
        subgraph_uid=saved.uid,
        updated=True,
    )
