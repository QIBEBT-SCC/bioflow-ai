"""Find, delegate creation of, and insert saved subgraph snapshots."""

import json
import uuid
from typing import Any, Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from bioflow_ai.crud import user as user_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.states import BuildEdge, BuildNode, SubgraphBuildState, WorkflowDraft
from bioflow_ai.llm.tools.managers.subgraph_interface_manager import NamedPort
from bioflow_ai.llm.workflow_builder.foreach_manifest import foreach_manifest_contract
from bioflow_ai.llm.workflow_builder.graphs import workflow_document
from bioflow_ai.llm.workflow_builder.node_ids import allocate_node_id
from bioflow_ai.schemas.tools import Handle
from bioflow_ai.schemas.user_config import UserRole
from bioflow_ai.services.workflow import catalog as workflow_catalog
from bioflow_ai.services.workflow.ports import effective_nodes
from bioflow_ai.services.workflow.validation import validate_workflow
from bioflow_ai.utils.formatters import (
    format_interface_ports,
    format_subgraph_info,
    format_validation_error,
    manager_help_command,
)
from bioflow_ai.utils.llm.artifacts import parent_handoff_request
from bioflow_ai.utils.llm.session import workflow_thread_id


class SearchSubgraph(BaseModel):
    query: str = Field(min_length=1)


class ViewSubgraph(BaseModel):
    uid: uuid.UUID


class CreateSubgraph(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    request: str = Field(min_length=1, description='The complete single-item processing requirement')
    purpose: Literal['ordinary', 'foreach'] = 'ordinary'
    inputs: list[NamedPort] = Field(default_factory=list, description='Expected external inputs')
    outputs: list[NamedPort] = Field(default_factory=list, description='Expected exported outputs')


class AddSubgraph(BaseModel):
    uid: uuid.UUID
    input_edges: list[BuildEdge] = Field(default_factory=list)


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'search': SearchSubgraph,
    'view': ViewSubgraph,
    'create': CreateSubgraph,
    'add': AddSubgraph,
}

HELP_MESSAGE = (
    'Specify the action to inspect, for example: subgraph_node_manager(action="help", params={"action": "search"}).'
)


def _message(runtime: ToolRuntime[ProjectContext], content: str, **updates: Any) -> Command:
    return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)], **updates})


@tool(parse_docstring=True)
async def subgraph_node_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[ProjectContext]) -> Command:
    """Manage reusable subgraph nodes in the current workflow.

    Args:
        action: One of search, view, create, add, or help.
        params: Parameters for the selected action.
    """
    match action:
        case 'help':
            return manager_help_command(params, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)
        case 'search':
            try:
                validate_param = SearchSubgraph.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('search', e, SearchSubgraph),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return await _search(validate_param.query, runtime)
        case 'view':
            try:
                validate_param = ViewSubgraph.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('view', e, ViewSubgraph),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _view(validate_param.uid, runtime)
        case 'create':
            try:
                validate_param = CreateSubgraph.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('create', e, CreateSubgraph),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return await _create(
                validate_param.name,
                validate_param.description,
                validate_param.request,
                validate_param.purpose,
                validate_param.inputs,
                validate_param.outputs,
                runtime,
            )
        case 'add':
            try:
                validate_param = AddSubgraph.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('add', e, AddSubgraph),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _add(validate_param.uid, validate_param.input_edges, runtime)
        case _:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'Unknown action: "{action}". Check the workflow-builder system prompt for available actions.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )


async def _search(query: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    with Session(engine) as session:
        user = user_crud.get_by_id(session, runtime.context.user_id)
        if user is None or not user.is_active:
            return Command(
                update={'messages': [ToolMessage(content='User is unavailable.', tool_call_id=runtime.tool_call_id)]}
            )
        items = await workflow_catalog.search_subgraphs(
            session,
            query,
            runtime.context.user_id,
            include_private=user.role == UserRole.ADMIN,
            include_owned=user.role >= UserRole.MEMBER,
        )
        images_by_wf = workflow_catalog.workflow_tool_image_names(session, items)
        sections = [
            format_subgraph_info(
                item,
                images_by_wf.get(item.uid, ()),
                heading=item.name,
                include_ports=False,
            )
            for item in items
        ]
    content = '\n\n'.join(sections) if sections else 'No subgraphs found.'
    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=content,
                    tool_call_id=runtime.tool_call_id,
                    artifact={'result_count': len(items)},
                )
            ]
        }
    )


def _view(uid: uuid.UUID, runtime: ToolRuntime[ProjectContext]) -> Command:
    with Session(engine) as session:
        item = workflow_catalog.get_accessible_subgraph(session, uid, runtime.context.user_id)
        if item is None:
            return _message(runtime, 'Subgraph not found.')
        images_by_wf = workflow_catalog.workflow_tool_image_names(session, [item])
        content = format_subgraph_info(
            item,
            images_by_wf.get(item.uid, ()),
            heading='Subgraph Info',
            include_ports=True,
        )
    return _message(runtime, content)


async def _create(
    name: str,
    description: str,
    request: str,
    purpose: Literal['ordinary', 'foreach'],
    inputs: list[NamedPort],
    outputs: list[NamedPort],
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    active_id = runtime.state.get('active_workflow_id')
    draft = runtime.state.get('workflow_drafts', {}).get(active_id) if active_id else None
    if draft is None:
        return _message(runtime, 'Select a parent workflow draft before creating a subgraph.')
    draft = draft if isinstance(draft, WorkflowDraft) else WorkflowDraft.model_validate(draft)
    from bioflow_ai.llm.agents.workflow_agent.subgraph_builder import build_subgraph_builder

    child = build_subgraph_builder()
    initial: SubgraphBuildState = {
        'messages': [
            HumanMessage(
                content=json.dumps(
                    {
                        'name': name,
                        'description': description,
                        'request': request,
                        'purpose': purpose,
                        'expected_inputs': [port.model_dump() for port in inputs],
                        'expected_outputs': [port.model_dump() for port in outputs],
                    },
                    ensure_ascii=False,
                )
            )
        ],
        'active_workflow_id': 'subgraph',
        'workflow_drafts': {
            'subgraph': WorkflowDraft(
                draft_id='subgraph',
                name=name,
                description=description,
                execution_scope=draft.execution_scope,
            )
        },
        'exist_nodes': [],
        'exist_edges': [],
        'next_node_index': 1,
        'subgraph_purpose': purpose,
        'subgraph_creation_id': f'{workflow_thread_id(runtime.config)}:{runtime.tool_call_id}',
    }
    child_context = runtime.context.model_copy(update={'delegate_human_requests': True})
    response = await child.ainvoke(initial, context=child_context)
    if (parent_request := parent_handoff_request(response.get('messages', ()))) is not None:
        return _message(runtime, f'Subgraph creation needs parent action: {parent_request or "unknown issue"}')
    uid = response.get('subgraph_uid')
    if not uid:
        return _message(runtime, 'Subgraph builder stopped without saving a valid subgraph.')
    with Session(engine) as session:
        item = workflow_catalog.get_accessible_subgraph(session, uuid.UUID(str(uid)), runtime.context.user_id)
        if item is None:
            return _message(runtime, 'Saved subgraph could not be loaded.')
        images_by_wf = workflow_catalog.workflow_tool_image_names(session, [item])
        content = format_subgraph_info(
            item,
            images_by_wf.get(item.uid, ()),
            heading='Subgraph Created',
            include_ports=True,
        )
    return _message(
        runtime,
        content,
        new_subgraphs=[item.uid],
        new_tools=response.get('new_tools', []),
        new_dbs=response.get('new_dbs', []),
    )


def _add(uid: uuid.UUID, input_edges: list[BuildEdge], runtime: ToolRuntime[ProjectContext]) -> Command:
    with Session(engine) as session:
        item = workflow_catalog.get_accessible_subgraph(session, uid, runtime.context.user_id)
    if item is None:
        return _message(runtime, 'Subgraph not found.')
    active_id = runtime.state.get('active_workflow_id')
    draft = runtime.state.get('workflow_drafts', {}).get(active_id) if active_id else None
    if draft is None:
        return _message(runtime, 'Select a workflow draft before adding a subgraph.')
    existing_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    existing_edges: list[BuildEdge] = runtime.state.get('exist_edges', [])
    interface = item.workflow.interface
    input_handles = [
        Handle(name=port.id, description=port.description) for port in (interface.inputs if interface else [])
    ]
    candidate_edges = [edge.model_copy(deep=True) for edge in input_edges]
    node_id, next_index = allocate_node_id(existing_nodes, runtime.state.get('next_node_index'))
    node = BuildNode(
        id=node_id,
        name=item.name,
        type='subgraph',
        input_handles=input_handles,
        output_handles=[
            Handle(name=port.id, description=port.description) for port in (interface.outputs if interface else [])
        ],
        data={
            'name': item.name,
            'description': item.description,
            'source_uid': str(item.uid),
            'execution_scope': item.execution_scope.name.lower(),
            'workflow': item.workflow.model_dump(mode='json'),
        },
    )
    for edge in candidate_edges:
        edge.target_node = node_id
    proposed_edges = [*existing_edges, *candidate_edges]
    proposed_nodes = [*existing_nodes, node]
    result = validate_workflow(
        workflow_document(
            runtime.state, nodes=proposed_nodes, edges=proposed_edges, project_id=runtime.context.project_id
        )
    )
    if not result.valid:
        return _message(runtime, result.message)
    node = effective_nodes(proposed_nodes, proposed_edges)[-1]
    inputs = interface.inputs if interface else []
    outputs = [
        f'- `{h.name}`: {h.description.strip()}' if h.description.strip() else f'- `{h.name}`'
        for h in node.output_handles
    ] or ['- None']
    lines = [
        '## Added Subgraph Node',
        f'- node_id: {node_id}',
        f'- name: {item.name}',
        '',
        '### Inputs',
        *format_interface_ports(inputs),
        '',
        '### Outputs',
        *outputs,
    ]
    if any(
        edge.source_handle == 'collection'
        and (source := next((candidate for candidate in existing_nodes if candidate.id == edge.source_node), None))
        is not None
        and source.type == 'collect_file_collection'
        for edge in candidate_edges
    ):
        contract = foreach_manifest_contract(item.workflow)
        lines.extend(
            ['', '### Foreach Manifest Contract', '```json', json.dumps(contract, indent=2, ensure_ascii=False), '```']
        )
    return _message(
        runtime,
        '\n'.join(lines),
        exist_nodes=[node],
        exist_edges=candidate_edges,
        next_node_index=next_index,
    )
