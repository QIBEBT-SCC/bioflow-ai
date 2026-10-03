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

from bioflow_ai.core.session import engine
from bioflow_ai.crud import user as user_crud
from bioflow_ai.crud import workflow as workflow_crud
from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.states import BuildEdge, BuildNode, SubgraphBuildState, WorkflowDraft
from bioflow_ai.llm.tools.managers.subgraph_interface_manager import NamedPort
from bioflow_ai.llm.utils.formatters import format_validation_error, manager_help_command
from bioflow_ai.llm.utils.session import workflow_thread_id
from bioflow_ai.llm.workflow_builder.foreach_manifest import foreach_manifest_contract
from bioflow_ai.llm.workflow_builder.node_ids import allocate_node_id
from bioflow_ai.llm.workflow_builder.serialization import to_execution_scope
from bioflow_ai.llm.workflow_builder.subgraphs import effective_nodes, validate_editing_edges
from bioflow_ai.llm.workflow_builder.validation import check_edge_handles, check_tool_node_edges
from bioflow_ai.models.workflow import Workflow
from bioflow_ai.schemas.tools import Handle
from bioflow_ai.schemas.user_config import UserRole
from bioflow_ai.schemas.workflow_config import WorkflowType
from bioflow_ai.services import workflow_catalog


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


SCHEMAS: dict[str, type[BaseModel]] = {
    'search': SearchSubgraph,
    'view': ViewSubgraph,
    'create': CreateSubgraph,
    'add': AddSubgraph,
}


def _message(runtime: ToolRuntime[ProjectContext], content: str, **updates: Any) -> Command:
    return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)], **updates})


def _accessible(uid: uuid.UUID, user_id: int) -> Workflow | None:
    with Session(engine) as session:
        user = user_crud.get_by_id(session, user_id)
        if user is None or not user.is_active:
            return None
        items = workflow_crud.get_all_by_uid(
            session,
            [uid],
            user_id,
            include_private=user.role == UserRole.ADMIN,
            include_owned=user.role >= UserRole.MEMBER,
            wf_type=WorkflowType.SUBMODULE,
        )
        if not items:
            return None
        return items[0]


@tool(parse_docstring=True)
async def subgraph_node_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[ProjectContext]) -> Command:
    """Manage reusable subgraph nodes in the current workflow.

    Args:
        action: One of search, view, create, add, or help.
        params: Parameters for the selected action.
    """
    if action == 'help':
        # pyrefly: ignore [bad-argument-type]
        return manager_help_command(params, SCHEMAS, 'Inspect a subgraph action.', runtime.tool_call_id)
    schema = SCHEMAS.get(action)
    if schema is None:
        return _message(runtime, f'Unknown subgraph action {action!r}')
    try:
        values = schema.model_validate(params)
    except ValidationError as error:
        return _message(runtime, format_validation_error(action, error, schema))
    if action == 'search':
        return await _search(values, runtime)
    if action == 'view':
        return _view(values, runtime)
    if action == 'create':
        return await _create(values, runtime)
    return _add(values, runtime)


async def _search(values: SearchSubgraph, runtime: ToolRuntime[ProjectContext]) -> Command:
    with Session(engine) as session:
        user = user_crud.get_by_id(session, runtime.context.user_id)
        if user is None or not user.is_active:
            return _message(runtime, 'User is unavailable.')
        items = await workflow_catalog.search_subgraphs(
            session,
            values.query,
            runtime.context.user_id,
            include_private=user.role == UserRole.ADMIN,
            include_owned=user.role >= UserRole.MEMBER,
        )
        rows = [workflow_crud.to_simple(item) for item in items]
    return _message(runtime, json.dumps(rows, default=str, ensure_ascii=False))


def _view(values: ViewSubgraph, runtime: ToolRuntime[ProjectContext]) -> Command:
    item = _accessible(values.uid, runtime.context.user_id)
    if item is None:
        return _message(runtime, 'Subgraph not found.')
    return _message(
        runtime,
        json.dumps(
            {
                'uid': str(item.uid),
                'name': item.name,
                'description': item.description,
                'execution_scope': item.execution_scope.name,
                'workflow': item.workflow.model_dump(mode='json'),
            },
            ensure_ascii=False,
        ),
    )


async def _create(values: CreateSubgraph, runtime: ToolRuntime[ProjectContext]) -> Command:
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
                        'name': values.name,
                        'description': values.description,
                        'request': values.request,
                        'purpose': values.purpose,
                        'expected_inputs': [port.model_dump() for port in values.inputs],
                        'expected_outputs': [port.model_dump() for port in values.outputs],
                    },
                    ensure_ascii=False,
                )
            )
        ],
        'active_workflow_id': 'subgraph',
        'workflow_drafts': {
            'subgraph': WorkflowDraft(
                draft_id='subgraph',
                name=values.name,
                description=values.description,
                execution_scope=draft.execution_scope,
            )
        },
        'exist_nodes': [],
        'exist_edges': [],
        'next_node_index': 1,
        'subgraph_purpose': values.purpose,
        'subgraph_creation_id': f'{workflow_thread_id(runtime.config)}:{runtime.tool_call_id}',
    }
    child_context = runtime.context.model_copy(update={'delegate_human_requests': True})
    # pyrefly: ignore [no-matching-overload]
    response = await child.ainvoke(initial, context=child_context)
    if response.get('return_to_parent'):
        return _message(
            runtime, f'Subgraph creation needs parent action: {response.get("parent_request", "unknown issue")}'
        )
    uid = response.get('subgraph_uid')
    if not uid:
        return _message(runtime, 'Subgraph builder stopped without saving a valid subgraph.')
    item = _accessible(uuid.UUID(str(uid)), runtime.context.user_id)
    if item is None:
        return _message(runtime, 'Saved subgraph could not be loaded.')
    return _message(
        runtime,
        json.dumps(workflow_crud.to_simple(item), default=str, ensure_ascii=False),
        new_subgraphs=[item.uid],
        new_tools=response.get('new_tools', []),
        new_dbs=response.get('new_dbs', []),
    )


def _add(values: AddSubgraph, runtime: ToolRuntime[ProjectContext]) -> Command:
    item = _accessible(values.uid, runtime.context.user_id)
    if item is None or item.workflow.interface is None:
        return _message(runtime, 'Subgraph not found or has no interface.')
    active_id = runtime.state.get('active_workflow_id')
    draft = runtime.state.get('workflow_drafts', {}).get(active_id) if active_id else None
    if draft is None:
        return _message(runtime, 'Select a workflow draft before adding a subgraph.')
    draft = draft if isinstance(draft, WorkflowDraft) else WorkflowDraft.model_validate(draft)
    if item.execution_scope != to_execution_scope(draft.execution_scope):
        return _message(runtime, 'Subgraph execution scope must match the active workflow draft.')
    existing_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    existing_edges: list[BuildEdge] = runtime.state.get('exist_edges', [])
    interface = item.workflow.interface
    input_handles = [Handle(name=port.id, description=port.description) for port in interface.inputs]
    candidate_edges = [edge.model_copy(deep=True) for edge in values.input_edges]
    try:
        current_nodes = effective_nodes(existing_nodes, existing_edges)
    except ValueError as error:
        return _message(runtime, str(error))
    if error := check_edge_handles(candidate_edges, input_handles, current_nodes):
        return _message(runtime, error)
    if error := check_tool_node_edges(candidate_edges, input_handles):
        return _message(runtime, error)
    node_id, next_index = allocate_node_id(existing_nodes, runtime.state.get('next_node_index'))
    node = BuildNode(
        id=node_id,
        name=item.name,
        type='subgraph',
        input_handles=input_handles,
        output_handles=[Handle(name=port.id, description=port.description) for port in interface.outputs],
        data={
            'name': item.name,
            'description': item.description,
            'source_uid': str(item.uid),
            'workflow': item.workflow.model_dump(mode='json'),
        },
    )
    for edge in candidate_edges:
        edge.target_node = node_id
    proposed_edges = [*existing_edges, *candidate_edges]
    proposed_nodes = [*existing_nodes, node]
    if problems := validate_editing_edges(proposed_nodes, proposed_edges):
        return _message(runtime, 'Cannot add subgraph: ' + '; '.join(problems))
    node = effective_nodes(proposed_nodes, proposed_edges)[-1]
    response: dict[str, Any] = {
        'node_id': node_id,
        'name': item.name,
        'inputs': [p.model_dump() for p in interface.inputs],
        'outputs': [h.name for h in node.output_handles],
    }
    if any(
        edge.source_handle == 'collection'
        and (source := next((candidate for candidate in existing_nodes if candidate.id == edge.source_node), None))
        is not None
        and source.type == 'collect_file_collection'
        for edge in candidate_edges
    ):
        response['foreach_manifest_contract'] = foreach_manifest_contract(item.workflow)
    return _message(
        runtime,
        json.dumps(response, ensure_ascii=False),
        exist_nodes=[node],
        exist_edges=candidate_edges,
        next_node_index=next_index,
    )
