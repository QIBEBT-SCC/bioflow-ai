import datetime
import json
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_milvus import BM25BuiltInFunction, Milvus
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from pydantic import BaseModel, Field, ValidationError
from sqlmodel import Session

import app.crud.workflow as wf_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.workflow import Workflow, WorkflowType
from app.schemas.react_flow import Edge, Node, Position, WorkflowDefinition
from llm.context import ProjectContext
from llm.models import qwen3_embedding
from llm.states import BuildEdge, BuildNode, WorkflowDraft
from llm.utils.edge_validation import check_orphan_edges
from llm.utils.formatters import format_validation_error, manager_help_command, node_formatter
from llm.utils.uid_mapper import generate_char_index


class HelpSchema(BaseModel):
    action: str = Field(description='The action name to get detailed schema for')


class ViewNodeSchema(BaseModel):
    """View details of a specific node."""

    node_id: str = Field(description='The ID of the node to view details for')


class DeleteNodeSchema(BaseModel):
    """Remove existing node from the panel."""

    node_id: str = Field(description='Node ID to be deleted')


class FixEdgeItem(BaseModel):
    source_node: str = Field(description='ID of the existing node that produces the data')
    source_handle: str = Field(description='Output handle name on the source node')
    target_node: str = Field(description='ID of the existing node that consumes the data')
    target_handle: str = Field(description='Input handle name on the target node')


class FixEdgeSchema(BaseModel):
    """Replace ALL orphan edges with the provided complete edge list in one call."""

    edges: list[FixEdgeItem] = Field(description='Complete list of edges to install in place of the orphan edges')


class CreateWorkflowDraftSchema(BaseModel):
    """Create a new workflow draft and switch the current panel to it."""

    name: str = Field(description="Workflow name, brief description of this workflow's tasks in English")
    description: str = Field(
        description='Description of the workflow, including the primary functions and the principal tools utilized.'
    )


class SwitchWorkflowDraftSchema(BaseModel):
    """Switch the current panel to another workflow draft."""

    draft_id: str = Field(description='Workflow draft ID shown by list_workflow_drafts')


class UpdateWorkflowMetadataSchema(BaseModel):
    """Update the active workflow draft name and/or description."""

    name: str | None = Field(default=None, description='New workflow name')
    description: str | None = Field(default=None, description='New workflow description')


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'list_workflow_drafts': None,
    'create_workflow_draft': CreateWorkflowDraftSchema,
    'switch_workflow_draft': SwitchWorkflowDraftSchema,
    'update_workflow_metadata': UpdateWorkflowMetadataSchema,
    'view_workflow': None,
    'view_node': ViewNodeSchema,
    'delete_node': DeleteNodeSchema,
    'fix_edge': FixEdgeSchema,
    'save_all_workflows': None,
}

HELP_MESSAGE = """workflow_manager contains the following available actions:
- `list_workflow_drafts`: List all workflow drafts in the current editing session
- `create_workflow_draft`: Create a new workflow draft and switch the current panel to it
- `switch_workflow_draft`: Snapshot the current panel and switch to another workflow draft
- `update_workflow_metadata`: Update the active workflow draft name or description
- `view_workflow`: View the current active workflow panel structure
- `view_node`: View the details of a specific node in the active workflow
- `delete_node`: Remove an existing node from the active workflow panel
- `fix_edge`: Replace orphan edges in the active workflow panel
- `save_all_workflows`: Save all workflow drafts into the database after every workflow is complete

Use action="help" with params={"action": "<action_name>"} to get the parameter schema for a specific action.
(e.g. workflow_manager(action="help" params={"action": "view_workflow"}))
"""


@tool(parse_docstring=True)
async def workflow_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[ProjectContext]) -> Command:
    """Workflow manager, execute operations related to workflow panel.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    match action:
        case 'help':
            # pyrefly: ignore [bad-argument-type]
            return manager_help_command(params, HelpSchema, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)

        case 'list_workflow_drafts':
            return _list_workflow_drafts(runtime)

        case 'create_workflow_draft':
            try:
                validate_param = CreateWorkflowDraftSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('create_workflow_draft', e, CreateWorkflowDraftSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _create_workflow_draft(validate_param.name, validate_param.description, runtime)

        case 'switch_workflow_draft':
            try:
                validate_param = SwitchWorkflowDraftSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('switch_workflow_draft', e, SwitchWorkflowDraftSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _switch_workflow_draft(validate_param.draft_id, runtime)

        case 'update_workflow_metadata':
            try:
                validate_param = UpdateWorkflowMetadataSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error(
                                    'update_workflow_metadata', e, UpdateWorkflowMetadataSchema
                                ),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _update_workflow_metadata(validate_param.name, validate_param.description, runtime)

        case 'view_workflow':
            return _view_workflow(runtime)

        case 'view_node':
            try:
                validate_param = ViewNodeSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('view_node', e, ViewNodeSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _view_node(validate_param.node_id, runtime)

        case 'delete_node':
            try:
                validate_param = DeleteNodeSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('view_node', e, DeleteNodeSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _delete_node(validate_param.node_id, runtime)

        case 'fix_edge':
            try:
                validate_param = FixEdgeSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('fix_edge', e, FixEdgeSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _fix_edge(validate_param, runtime)

        case 'save_all_workflows':
            return await _save_all_workflows(runtime)

        case _:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'Unknown action: "{action}". Use action="help" with params={{}} to see available actions.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )


def _get_workflow_drafts(runtime: ToolRuntime[ProjectContext]) -> dict[str, WorkflowDraft]:
    drafts = runtime.state.get('workflow_drafts', {})
    return {
        draft_id: draft if isinstance(draft, WorkflowDraft) else WorkflowDraft.model_validate(draft)
        for draft_id, draft in drafts.items()
    }


def _next_draft_id(drafts: dict[str, WorkflowDraft]) -> str:
    index = len(drafts) + 1
    while True:
        draft_id = f'wf_{generate_char_index(index)}'
        if draft_id not in drafts:
            return draft_id
        index += 1


def _snapshot_active_workflow(runtime: ToolRuntime[ProjectContext]) -> dict[str, WorkflowDraft]:
    drafts = _get_workflow_drafts(runtime)
    active_workflow_id = runtime.state.get('active_workflow_id')
    if not active_workflow_id:
        return drafts

    draft = drafts.get(active_workflow_id)
    if draft is None:
        draft = WorkflowDraft(draft_id=active_workflow_id, name=active_workflow_id)

    drafts[active_workflow_id] = draft.model_copy(
        update={
            'nodes': runtime.state.get('exist_nodes', []),
            'edges': runtime.state.get('exist_edges', []),
        },
        deep=True,
    )
    return drafts


def _list_workflow_drafts(runtime: ToolRuntime[ProjectContext]) -> Command:
    drafts = _snapshot_active_workflow(runtime)
    active_workflow_id = runtime.state.get('active_workflow_id')

    if not drafts:
        content = 'No workflow drafts exist yet. Use create_workflow_draft before building a workflow.'
    else:
        lines = ['Workflow drafts:']
        for draft in drafts.values():
            active_marker = ' (active)' if draft.draft_id == active_workflow_id else ''
            description = f' - {draft.description}' if draft.description else ''
            lines.append(
                f'- {draft.draft_id}{active_marker}: {draft.name}'
                f' [{len(draft.nodes)} nodes, {len(draft.edges)} edges]{description}'
            )
        content = '\n'.join(lines)

    return Command(
        update={
            'workflow_drafts': drafts,
            'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)],
        }
    )


def _create_workflow_draft(name: str, description: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    drafts = _snapshot_active_workflow(runtime)
    draft_id = _next_draft_id(drafts)
    current_nodes = runtime.state.get('exist_nodes', [])
    current_edges = runtime.state.get('exist_edges', [])

    if not runtime.state.get('active_workflow_id') and current_nodes:
        new_draft = WorkflowDraft(
            draft_id=draft_id,
            name=name,
            description=description,
            nodes=current_nodes,
            edges=current_edges,
        )
        drafts[draft_id] = new_draft
        msg = f'Registered the current panel as workflow draft {draft_id} ("{name}"). This draft is now active.'
        return Command(
            update={
                'active_workflow_id': draft_id,
                'workflow_drafts': drafts,
                'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
            }
        )

    new_draft = WorkflowDraft(draft_id=draft_id, name=name, description=description)
    drafts[draft_id] = new_draft
    msg = f'Created workflow draft {draft_id} ("{name}") and switched to an empty panel.'
    return Command(
        update={
            'active_workflow_id': draft_id,
            'workflow_drafts': drafts,
            'exist_nodes': Overwrite([]),
            'exist_edges': Overwrite([]),
            'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
        }
    )


def _switch_workflow_draft(draft_id: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    if not runtime.state.get('active_workflow_id') and runtime.state.get('exist_nodes', []):
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=(
                            'Cannot switch workflow drafts because the current panel has nodes but is not assigned '
                            'to any draft. Use create_workflow_draft to register the current panel first.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    drafts = _snapshot_active_workflow(runtime)
    target = drafts.get(draft_id)
    if target is None:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'Workflow draft "{draft_id}" not found. Use list_workflow_drafts to see available drafts.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    msg = f'Switched to workflow draft {target.draft_id} ("{target.name}").'
    return Command(
        update={
            'active_workflow_id': target.draft_id,
            'workflow_drafts': drafts,
            'exist_nodes': Overwrite(target.nodes),
            'exist_edges': Overwrite(target.edges),
            'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
        }
    )


def _update_workflow_metadata(
    name: str | None,
    description: str | None,
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    active_workflow_id = runtime.state.get('active_workflow_id')
    if not active_workflow_id:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='No active workflow draft. Use create_workflow_draft first.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    if name is None and description is None:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='No metadata changes provided. Pass at least one of name or description.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    drafts = _snapshot_active_workflow(runtime)
    draft = drafts[active_workflow_id]
    update_data = {}
    if name is not None:
        update_data['name'] = name
    if description is not None:
        update_data['description'] = description
    drafts[active_workflow_id] = draft.model_copy(update=update_data, deep=True)

    return Command(
        update={
            'workflow_drafts': drafts,
            'messages': [
                ToolMessage(
                    content=f'Updated metadata for workflow draft {active_workflow_id}.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _view_workflow(runtime: ToolRuntime[ProjectContext]) -> Command:
    if runtime.state.get('exist_nodes', []):
        nodes = [
            {
                'id': n.id,
                'name': n.name,
                'type': n.type,
                'x': n.x,
                'y': n.y,
            }
            for n in runtime.state.get('exist_nodes', [])
        ]
        edges = [
            {
                'source_node': e.source_node,
                'source_handle': e.source_handle,
                'target_node': e.target_node,
                'target_handle': e.target_handle,
            }
            for e in runtime.state.get('exist_edges', [])
        ]
        content = json.dumps({'nodes': nodes, 'edges': edges}, ensure_ascii=False)
    else:
        content = 'The current workflow panel is empty.'

    return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})


def _view_node(node_id: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    for n in runtime.state.get('exist_nodes', []):
        if n.id == node_id:
            return Command(
                update={'messages': [ToolMessage(content=node_formatter(n), tool_call_id=runtime.tool_call_id)]}
            )

    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=f'Node with ID {node_id} not found.',
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


def _delete_node(node_id: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    exist_edges: list[BuildEdge] = runtime.state.get('exist_edges', [])

    target_node = next((n for n in exist_nodes if n.id == node_id), None)
    if target_node is None:
        return Command(
            update={
                'messages': [
                    ToolMessage(content=f'Error: Node "{node_id}" not found.', tool_call_id=runtime.tool_call_id)
                ],
            }
        )

    removed_edges = [e for e in exist_edges if node_id in (e.source_node, e.target_node)]
    remaining_edges = [e for e in exist_edges if node_id not in (e.source_node, e.target_node)]
    remaining_nodes = [n for n in exist_nodes if n.id != node_id]

    # Reassign IDs sequentially and build remapping table
    id_map: dict[str, str] = {}
    for i, node in enumerate(remaining_nodes):
        new_id = generate_char_index(i + 1)
        if node.id != new_id:
            id_map[node.id] = new_id
            node.id = new_id

    # Update edge references with new IDs
    for edge in remaining_edges:
        if edge.source_node in id_map:
            edge.source_node = id_map[edge.source_node]
        if edge.target_node in id_map:
            edge.target_node = id_map[edge.target_node]

    # Build deletion report
    if removed_edges:
        edge_lines = []
        for e in removed_edges:
            if e.source_node == node_id:
                edge_lines.append(
                    f'  - outgoing → node {e.target_node} (handle: {e.source_handle} → {e.target_handle})'
                )
            else:
                edge_lines.append(
                    f'  - incoming ← node {e.source_node} (handle: {e.source_handle} → {e.target_handle})'
                )
        edge_summary = f'{len(removed_edges)} edge(s) removed:\n' + '\n'.join(edge_lines)
    else:
        edge_summary = 'No edges were connected to this node.'

    id_remap_summary = ', '.join(f'{old}→{new}' for old, new in id_map.items()) if id_map else 'no changes'

    msg = (
        f'Deleted node {node_id} ("{target_node.name}", type={target_node.type}, '
        f'pos={{x:{target_node.x}, y:{target_node.y}}}).\n'
        f'{edge_summary}\n'
        f'Node ID reassignment: {id_remap_summary}.'
    )

    return Command(
        update={
            'exist_nodes': Overwrite(remaining_nodes),
            'exist_edges': Overwrite(remaining_edges),
            'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
        }
    )


def _orphan_edges_report(orphans: list[BuildEdge], exist_nodes: list[BuildNode], tool_call_id: str) -> Command:
    node_names = {n.id: n.name for n in exist_nodes}
    remaining_ids = sorted(node_names)
    affected_sources = sorted({e.source_node for e in orphans})

    lines = [
        f'Cannot save workflow: {len(orphans)} orphan edge(s) detected (target_node is missing).',
        '',
        'Orphan edges:',
    ]
    for i, e in enumerate(orphans, 1):
        src_name = node_names.get(e.source_node, '<unknown>')
        lines.append(f'  {i}. {e.source_node}({src_name}) [{e.source_handle}] -> <missing> [{e.target_handle}]')

    lines += [
        '',
        f'Source nodes referenced by orphans: {affected_sources}',
        f'Current nodes on the panel: {remaining_ids}',
        '',
        'Use action="fix_edge" with params={"edges": [...]} to replace ALL orphan edges in one call. '
        'Pass the complete list of edges that should replace the orphans; '
        'every orphan edge is removed atomically and the provided edges are appended. '
        'After fixing, retry save_all_workflows.',
        '',
        'fix_edge parameters:',
        '  - edges (list): each item has source_node, source_handle, target_node, target_handle (all str)',
    ]
    return Command(update={'messages': [ToolMessage(content='\n'.join(lines), tool_call_id=tool_call_id)]})


def _fix_edge(params: FixEdgeSchema, runtime: ToolRuntime[ProjectContext]) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    exist_edges: list[BuildEdge] = runtime.state.get('exist_edges', [])

    nodes_by_id = {n.id: n for n in exist_nodes}
    errors: list[str] = []
    for i, item in enumerate(params.edges, 1):
        src = nodes_by_id.get(item.source_node)
        tgt = nodes_by_id.get(item.target_node)
        if src is None:
            errors.append(f"  edge {i}: source node '{item.source_node}' does not exist.")
            continue
        if tgt is None:
            errors.append(f"  edge {i}: target node '{item.target_node}' does not exist.")
            continue
        src_handles = {h.name for h in src.output_handles}
        if item.source_handle not in src_handles:
            errors.append(
                f"  edge {i}: source '{item.source_node}' has no output handle "
                f"'{item.source_handle}'. Available: {sorted(src_handles)}."
            )
        tgt_handles = {h.name for h in tgt.input_handles}
        if item.target_handle not in tgt_handles:
            errors.append(
                f"  edge {i}: target '{item.target_node}' has no input handle "
                f"'{item.target_handle}'. Available: {sorted(tgt_handles)}."
            )

    if errors:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='fix_edge rejected, no changes applied:\n' + '\n'.join(errors),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    cleaned_edges = [e for e in exist_edges if e.target_node is not None]
    removed = len(exist_edges) - len(cleaned_edges)
    new_edges = [
        BuildEdge(
            source_node=item.source_node,
            source_handle=item.source_handle,
            target_node=item.target_node,
            target_handle=item.target_handle,
        )
        for item in params.edges
    ]
    cleaned_edges.extend(new_edges)

    added_lines = '\n'.join(
        f'  + {item.source_node}[{item.source_handle}] -> {item.target_node}[{item.target_handle}]'
        for item in params.edges
    )
    msg = f'Removed {removed} orphan edge(s). Added {len(new_edges)} new edge(s):\n{added_lines}'

    return Command(
        update={
            'exist_edges': Overwrite(cleaned_edges),
            'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
        }
    )


def _build_workflow_definition(exist_nodes: list[BuildNode], exist_edges: list[BuildEdge]) -> WorkflowDefinition:
    nodes = []
    for n in exist_nodes:
        if n.type in {'resource_ncbi_genome', 'resource_GRCh38', 'resource_GRCm39'}:
            required_index = list(dict.fromkeys(e.source_handle for e in exist_edges if e.source_node == n.id))
            node = Node(
                id=n.id,
                type=n.type,
                data={'required_index': required_index},
                position=Position(x=n.x * 500, y=n.y * 400),
                dragHandle='.nodeDragable',
                zIndex=-10 if n.type == 'note' else 20,
            )
        else:
            node = Node(
                id=n.id,
                type=n.type,
                data=n.data,
                position=Position(x=n.x * 500, y=n.y * 400),
                dragHandle='.nodeDragable',
                zIndex=-10 if n.type == 'note' else 20,
            )
        nodes.append(node)

    edges = []
    for e in exist_edges:
        source_handle = f'{e.source_node}-out-{e.source_handle}'
        target_handle = f'{e.target_node}-in-{e.target_handle}'
        edge = Edge(
            id=f'xy-edge__{source_handle}-{target_handle}',
            animated=True,
            source=e.source_node,
            # pyrefly: ignore [bad-argument-type]
            target=e.target_node,
            sourceHandle=source_handle,
            targetHandle=target_handle,
        )
        edges.append(edge)

    return WorkflowDefinition(nodes=nodes, edges=edges)


async def _save_all_workflows(runtime: ToolRuntime[ProjectContext]) -> Command:
    drafts = _snapshot_active_workflow(runtime)

    if not drafts:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='No workflow drafts to save. Use create_workflow_draft before building workflows.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    empty_drafts = [draft.draft_id for draft in drafts.values() if not draft.nodes]
    if empty_drafts:
        return Command(
            update={
                'workflow_drafts': drafts,
                'messages': [
                    ToolMessage(
                        content=f'Cannot save: these workflow drafts are empty: {empty_drafts}.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    validation_errors = []
    for draft in drafts.values():
        orphans = check_orphan_edges(draft.edges)
        if orphans:
            validation_errors.append(f'- {draft.draft_id} ("{draft.name}") has {len(orphans)} orphan edge(s).')

    if validation_errors:
        return Command(
            update={
                'workflow_drafts': drafts,
                'messages': [
                    ToolMessage(
                        content=(
                            'Cannot save all workflows because some drafts have orphan edges:\n'
                            + '\n'.join(validation_errors)
                            + '\nSwitch to the affected draft, use fix_edge, then retry save_all_workflows.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    now = datetime.datetime.now(datetime.UTC)
    saved_workflows = []
    with Session(engine) as session:
        for draft in drafts.values():
            workflow_define = _build_workflow_definition(draft.nodes, draft.edges)
            workflow = Workflow(
                name=f'{now.strftime("%y%m%d")}_{draft.name}(AI)',
                description=draft.description,
                owner_id=runtime.context.user_id,
                wf_type=WorkflowType.TEMPLATE,
                workflow=workflow_define,
                public=True,
                auto_summary=True,
            )
            saved_workflows.append(wf_crud.insert(session, workflow))

        embedding_model = qwen3_embedding()
        vectorstore = Milvus(
            collection_name='WorkflowCollection',
            connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
            embedding_function=embedding_model,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],
            index_params=[{'index_type': 'HNSW', 'metric_type': 'L2'}, {}],
            consistency_level='Strong',
            drop_old=False,
            auto_id=True,
        )

        docs = [
            Document(
                page_content=workflow.description,
                metadata={
                    'wf_uid': str(workflow.uid),
                    'name': workflow.name,
                },
            )
            for workflow in saved_workflows
        ]
        await vectorstore.aadd_documents(docs)

    saved_lines = '\n'.join(f'- {workflow.name}: {workflow.uid}' for workflow in saved_workflows)
    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=f'Saved {len(saved_workflows)} workflow(s) to the database:\n{saved_lines}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            'exist_nodes': Overwrite([]),
            'exist_edges': Overwrite([]),
            'active_workflow_id': None,
            'workflow_drafts': {},
            'workflow_uids': [workflow.uid for workflow in saved_workflows],
            'updated': True,
        }
    )
