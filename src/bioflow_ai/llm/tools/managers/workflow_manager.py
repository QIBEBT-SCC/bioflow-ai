import json
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from pydantic import BaseModel, Field, ValidationError

from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.states import BuildEdge, BuildNode
from bioflow_ai.llm.workflow_builder.drafts import (
    WorkflowScope,
    build_workflow_draft,
    next_draft_id,
    snapshot_active_workflow,
)
from bioflow_ai.llm.workflow_builder.graphs import workflow_document
from bioflow_ai.llm.workflow_builder.messages import format_saved_workflows, format_workflow_drafts
from bioflow_ai.llm.workflow_builder.node_ids import remove_node_from_build_graph
from bioflow_ai.llm.workflow_builder.persistence import save_workflow_drafts, workflow_uids
from bioflow_ai.services.workflow.ports import effective_nodes
from bioflow_ai.services.workflow.validation import validate_workflow
from bioflow_ai.utils.formatters import format_validation_error, manager_help_command, node_formatter


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
    replace_all: bool = Field(
        default=False,
        description='Set true only when replacing the active workflow complete edge set, including valid edges.',
    )


class CreateWorkflowDraftSchema(BaseModel):
    """Create a new workflow draft and switch the current panel to it."""

    name: str = Field(description="Workflow name, brief description of this workflow's tasks in English")
    description: str = Field(
        description='Description of the workflow, including the primary functions and the principal tools utilized.'
    )
    execution_scope: Literal['sample_level', 'project_level'] = Field(
        description=(
            'Workflow execution scope. Use "sample_level" for workflows run once per sample; '
            'use "project_level" for preparatory or aggregation workflows run once globally for the project.'
        )
    )


class SwitchWorkflowDraftSchema(BaseModel):
    """Switch the current panel to another workflow draft."""

    draft_id: str = Field(description='Workflow draft ID shown by list_workflow_drafts')


class DeleteWorkflowDraftSchema(BaseModel):
    """Permanently delete an unsaved workflow draft from the current editing session."""

    draft_id: str = Field(description='Workflow draft ID shown by list_workflow_drafts')


class UpdateWorkflowMetadataSchema(BaseModel):
    """Update the active workflow draft name and/or description."""

    name: str | None = Field(default=None, description='New workflow name')
    description: str | None = Field(default=None, description='New workflow description')
    execution_scope: Literal['sample_level', 'project_level'] | None = Field(
        default=None,
        description='New workflow execution scope: "sample_level" or "project_level"',
    )


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'list_workflow_drafts': None,
    'create_workflow_draft': CreateWorkflowDraftSchema,
    'switch_workflow_draft': SwitchWorkflowDraftSchema,
    'delete_workflow_draft': DeleteWorkflowDraftSchema,
    'update_workflow_metadata': UpdateWorkflowMetadataSchema,
    'view_workflow': None,
    'view_node': ViewNodeSchema,
    'delete_node': DeleteNodeSchema,
    'fix_edge': FixEdgeSchema,
    'save_all_workflows': None,
}

HELP_MESSAGE = (
    'Call workflow_manager with action="help" and params={"action": "<action_name>"} '
    'to obtain the parameter schema for a specific action.'
)


@tool(parse_docstring=True)
async def workflow_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[ProjectContext]) -> Command:
    """Workflow manager, execute operations related to workflow panel.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    match action:
        case 'help':
            # ty: ignore[invalid-argument-type]
            return manager_help_command(params, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)

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
            return _create_workflow_draft(
                validate_param.name,
                validate_param.description,
                validate_param.execution_scope,
                runtime,
            )

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

        case 'delete_workflow_draft':
            try:
                validate_param = DeleteWorkflowDraftSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('delete_workflow_draft', e, DeleteWorkflowDraftSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _delete_workflow_draft(validate_param.draft_id, runtime)

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
            return _update_workflow_metadata(
                validate_param.name,
                validate_param.description,
                validate_param.execution_scope,
                runtime,
            )

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
                            content=f'Unknown action: "{action}". Check the workflow-builder system prompt for available actions.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )


def _list_workflow_drafts(runtime: ToolRuntime[ProjectContext]) -> Command:
    drafts = snapshot_active_workflow(runtime.state)
    active_workflow_id = runtime.state.get('active_workflow_id')
    content = format_workflow_drafts(list(drafts.values()), active_workflow_id)

    return Command(
        update={
            'workflow_drafts': drafts,
            'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)],
        }
    )


def _create_workflow_draft(
    name: str,
    description: str,
    execution_scope: WorkflowScope,
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    drafts = snapshot_active_workflow(runtime.state)
    normalized_name = _normalize_workflow_name(name)
    duplicate = next(
        (draft for draft in drafts.values() if _normalize_workflow_name(draft.name) == normalized_name),
        None,
    )
    if duplicate is not None:
        return Command(
            update={
                'workflow_drafts': drafts,
                'messages': [
                    ToolMessage(
                        content=(
                            f'Cannot create workflow draft "{name}": the same workflow already exists as '
                            f'{duplicate.draft_id} ("{duplicate.name}"). Use switch_workflow_draft to continue '
                            'editing it, or delete_workflow_draft first if it was created by mistake.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    draft_id = next_draft_id(drafts)
    current_nodes = runtime.state.get('exist_nodes', [])
    current_edges = runtime.state.get('exist_edges', [])

    if not runtime.state.get('active_workflow_id') and current_nodes:
        new_draft = build_workflow_draft(
            draft_id=draft_id,
            name=name,
            description=description,
            execution_scope=execution_scope,
            nodes=current_nodes,
            edges=current_edges,
            next_node_index=runtime.state.get('next_node_index'),
        )
        drafts[draft_id] = new_draft
        msg = (
            f'Registered the current panel as workflow draft {draft_id} '
            f'("{name}", scope={execution_scope}). This draft is now active.'
        )
        return Command(
            update={
                'active_workflow_id': draft_id,
                'workflow_drafts': drafts,
                'next_node_index': new_draft.next_node_index,
                'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
            }
        )

    new_draft = build_workflow_draft(
        draft_id=draft_id,
        name=name,
        description=description,
        execution_scope=execution_scope,
    )
    drafts[draft_id] = new_draft
    msg = (
        f'Created workflow draft {draft_id} ("{name}", scope={execution_scope}) and switched to an empty panel. '
        'Reuse this draft with switch_workflow_draft; do not create another draft for the same planned workflow.'
    )
    return Command(
        update={
            'active_workflow_id': draft_id,
            'workflow_drafts': drafts,
            'exist_nodes': Overwrite([]),
            'exist_edges': Overwrite([]),
            'next_node_index': new_draft.next_node_index,
            'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
        }
    )


def _normalize_workflow_name(name: str) -> str:
    """Normalize a workflow name for duplicate-draft checks."""
    return ' '.join(name.split()).casefold()


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

    drafts = snapshot_active_workflow(runtime.state)
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
            'next_node_index': target.next_node_index,
            'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
        }
    )


def _delete_workflow_draft(draft_id: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    drafts = snapshot_active_workflow(runtime.state)
    target = drafts.get(draft_id)
    if target is None:
        return Command(
            update={
                'workflow_drafts': drafts,
                'messages': [
                    ToolMessage(
                        content=(
                            f'Workflow draft "{draft_id}" not found. Use list_workflow_drafts to inspect the '
                            'current drafts before deleting one.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    del drafts[draft_id]
    remaining = ', '.join(drafts) if drafts else 'none'
    update: dict[str, Any] = {
        'workflow_drafts': drafts,
        'messages': [
            ToolMessage(
                content=f'Deleted workflow draft {draft_id} ("{target.name}"). Remaining drafts: {remaining}.',
                tool_call_id=runtime.tool_call_id,
            )
        ],
    }
    if runtime.state.get('active_workflow_id') == draft_id:
        update.update(
            {
                'active_workflow_id': None,
                'exist_nodes': Overwrite([]),
                'exist_edges': Overwrite([]),
                'next_node_index': 1,
            }
        )
    return Command(update=update)


def _update_workflow_metadata(
    name: str | None,
    description: str | None,
    execution_scope: WorkflowScope | None,
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

    if name is None and description is None and execution_scope is None:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='No metadata changes provided. Pass at least one of name, description, or execution_scope.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    drafts = snapshot_active_workflow(runtime.state)
    draft = drafts[active_workflow_id]
    if name is not None:
        normalized_name = _normalize_workflow_name(name)
        duplicate = next(
            (
                candidate
                for candidate in drafts.values()
                if candidate.draft_id != active_workflow_id
                and _normalize_workflow_name(candidate.name) == normalized_name
            ),
            None,
        )
        if duplicate is not None:
            return Command(
                update={
                    'workflow_drafts': drafts,
                    'messages': [
                        ToolMessage(
                            content=(
                                f'Cannot rename {active_workflow_id} to "{name}": that workflow name is already '
                                f'used by {duplicate.draft_id}. Switch to the existing draft or choose a unique name.'
                            ),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

    update_data = {}
    if name is not None:
        update_data['name'] = name
    if description is not None:
        update_data['description'] = description
    if execution_scope is not None:
        update_data['execution_scope'] = execution_scope
    drafts[active_workflow_id] = draft.model_copy(update=update_data, deep=True)

    candidate_state = {**runtime.state, 'workflow_drafts': drafts}
    result = validate_workflow(workflow_document(candidate_state, project_id=runtime.context.project_id))
    if not result.valid:
        return Command(update={'messages': [ToolMessage(content=result.message, tool_call_id=runtime.tool_call_id)]})

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
        nodes = []
        for node in effective_nodes(runtime.state.get('exist_nodes', []), runtime.state.get('exist_edges', [])):
            node_summary = {
                'id': node.id,
                'name': node.name,
                'type': node.type,
            }
            if node.type.startswith('code_'):
                node_summary['description'] = node.data.get('description', '')
            nodes.append(node_summary)
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
    for n in effective_nodes(runtime.state.get('exist_nodes', []), runtime.state.get('exist_edges', [])):
        if n.id == node_id:
            if n.type.startswith('code_'):
                content = json.dumps(
                    {
                        'id': n.id,
                        'name': n.name,
                        'type': n.type,
                        'description': n.data.get('description', ''),
                    },
                    ensure_ascii=False,
                )
            else:
                content = node_formatter(n)
            return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})

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

    remaining_nodes, remaining_edges, removed_edges = remove_node_from_build_graph(
        exist_nodes,
        exist_edges,
        node_id,
    )
    result = validate_workflow(
        workflow_document(
            runtime.state, nodes=remaining_nodes, edges=remaining_edges, project_id=runtime.context.project_id
        )
    )
    if not result.valid:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Cannot delete node; repair dependent edges first: ' + result.message,
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    remaining_nodes = effective_nodes(remaining_nodes, remaining_edges)

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

    msg = (
        f'Deleted node {node_id} ("{target_node.name}", type={target_node.type}).\n'
        f'{edge_summary}\n'
        'Remaining node IDs were preserved.'
    )

    update: dict[str, Any] = {
        'exist_nodes': Overwrite(remaining_nodes),
        'exist_edges': Overwrite(remaining_edges),
        'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
    }
    if 'subgraph_outputs' in runtime.state:
        kept_outputs = [output for output in runtime.state['subgraph_outputs'] if output['source_node'] != node_id]
        update['subgraph_outputs'] = kept_outputs
        if len(kept_outputs) != len(runtime.state['subgraph_outputs']):
            update['subgraph_outputs_set'] = False
    return Command(update=update)


def _fix_edge(params: FixEdgeSchema, runtime: ToolRuntime[ProjectContext]) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    exist_edges: list[BuildEdge] = runtime.state.get('exist_edges', [])

    cleaned_edges = [] if params.replace_all else [e for e in exist_edges if e.target_node is not None]
    removed = len(exist_edges) if params.replace_all else len(exist_edges) - len(cleaned_edges)
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
    result = validate_workflow(
        workflow_document(runtime.state, nodes=exist_nodes, edges=cleaned_edges, project_id=runtime.context.project_id)
    )
    if not result.valid:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='fix_edge rejected, no changes applied:\n' + result.message,
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    projected_nodes = effective_nodes(exist_nodes, cleaned_edges)

    added_lines = '\n'.join(
        f'  + {item.source_node}[{item.source_handle}] -> {item.target_node}[{item.target_handle}]'
        for item in params.edges
    )
    removed_kind = 'existing' if params.replace_all else 'orphan'
    msg = f'Removed {removed} {removed_kind} edge(s). Added {len(new_edges)} new edge(s):\n{added_lines}'

    return Command(
        update={
            'exist_nodes': Overwrite(projected_nodes),
            'exist_edges': Overwrite(cleaned_edges),
            'messages': [ToolMessage(content=msg, tool_call_id=runtime.tool_call_id)],
        }
    )


async def _save_all_workflows(runtime: ToolRuntime[ProjectContext]) -> Command:
    drafts = snapshot_active_workflow(runtime.state)

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

    drafts_by_name: dict[str, list[str]] = {}
    for draft in drafts.values():
        drafts_by_name.setdefault(_normalize_workflow_name(draft.name), []).append(draft.draft_id)
    duplicate_names = [draft_ids for draft_ids in drafts_by_name.values() if len(draft_ids) > 1]
    if duplicate_names:
        duplicate_summary = '; '.join(', '.join(draft_ids) for draft_ids in duplicate_names)
        return Command(
            update={
                'workflow_drafts': drafts,
                'messages': [
                    ToolMessage(
                        content=(
                            f'Cannot save because duplicate workflow drafts exist: {duplicate_summary}. Keep the '
                            'intended draft and remove the duplicate with delete_workflow_draft, then retry.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    validation_errors = []
    for draft in drafts.values():
        state = {'active_workflow_id': draft.draft_id, 'workflow_drafts': {draft.draft_id: draft}}
        document = workflow_document(
            state,
            nodes=draft.nodes,
            edges=draft.edges,
            fresh=True,
            project_id=runtime.context.project_id,
        )
        result = validate_workflow(document)
        if not result.valid:
            validation_errors.append(f'- {draft.draft_id} ("{draft.name}"): {result.message}')

    if validation_errors:
        return Command(
            update={
                'workflow_drafts': drafts,
                'messages': [
                    ToolMessage(
                        content=(
                            'Cannot save all workflows because some drafts are invalid:\n'
                            + '\n'.join(validation_errors)
                            + '\nSwitch to each affected draft, resolve the reported issue, then retry save_all_workflows.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    saved_workflows = await save_workflow_drafts(list(drafts.values()), runtime.context.user_id)
    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=format_saved_workflows(saved_workflows),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            'exist_nodes': Overwrite([]),
            'exist_edges': Overwrite([]),
            'active_workflow_id': None,
            'workflow_drafts': {},
            'next_node_index': 1,
            'workflow_uids': Overwrite(workflow_uids(saved_workflows)),
            'updated': True,
        }
    )
