import json
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from pydantic import BaseModel, Field, ValidationError

from llm.context import ProjectContext
from llm.states import BuildEdge, BuildNode
from llm.utils.edge_validation import check_orphan_edges
from llm.utils.formatters import format_validation_error, manager_help_command, node_formatter
from llm.utils.uid_mapper import generate_char_index
from llm.workflow_builder.drafts import (
    WorkflowScope,
    build_workflow_draft,
    next_draft_id,
    snapshot_active_workflow,
)
from llm.workflow_builder.messages import (
    format_saved_workflows,
    format_workflow_drafts,
)
from llm.workflow_builder.persistence import save_workflow_drafts, workflow_uids


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
            # pyrefly: ignore [bad-argument-type]
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
                                content=format_validation_error(
                                    'delete_workflow_draft', e, DeleteWorkflowDraftSchema
                                ),
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

    empty_drafts = [draft.draft_id for draft in drafts.values() if not draft.nodes]
    if empty_drafts:
        return Command(
            update={
                'workflow_drafts': drafts,
                'messages': [
                    ToolMessage(
                        content=(
                            f'Cannot save: these workflow drafts are empty: {empty_drafts}. Switch to an intended '
                            'draft and complete it, or use delete_workflow_draft to remove a mistaken/duplicate draft. '
                            'Do not add placeholder nodes merely to make an unintended draft non-empty.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
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
        orphans = check_orphan_edges(draft.edges)
        if orphans:
            validation_errors.append(f'- {draft.draft_id} ("{draft.name}") has {len(orphans)} orphan edge(s).')

        node_types = {node.type for node in draft.nodes}
        if draft.execution_scope == 'sample_level':
            invalid = sorted(node_types & {'project_mark', 'resource_sample_mark_collection'})
            if invalid:
                validation_errors.append(
                    f'- {draft.draft_id} ("{draft.name}") is sample-level but contains project-only nodes: {invalid}.'
                )
        elif 'sample_mark' in node_types:
            validation_errors.append(
                f'- {draft.draft_id} ("{draft.name}") is project-level but contains `sample_mark`.'
            )

        mark_names = [
            str(node.data.get('mark_name', ''))
            for node in draft.nodes
            if node.type in {'project_mark', 'sample_mark'}
        ]
        duplicate_marks = sorted({name for name in mark_names if mark_names.count(name) > 1})
        if duplicate_marks:
            validation_errors.append(
                f'- {draft.draft_id} ("{draft.name}") publishes duplicate mark names: {duplicate_marks}.'
            )

        if draft.execution_scope == 'project_level':
            invalid_sample_refs = [
                node.id
                for node in draft.nodes
                if node.type == 'resource_file'
                and str(node.data.get('file_path', '')).startswith('sample:')
            ]
            if invalid_sample_refs:
                validation_errors.append(
                    f'- {draft.draft_id} ("{draft.name}") is project-level but contains sample-bound file '
                    f'references on nodes: {invalid_sample_refs}. Use `sample_mark_collection` for aggregation.'
                )

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
            'workflow_uids': Overwrite(workflow_uids(saved_workflows)),
            'updated': True,
        }
    )
