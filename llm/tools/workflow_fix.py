"""Final persistence step for the workflow-fixer agent."""

import uuid
from pathlib import Path
from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlalchemy.orm import Session

from app.core.session import engine
from app.models.workflow import Workflow
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.workflow_config import WorkflowType
from app.services.workflow.compiler import validate_workflow
from app.services.workflow.subgraphs import validate_interface_handles
from app.utils.system import utc_now
from llm.context import WorkflowFixContext
from llm.states import BuildEdge, BuildNode, WorkflowFixState
from llm.utils.uid_mapper import generate_char_index
from llm.workflow_builder.node_ids import compose_node_id_mappings
from llm.workflow_builder.serialization import build_workflow_definition
from llm.workflow_builder.subgraphs import effective_nodes, remap_interface, validate_editing_edges
from llm.workflow_builder.validation import check_orphan_edges, validate_note_anchor


def _validation_errors(nodes: list[BuildNode], edges: list[BuildEdge]) -> list[str]:
    errors: list[str] = []
    if not nodes:
        errors.append('The workflow cannot be empty.')
    if orphans := check_orphan_edges(edges):
        errors.append(f'The workflow has {len(orphans)} orphan edge(s).')
    errors.extend(validate_editing_edges(nodes, edges))
    nodes = effective_nodes(nodes, edges) if not errors else nodes

    nodes_by_id = {node.id: node for node in nodes}
    if len(nodes_by_id) != len(nodes):
        errors.append('The workflow contains duplicate node IDs.')
    for edge in edges:
        source = nodes_by_id.get(edge.source_node)
        target = nodes_by_id.get(edge.target_node or '')
        if source is None or target is None:
            errors.append(f'Edge {edge.source_node} -> {edge.target_node} references a missing node.')
            continue
        if edge.source_handle not in {handle.name for handle in source.output_handles}:
            errors.append(f'Node {source.id} has no output handle {edge.source_handle!r}.')
        if edge.target_handle not in {handle.name for handle in target.input_handles}:
            errors.append(f'Node {target.id} has no input handle {edge.target_handle!r}.')
    errors.extend(
        f'Note {node.id}: {error}'
        for node in nodes
        if node.type == 'note'
        if (error := validate_note_anchor(node.data.get('anchor_node_id'), nodes))
    )
    return errors


def _preserve_canvas_metadata(
    definition: WorkflowDefinition,
    original: WorkflowDefinition,
    saved_to_original_ids: dict[str, str],
) -> WorkflowDefinition:
    original_nodes = {node.id: node for node in original.nodes}
    nodes = []
    for node in definition.nodes:
        original_id = saved_to_original_ids.get(node.id)
        previous = original_nodes.get(original_id) if original_id is not None else None
        if previous is None:
            nodes.append(node)
            continue
        nodes.append(
            node.model_copy(
                update={
                    'position': previous.position,
                    'dragHandle': previous.dragHandle,
                    'zIndex': previous.zIndex,
                }
            )
        )
    return definition.model_copy(update={'nodes': nodes})


def _preserve_node_ids(
    nodes: list[BuildNode],
    edges: list[BuildEdge],
    original_to_editing: dict[str, str],
) -> tuple[list[BuildNode], list[BuildEdge], dict[str, str]]:
    """Restore IDs of surviving nodes and allocate new IDs without reusing old ones."""
    editing_to_original = {editing: original for original, editing in original_to_editing.items()}
    occupied = set(original_to_editing)
    mapping: dict[str, str] = {}
    next_index = 1
    for node in nodes:
        if node.id in editing_to_original:
            mapping[node.id] = editing_to_original[node.id]
            continue
        while (candidate := generate_char_index(next_index)) in occupied:
            next_index += 1
        mapping[node.id] = candidate
        occupied.add(candidate)
        next_index += 1
    saved_nodes = [node.model_copy(deep=True) for node in nodes]
    for node in saved_nodes:
        node.id = mapping[node.id]
        if node.type == 'note' and node.data.get('anchor_node_id') in mapping:
            node.data['anchor_node_id'] = mapping[node.data['anchor_node_id']]
    saved_edges = [edge.model_copy(deep=True) for edge in edges]
    for edge in saved_edges:
        edge.source_node = mapping[edge.source_node]
        if edge.target_node is not None:
            edge.target_node = mapping[edge.target_node]
    return saved_nodes, saved_edges, mapping


def _stage_update(
    runtime: ToolRuntime[WorkflowFixContext, WorkflowFixState],
    summary: str,
    sync_lines: list[str],
) -> tuple[Path, Path, str]:
    project_root = Path(runtime.context.workdir).expanduser().resolve()
    diagnosis_uid = uuid.UUID(runtime.context.diagnosis_run_uid) if runtime.context.diagnosis_run_uid else None
    fix_uid = uuid.UUID(runtime.context.agent_run_uid)
    output_dir = (
        project_root / 'fixes' / str(diagnosis_uid or runtime.context.source_run_uid) / 'updates' / str(fix_uid)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    update_path = output_dir / 'UPDATE.md'
    temporary_path = output_dir / f'.UPDATE.{uuid.uuid4().hex}.tmp'
    diagnosis_field = f'diagnosis_run_uid: {diagnosis_uid}\n' if diagnosis_uid else ''
    content = (
        '---\n'
        'schema_version: 1\n'
        f'source_run_uid: {runtime.context.source_run_uid}\n'
        f'{diagnosis_field}'
        f'fix_run_uid: {fix_uid}\n'
        f'workflow_uid: {runtime.context.workflow_uid}\n'
        f'updated_at: {utc_now().isoformat()}\n'
        '---\n\n'
        '# 工作流修复记录\n\n'
        f'{summary.strip()}\n\n'
        '## 子图来源同步\n\n' + '\n'.join(f'- {line}' for line in sync_lines) + '\n'
    )
    temporary_path.write_text(content, encoding='utf-8')
    return temporary_path, update_path, update_path.relative_to(project_root).as_posix()


@tool('finalize_workflow_fix', parse_docstring=True)
async def finalize_workflow_fix(
    mode: Literal['save', 'no_change'],
    summary: str,
    runtime: ToolRuntime[WorkflowFixContext, WorkflowFixState],
) -> Command:
    """Finish Fix by saving the active workflow or recording a no-change conclusion.

    Args:
        mode: ``save`` for an actual workflow mutation, or ``no_change`` when no safe workflow edit is needed.
        summary: Concrete explanation in the language required by the system prompt of changes, or the reason and
            manual recommendation for no change.
    """
    if not summary.strip():
        return Command(
            update={
                'messages': [
                    ToolMessage(content='A concrete final summary is required.', tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    base_update = {
        'fix_completed': True,
        'workflow_changed': False,
        'update_path': None,
        'updated': True,
    }
    if runtime.state.get('subgraph_stack'):
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='A subgraph is still loaded. Save or cancel each subgraph before finalizing.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    if mode == 'no_change':
        return Command(
            update={
                **base_update,
                'messages': [
                    ToolMessage(
                        content=(
                            'Fix finalized without changing the workflow. In your final response, clearly explain '
                            'the reason and the concrete manual or operational recommendation from the submitted summary.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    nodes = list(runtime.state.get('exist_nodes', []))
    edges = list(runtime.state.get('exist_edges', []))
    if errors := _validation_errors(nodes, edges):
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Cannot save the workflow:\n' + '\n'.join(f'- {error}' for error in errors),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    original = WorkflowDefinition.model_validate(runtime.state['original_workflow'])
    original_to_editing = runtime.state.get('original_node_id_mapping')
    if original_to_editing is None:
        original_to_editing = {node.id: node.id for node in original.nodes}
    saved_nodes, saved_edges, editing_to_saved = _preserve_node_ids(nodes, edges, original_to_editing)
    saved_to_original = compose_node_id_mappings(original_to_editing, editing_to_saved)
    definition = _preserve_canvas_metadata(
        build_workflow_definition(
            saved_nodes,
            saved_edges,
            remap_interface(
                original.interface,
                {
                    original_id: editing_to_saved[editing_id]
                    for original_id, editing_id in original_to_editing.items()
                    if editing_id in editing_to_saved
                },
            ),
        ),
        original,
        saved_to_original,
    )
    if {'subgraph', 'collect_file_collection'} & {node.type for node in definition.nodes}:
        try:
            validate_workflow(definition)
            with Session(engine) as session:
                validate_interface_handles(definition, session)
        except ValueError as error:
            return Command(
                update={
                    'messages': [
                        ToolMessage(content=f'Cannot save the workflow: {error}', tool_call_id=runtime.tool_call_id)
                    ]
                }
            )
    if definition.model_dump(mode='json') == original.model_dump(mode='json'):
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='The current workflow is unchanged. Call this tool again with mode="no_change".',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    temporary_path: Path | None = None
    changed_uids = [runtime.context.workflow_uid]
    sync_lines: list[str] = []
    source_targets: dict[uuid.UUID, WorkflowDefinition] = {}
    baselines = runtime.state.get('subgraph_source_baselines', {})
    original_to_saved = {
        original_id: editing_to_saved[editing_id]
        for original_id, editing_id in original_to_editing.items()
        if editing_id in editing_to_saved
    }

    def embedded(graph: WorkflowDefinition, path: list[str], *, saved: bool) -> WorkflowDefinition | None:
        for index, node_id in enumerate(path):
            actual_id = original_to_saved.get(node_id, node_id) if saved and index == 0 else node_id
            node = next((item for item in graph.nodes if item.id == actual_id and item.type == 'subgraph'), None)
            if node is None:
                return None
            graph = WorkflowDefinition.model_validate(node.data['workflow'])
        return graph

    try:
        with Session(engine) as session, session.begin():
            main = session.get(Workflow, uuid.UUID(runtime.context.workflow_uid), with_for_update=True)
            if main is None or main.owner_id != runtime.context.user_id:
                message = 'The workflow template is no longer editable.'
                raise ValueError(message)
            if main.workflow.model_dump(mode='json') != original.model_dump(mode='json'):
                message = 'The workflow template changed during Fix. Reload and try again.'
                raise ValueError(message)

            for path in runtime.state.get('changed_subgraph_paths', []):
                if not path:
                    continue
                old_child = embedded(original, path, saved=False)
                new_child = embedded(definition, path, saved=True)
                if (
                    old_child is None
                    or new_child is None
                    or old_child.model_dump(mode='json') == new_child.model_dump(mode='json')
                ):
                    continue
                parent = original
                for node_id in path[:-1]:
                    ancestor = next(item for item in parent.nodes if item.id == node_id)
                    parent = WorkflowDefinition.model_validate(ancestor.data['workflow'])
                source_node = next(item for item in parent.nodes if item.id == path[-1])
                source_uid_raw = source_node.data.get('source_uid')
                label = '/'.join(path)
                if not isinstance(source_uid_raw, str):
                    sync_lines.append(f'{label}: no source UID; only the main template was updated.')
                    continue
                try:
                    source_uid = uuid.UUID(source_uid_raw)
                except ValueError:
                    sync_lines.append(f'{label}: invalid source UID; only the main template was updated.')
                    continue
                if source_uid == main.uid:
                    sync_lines.append(
                        f'{label}: source is the current main template; only its embedded copy was updated.'
                    )
                    continue
                source = session.get(Workflow, source_uid, with_for_update=True)
                if source is None or source.wf_type != WorkflowType.SUBMODULE:
                    sync_lines.append(f'{label}: source no longer exists; only the main template was updated.')
                    continue
                if source.owner_id != runtime.context.user_id:
                    sync_lines.append(f'{label}: source is not owned by this user; only the main template was updated.')
                    continue
                baseline = baselines.get(source_uid_raw)
                if baseline is None:
                    sync_lines.append(
                        f'{label}: source was not available when loaded; only the main template was updated.'
                    )
                    continue
                if source.workflow.model_dump(mode='json') != baseline:
                    message = f'Source subgraph {source_uid} changed during Fix. Reload and try again.'
                    raise ValueError(message)
                prior = source_targets.get(source_uid)
                if prior is not None and prior.model_dump(mode='json') != new_child.model_dump(mode='json'):
                    message = f'Source subgraph {source_uid} has conflicting edited instances.'
                    raise ValueError(message)
                source_targets[source_uid] = new_child
                suffix = (
                    ' (replaced a source definition that differed from the embedded copy)'
                    if baseline != old_child.model_dump(mode='json')
                    else ''
                )
                sync_lines.append(f'{label}: synchronized source {source_uid}{suffix}.')
            main.workflow = definition
            for uid, child in source_targets.items():
                source = session.get(Workflow, uid)
                if source is None:
                    message = f'Source subgraph {uid} disappeared during Fix.'
                    raise ValueError(message)
                source.workflow = child
                changed_uids.append(str(uid))
            if not sync_lines:
                sync_lines.append('No edited source subgraphs.')
            temporary_path, final_path, update_path = _stage_update(runtime, summary, sync_lines)
            session.flush()
        temporary_path.replace(final_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return Command(
        update={
            **base_update,
            'workflow_changed': True,
            'changed_workflow_uids': changed_uids,
            'update_path': update_path,
            'messages': [
                ToolMessage(
                    content=(
                        f'The workflow was updated in place and the change record was saved to {update_path}. '
                        f'Subgraph source results: {"; ".join(sync_lines)}. '
                        'Summarize the completed fix and recommended validation in the final response.'
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
