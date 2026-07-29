"""Context-aware validation performed before adding a node to a workflow draft."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from app.core.session import engine
from app.crud import project_file_mapping as project_file_mapping_crud
from app.crud.resource import sample as sample_crud
from llm.workflow_builder.drafts import get_workflow_drafts

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langgraph.prebuilt import ToolRuntime

    from app.schemas.tools import Handle
    from llm.context import ProjectContext
    from llm.states import BuildEdge, BuildNode

MARK_NODE_TYPES = {'project_mark', 'sample_mark'}


def validate_node_addition(
    state: Mapping[str, Any],
    *,
    node_type: str,
    arguments: Mapping[str, Any],
) -> str | None:
    """Return an actionable error when a node is invalid for the active draft."""
    active_workflow_id = state.get('active_workflow_id')
    drafts = get_workflow_drafts(state)
    draft = drafts.get(active_workflow_id) if active_workflow_id else None
    if draft is None:
        return 'Create or select a workflow draft before adding nodes.'

    scope = draft.execution_scope
    if scope == 'sample_level' and node_type == 'project_mark':
        return 'Cannot add `project_mark` to a sample-level workflow. Use `sample_mark` instead.'
    if scope == 'sample_level' and node_type == 'resource_sample_mark_collection':
        return (
            'Cannot add `sample_mark_collection` to a sample-level workflow. '
            'Create a project-level aggregation workflow for cross-sample collection.'
        )
    if scope == 'project_level' and node_type == 'sample_mark':
        return 'Cannot add `sample_mark` to a project-level workflow. Use `project_mark` instead.'

    if node_type in MARK_NODE_TYPES:
        mark_name = str(arguments.get('mark_name', ''))
        existing_nodes: list[BuildNode] = state.get('exist_nodes', [])
        conflict = next(
            (
                node
                for node in existing_nodes
                if node.type in MARK_NODE_TYPES and str(node.data.get('mark_name', '')) == mark_name
            ),
            None,
        )
        if conflict is not None:
            return (
                f'Cannot add duplicate mark `{mark_name}`. Node {conflict.id} '
                f'(`{conflict.type}`) already uses this name in the active workflow.'
            )

    if (
        scope == 'project_level'
        and node_type == 'resource_file'
        and str(arguments.get('file_path', '')).startswith('sample:')
    ):
        return (
            'Cannot add a sample-bound `sample:` file reference to a project-level workflow. '
            'Use `sample_mark_collection` for cross-sample aggregation.'
        )

    return None


def validate_note_anchor(anchor_node_id: str | None, exist_nodes: list[BuildNode]) -> str | None:
    """Ensure an optional note anchor refers to an existing non-note node."""
    if anchor_node_id is None:
        return None

    anchor_node = next((node for node in exist_nodes if node.id == anchor_node_id), None)
    if anchor_node is None:
        return f'Anchor node "{anchor_node_id}" does not exist.'
    if anchor_node.type == 'note':
        return 'A note can only be anchored to a non-note node.'
    return None


def check_edge_handles(
    input_edges: list[BuildEdge],
    new_node_input_handles: list[Handle],
    exist_nodes: list[BuildNode],
) -> str | None:
    """Verify that source/target handles referenced by edges actually exist.

    Returns an error message string if validation fails, None if valid.
    """
    node_handle_map = {node.id: {h.name for h in node.output_handles} for node in exist_nodes}
    target_handle_names = {h.name for h in new_node_input_handles}

    for edge in input_edges:
        src_handles = node_handle_map.get(edge.source_node)
        if src_handles is None:
            return f"Source node '{edge.source_node}' does not exist on the panel."
        if edge.source_handle not in src_handles:
            return (
                f"Source node '{edge.source_node}' has no output handle named '{edge.source_handle}'. "
                f'Available handles: {sorted(src_handles)}.'
            )
        if edge.target_handle not in target_handle_names:
            return (
                f"This node has no input handle named '{edge.target_handle}'. "
                f'Available handles: {sorted(target_handle_names)}.'
            )

    return None


def check_tool_node_edges(
    input_edges: list[BuildEdge],
    new_node_input_handles: list[Handle],
) -> str | None:
    """Tool-node-specific edge checks: one edge per handle, all handles connected.

    Returns an error message string if validation fails, None if valid.
    """
    counts = Counter(e.target_handle for e in input_edges)
    duplicates = [h for h, c in counts.items() if c > 1]
    if duplicates:
        return f'Each input handle allows only one edge. Handles with multiple edges: {duplicates}.'

    connected = {e.target_handle for e in input_edges}
    missing = [h.name for h in new_node_input_handles if h.name not in connected]
    if missing:
        return f'All input handles must be connected. Missing edges for: {missing}.'

    return None


def check_code_node_edges(input_edges: list[BuildEdge], exist_nodes: list[BuildNode]) -> str | None:
    """Code-node-specific edge checks: all inputs must originate from the bind_param node."""
    node_type_map = {node.id: node.type for node in exist_nodes}
    invalid_edges = [e for e in input_edges if node_type_map.get(e.source_node) != 'bind_param']
    if invalid_edges:
        invalid_node_ids = [e.source_node for e in invalid_edges]
        return (
            f'Error: The input of a Bash or Python code node must come from `bind_param` processor nodes. '
            f'The following source nodes are not `bind_param`: {invalid_node_ids}. '
            f'Please refer to the using-bash-node or using-python-node skill for the correct usage pattern.'
        )

    return None


def check_processor_edges(
    input_edges: list[BuildEdge],
    new_node_input_handles: list[Handle],
) -> str | None:
    """Processor-node-specific edge checks: all handles must have at least one edge.

    Returns an error message string if validation fails, None if valid.
    """
    connected = {e.target_handle for e in input_edges}
    missing = [h.name for h in new_node_input_handles if h.name not in connected]
    if missing:
        return f'All processor input handles must be connected. Missing edges for: {missing}.'

    return None


def check_orphan_edges(exist_edges: list[BuildEdge]) -> list[BuildEdge]:
    """Return edges whose target_node is None (orphan edges lingering in state)."""
    return [e for e in exist_edges if e.target_node is None]


def check_ncbi_genome_edges(
    input_edges: list[BuildEdge],
    new_node_input_handles: list[Handle],
) -> str | None:
    """NCBI genome node edge checks: at least one of the two input handles must be connected.

    Returns an error message string if validation fails, None if valid.
    """
    if not input_edges:
        handle_names = [h.name for h in new_node_input_handles]
        return f'NCBI genome node requires at least one input. Available handles: {handle_names}.'

    return None


def validate_mark_storage_conflict(
    processor_name: str,
    arguments: dict[str, Any],
    runtime: ToolRuntime[ProjectContext],
) -> str | None:
    """Reject mark names that are already occupied in the project file system."""
    if processor_name not in {'project_mark', 'sample_mark'}:
        return None

    mark_name = str(arguments['mark_name'])
    with Session(engine) as session:
        if processor_name == 'project_mark':
            mapping = project_file_mapping_crud.get_by_keyword(
                session,
                runtime.context.project_id,
                mark_name,
            )
            if mapping is None:
                return None
            source = 'dynamic mark' if mapping.is_dynamic else 'initial project resource'
            return f'Cannot add mark `{mark_name}` because `proj:{mark_name}` already exists as a {source}.'

        matches = sample_crud.get_files_by_tag_in_project(
            session,
            runtime.context.project_id,
            mark_name,
        )
        if not matches:
            return None
        sample_names = ', '.join(sample_name for _, sample_name in matches)
        return (
            f'Cannot add mark `{mark_name}` because `sample:{mark_name}` already exists '
            f'on project sample(s): {sample_names}.'
        )
