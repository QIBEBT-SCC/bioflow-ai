"""Pure connection rules shared by graph diagnostics and builder adapters."""

from collections import Counter
from collections.abc import Mapping
from typing import Any

from bioflow_ai.schemas.tools import Handle
from bioflow_ai.schemas.workflow_build import BuildEdge, BuildNode

MARK_NODE_TYPES = {'project_mark', 'sample_mark'}


def _validate_note_anchor(anchor_node_id: str | None, exist_nodes: list[BuildNode]) -> str | None:
    """Ensure an optional note anchor refers to an existing non-note node."""
    if anchor_node_id is None:
        return None

    anchor_node = next((node for node in exist_nodes if node.id == anchor_node_id), None)
    if anchor_node is None:
        return f'Anchor node "{anchor_node_id}" does not exist.'
    if anchor_node.type == 'note':
        return 'A note can only be anchored to a non-note node.'
    return None


def _check_edge_handles(
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


def _check_tool_node_edges(
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


def _check_code_node_edges(input_edges: list[BuildEdge], exist_nodes: list[BuildNode]) -> str | None:
    """Code-node-specific edge checks: all inputs must originate from the bind_param node."""
    node_type_map = {node.id: node.type for node in exist_nodes}
    invalid_edges = [e for e in input_edges if node_type_map.get(e.source_node) != 'bind_param']
    if invalid_edges:
        invalid_node_ids = [e.source_node for e in invalid_edges]
        return (
            f'Error: The input of a Bash, Python, or R code node must come from `bind_param` processor nodes. '
            f'The following source nodes are not `bind_param`: {invalid_node_ids}. '
            f'Please refer to the matching CODE-node skill for the correct usage pattern.'
        )

    return None


def _check_processor_edges(
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


def _check_ncbi_genome_edges(
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


def _validate_node_context(
    node_type: str,
    arguments: Mapping[str, Any],
    nodes: list[BuildNode],
    *,
    scope: str,
    purpose: str = 'ordinary',
) -> str | None:
    """Check workflow scope and mark identities against supplied graph values."""
    if purpose == 'foreach' and node_type in {
        'collect_file_collection',
        'sample_mark',
        'project_mark',
    }:
        return f'{node_type} is unavailable inside a ForEach subgraph.'

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
        existing_nodes = nodes
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
