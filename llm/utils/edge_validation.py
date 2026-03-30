from collections import Counter

from app.schemas.tools import Handle
from llm.states import BuildEdge, BuildNode


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
            f'Error: The input of a bash code node must come from `bind_param` processor nodes. '
            f'The following source nodes are not `bind_param`: {invalid_node_ids}. '
            f'Please refer to the USING_BASH_NODE skill for the correct usage pattern.'
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
