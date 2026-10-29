"""Node-ID allocation and canonicalization for workflow editing sessions."""

import re
from collections.abc import Mapping, Sequence

from bioflow_ai.llm.states import BuildEdge, BuildNode
from bioflow_ai.utils.uid_mapper import generate_char_index

LETTER_NODE_ID = re.compile(r'^[A-Z]+$')


def node_id_sort_key(node_id: str) -> tuple[int, str]:
    """Sort Excel-style letter IDs as A..Z, AA..ZZ, and so on."""
    return len(node_id), node_id


def has_canonical_node_ids(nodes: Sequence[BuildNode]) -> bool:
    """Return whether node IDs are unique and exactly cover A through the node count."""
    node_ids = [node.id for node in nodes]
    if len(node_ids) != len(set(node_ids)) or any(LETTER_NODE_ID.fullmatch(node_id) is None for node_id in node_ids):
        return False
    expected = [generate_char_index(index) for index in range(1, len(nodes) + 1)]
    return sorted(node_ids, key=node_id_sort_key) == expected


def allocate_node_id(nodes: Sequence[BuildNode], next_index: int | None) -> tuple[str, int]:
    """Allocate one session-local ID without reusing IDs after deletions.

    ``next_index`` is monotonically advanced by callers. Existing IDs are still
    checked so old or resumed drafts with sparse IDs cannot collide.
    """
    index = max(next_index or 1, len(nodes) + 1)
    occupied = {node.id for node in nodes}
    candidate = generate_char_index(index)
    while candidate in occupied:
        index += 1
        candidate = generate_char_index(index)
    return candidate, index + 1


def canonicalize_build_graph(
    nodes: Sequence[BuildNode],
    edges: Sequence[BuildEdge],
) -> tuple[list[BuildNode], list[BuildEdge], dict[str, str]]:
    """Return a deep-copied graph whose node IDs exactly cover A through N.

    The returned mapping contains every old ID, including identity mappings.
    Node list order is preserved; only IDs and references are rewritten.
    """
    node_ids = [node.id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        msg = 'Cannot canonicalize a workflow with duplicate node IDs.'
        raise ValueError(msg)

    ordered_ids = sorted(node_ids, key=node_id_sort_key)
    id_mapping = {old_id: generate_char_index(index) for index, old_id in enumerate(ordered_ids, start=1)}

    canonical_nodes = [node.model_copy(deep=True) for node in nodes]
    for node in canonical_nodes:
        old_id = node.id
        node.id = id_mapping[old_id]
        if node.type == 'note':
            anchor_node_id = node.data.get('anchor_node_id')
            if isinstance(anchor_node_id, str) and anchor_node_id in id_mapping:
                node.data['anchor_node_id'] = id_mapping[anchor_node_id]

    canonical_edges = [edge.model_copy(deep=True) for edge in edges]
    for edge in canonical_edges:
        edge.source_node = id_mapping.get(edge.source_node, edge.source_node)
        if edge.target_node is not None:
            edge.target_node = id_mapping.get(edge.target_node, edge.target_node)

    return canonical_nodes, canonical_edges, id_mapping


def remove_node_from_build_graph(
    nodes: Sequence[BuildNode],
    edges: Sequence[BuildEdge],
    node_id: str,
) -> tuple[list[BuildNode], list[BuildEdge], list[BuildEdge]]:
    """Remove a node and its references without changing any surviving node ID."""
    removed_edges = [edge.model_copy(deep=True) for edge in edges if node_id in (edge.source_node, edge.target_node)]
    remaining_edges = [
        edge.model_copy(deep=True) for edge in edges if node_id not in (edge.source_node, edge.target_node)
    ]
    remaining_nodes = [node.model_copy(deep=True) for node in nodes if node.id != node_id]
    for node in remaining_nodes:
        if node.type == 'note' and node.data.get('anchor_node_id') == node_id:
            node.data['anchor_node_id'] = None
    return remaining_nodes, remaining_edges, removed_edges


def compose_node_id_mappings(
    original_to_editing: Mapping[str, str],
    editing_to_saved: Mapping[str, str],
) -> dict[str, str]:
    """Compose load-time and save-time mappings as saved ID to original ID."""
    return {
        editing_to_saved[editing_id]: original_id
        for original_id, editing_id in original_to_editing.items()
        if editing_id in editing_to_saved
    }
