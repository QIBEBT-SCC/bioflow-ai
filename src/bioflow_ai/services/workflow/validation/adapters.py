"""Adapters preserve editing identities and copy all submitted values."""

from copy import deepcopy
from typing import Any

from bioflow_ai.schemas.workflow_build import BuildEdge, BuildNode


def build_graph(nodes: list[BuildNode], edges: list[BuildEdge]) -> dict[str, Any]:
    """Convert builder values to a graph without assigning or renumbering IDs."""
    return {
        'nodes': [node.model_dump() for node in nodes],
        'edges': [
            {
                'id': f'xy-edge__{edge.source_node}-out-{edge.source_handle}-{edge.target_node}-in-{edge.target_handle}',
                'source': edge.source_node,
                'sourceHandle': f'{edge.source_node}-out-{edge.source_handle}',
                'target': edge.target_node,
                'targetHandle': f'{edge.target_node}-in-{edge.target_handle}',
            }
            for edge in edges
        ],
    }


def bind_candidate_edges(node_id: str, edges: list[BuildEdge]) -> list[BuildEdge]:
    """Prepare incoming edges without changing caller-owned edge objects."""
    return [edge.model_copy(update={'target_node': node_id}, deep=True) for edge in edges]


def copy_graph(graph: Any) -> Any:
    """Model dumping must not mutate the caller's nested data."""
    return deepcopy(graph.model_dump(mode='json') if hasattr(graph, 'model_dump') else graph)


def subgraph_graph(nodes: list[BuildNode], edges: list[BuildEdge], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert virtual sources to raw interface bindings before validation."""
    virtual = next((node for node in nodes if node.type == 'subgraph_inputs'), None)
    source_id = virtual.id if virtual else None
    graph = build_graph(
        [node for node in nodes if node is not virtual], [edge for edge in edges if edge.source_node != source_id]
    )
    graph['interface'] = {
        'inputs': [
            {
                **port,
                'targets': [
                    {'node_id': edge.target_node, 'handle': edge.target_handle}
                    for edge in edges
                    if edge.source_node == source_id and edge.source_handle == port['id']
                ],
            }
            for port in virtual.data.get('ports', [])
        ]
        if virtual
        else [],
        'outputs': [
            {
                'id': output['id'],
                'name': output['name'],
                'description': output.get('description', ''),
                'source': output.get('source') or {'node_id': output['source_node'], 'handle': output['source_handle']},
            }
            for output in outputs
        ],
    }
    return graph
