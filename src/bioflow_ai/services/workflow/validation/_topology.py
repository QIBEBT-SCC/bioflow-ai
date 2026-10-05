"""Internal stable topological analysis."""

from collections import defaultdict, deque


def order_nodes(node_ids: list[str], relationships: list[tuple[str, str]]) -> list[str]:
    """Order dependencies stably, deduplicating multiple parameter connections."""
    indegree = dict.fromkeys(node_ids, 0)
    adjacency: dict[str, dict[str, None]] = defaultdict(dict)
    for source, target in relationships:
        if target not in adjacency[source]:
            adjacency[source][target] = None
            indegree[target] += 1
    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    order = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for successor in adjacency[node_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if len(order) != len(node_ids):
        message = 'Workflow graph contains a cycle'
        raise ValueError(message)
    return order
