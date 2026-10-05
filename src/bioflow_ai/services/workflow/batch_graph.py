"""Transform validated collection-to-subgraph edges into batch control nodes."""

from bioflow_ai.schemas.react_flow import Edge, WorkflowDefinition


def lower_batch_subgraphs(workflow: WorkflowDefinition) -> WorkflowDefinition:
    """Keep saved graphs visual while compiling their validated batch boundaries."""
    graph = workflow.model_copy(deep=True)
    for node in graph.nodes:
        if node.type == 'subgraph':
            node.data['workflow'] = lower_batch_subgraphs(
                WorkflowDefinition.model_validate(node.data['workflow'])
            ).model_dump(mode='json')
    nodes = {node.id: node for node in graph.nodes}
    edges = []
    for edge in graph.edges:
        source = nodes.get(edge.source)
        if source is not None and source.type == 'collect_file_collection':
            target = nodes[edge.target]
            target.type = 'foreach'
            target.data['item_port_id'] = edge.targetHandle.removeprefix(f'{target.id}-in-')
            target.data['subfolder_prefix'] = source.data.get('subfolder_prefix', 'item')
            edges.append(
                Edge(
                    id=edge.id,
                    source=edge.source,
                    sourceHandle=edge.sourceHandle,
                    target=edge.target,
                    targetHandle=f'{target.id}-in-collection',
                )
            )
        else:
            edges.append(edge)
    graph.edges = edges
    return graph
