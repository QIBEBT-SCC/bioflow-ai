"""Lower a collection-to-subgraph edge into one durable batch control node."""

from app.schemas.react_flow import Edge, WorkflowDefinition


def lower_batch_subgraphs(workflow: WorkflowDefinition) -> WorkflowDefinition:
    """Keep the saved graph visual while compiling its batch boundaries."""
    graph = workflow.model_copy(deep=True)
    for node in graph.nodes:
        if node.type == 'foreach':
            message = f'Node {node.id}: embedded ForEach must be converted to a collection and subgraph'
            raise ValueError(message)
        if node.type == 'subgraph':
            node.data['workflow'] = lower_batch_subgraphs(
                WorkflowDefinition.model_validate(node.data['workflow'])
            ).model_dump(mode='json')

    nodes = {node.id: node for node in graph.nodes}
    consumed: set[str] = set()
    replacement_edges: list[Edge] = []
    for edge in graph.edges:
        source = nodes.get(edge.source)
        if source is None or source.type != 'collect_file_collection':
            replacement_edges.append(edge)
            continue
        if edge.sourceHandle != f'{source.id}-out-collection':
            message = f'ForEach {source.id}: invalid collection output'
            raise ValueError(message)
        target = nodes.get(edge.target)
        if target is None or target.type != 'subgraph':
            message = f'ForEach {source.id}: its collection must connect to a subgraph input'
            raise ValueError(message)
        child = WorkflowDefinition.model_validate(target.data['workflow'])
        port_id = edge.targetHandle.removeprefix(f'{target.id}-in-')
        if child.interface is None or port_id not in {port.id for port in child.interface.inputs}:
            message = f'ForEach {source.id}: invalid subgraph input {port_id!r}'
            raise ValueError(message)
        if any(port.id == 'collection' and port.id != port_id for port in child.interface.inputs):
            message = f'Subgraph {target.id}: collection is reserved for the ForEach input'
            raise ValueError(message)
        if target.id in consumed:
            message = f'Subgraph {target.id}: only one ForEach input is supported'
            raise ValueError(message)
        consumed.add(target.id)
        target.type = 'foreach'
        target.data['item_port_id'] = port_id
        target.data['subfolder_prefix'] = source.data.get('subfolder_prefix', '')
        replacement_edges.append(
            Edge(
                id=edge.id,
                source=edge.source,
                sourceHandle=edge.sourceHandle,
                target=edge.target,
                targetHandle=f'{target.id}-in-collection',
            )
        )
    for node in graph.nodes:
        if node.type == 'collect_file_collection':
            outgoing = [edge for edge in graph.edges if edge.source == node.id]
            if len(outgoing) != 1:
                message = f'ForEach {node.id}: connect its output to exactly one subgraph input'
                raise ValueError(message)
    for edge in replacement_edges:
        source = nodes.get(edge.source)
        if source is not None and source.type == 'foreach' and edge.sourceHandle != f'{source.id}-out-results_folder':
            message = f'Batch subgraph {source.id}: use its results_folder output'
            raise ValueError(message)
        target = nodes.get(edge.target)
        if target is not None and target.type == 'foreach' and edge.targetHandle != f'{target.id}-in-collection':
            child = WorkflowDefinition.model_validate(target.data['workflow'])
            shared = {port.id for port in child.interface.inputs if port.id != target.data['item_port_id']}
            handle = edge.targetHandle.removeprefix(f'{target.id}-in-')
            if not edge.targetHandle.startswith(f'{target.id}-in-') or handle not in shared:
                message = f'Batch subgraph {target.id}: invalid shared input {handle!r}'
                raise ValueError(message)
    if graph.interface is not None:
        for port in graph.interface.outputs:
            if (
                nodes.get(port.source.node_id) is not None
                and nodes[port.source.node_id].type == 'collect_file_collection'
            ):
                message = f'ForEach {port.source.node_id}: collection output must connect to a subgraph input'
                raise ValueError(message)
            if (
                nodes.get(port.source.node_id) is not None
                and nodes[port.source.node_id].type == 'foreach'
                and port.source.handle != 'results_folder'
            ):
                message = f'Batch subgraph {port.source.node_id}: expose results_folder instead'
                raise ValueError(message)
    graph.edges = replacement_edges
    return graph
