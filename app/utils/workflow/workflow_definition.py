from uuid_extensions import uuid7str

from app.schemas.react_flow import WorkflowDefinition


def update_node_id(workflow: WorkflowDefinition) -> WorkflowDefinition:
    """Refresh node IDs and every reference to them.

    Used to refresh node IDs when instantiating nodes from a template.

    Args:
        workflow: The react-flow object parsed from the template.

    Returns:
        The react-flow object with updated node UIDs.
    """
    id_mapping = {node.id: uuid7str() for node in workflow.nodes}

    for node in workflow.nodes:
        node.id = id_mapping[node.id]

        if node.type == 'note':
            anchor_node_id = node.data.get('anchor_node_id')
            if isinstance(anchor_node_id, str) and anchor_node_id in id_mapping:
                node.data['anchor_node_id'] = id_mapping[anchor_node_id]

    # No need to refresh edge IDs; execution does not use them.
    for edge in workflow.edges:
        old_source = edge.source
        old_target = edge.target
        edge.source = id_mapping.get(old_source, old_source)
        edge.target = id_mapping.get(old_target, old_target)

        if edge.sourceHandle.startswith(f'{old_source}-'):
            edge.sourceHandle = f'{edge.source}{edge.sourceHandle[len(old_source) :]}'
        if edge.targetHandle.startswith(f'{old_target}-'):
            edge.targetHandle = f'{edge.target}{edge.targetHandle[len(old_target) :]}'

    return workflow
