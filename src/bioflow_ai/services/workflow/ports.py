"""Pure dynamic-port and editing-boundary checks."""

from bioflow_ai.schemas.react_flow import WorkflowDefinition
from bioflow_ai.schemas.tools import Handle
from bioflow_ai.schemas.workflow_build import BuildEdge, BuildNode


def effective_nodes(nodes: list[BuildNode], edges: list[BuildEdge]) -> list[BuildNode]:
    """Project output handles after a collection connects to a subgraph."""
    by_id = {node.id: node for node in nodes}
    result: list[BuildNode] = []
    for node in nodes:
        if node.type != 'subgraph':
            result.append(node)
            continue
        child = WorkflowDefinition.model_validate(node.data['workflow'])
        if child.interface is None:
            message = f'Subgraph {node.id} has no interface'
            raise ValueError(message)
        batch_edges = [
            edge
            for edge in edges
            if edge.target_node == node.id
            and (source := by_id.get(edge.source_node)) is not None
            and source.type == 'collect_file_collection'
            and edge.source_handle == 'collection'
        ]
        if len(batch_edges) > 1:
            message = f'Subgraph {node.id} has more than one ForEach input'
            raise ValueError(message)
        output_handles = (
            [
                Handle(
                    name='results_folder',
                    description='Folder with manifest.json v1; each item.outputs[child output port ID] is a relative export path',
                )
            ]
            if batch_edges
            else [Handle(name=port.id, description=port.description) for port in child.interface.outputs]
        )
        result.append(node.model_copy(update={'output_handles': output_handles}, deep=True))
    return result
