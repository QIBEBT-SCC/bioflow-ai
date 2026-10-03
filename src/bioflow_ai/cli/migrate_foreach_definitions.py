"""One-time conversion of saved embedded ForEach definitions to separate nodes.

Run after the foreach schema migration with the workflow dispatcher and workers stopped.
Historical run snapshots are intentionally left unchanged.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from bioflow_ai.core.session import engine
from bioflow_ai.models.workflow import NodeRun, Workflow, WorkflowRun
from bioflow_ai.schemas.react_flow import Edge, WorkflowDefinition
from bioflow_ai.services.workflow.compiler import validate_workflow


def convert_definition(workflow: WorkflowDefinition) -> tuple[WorkflowDefinition, bool]:
    """Split every embedded ForEach while preserving its outer bindings."""
    graph = workflow.model_copy(deep=True)
    changed = False
    used_ids = {node.id for node in graph.nodes}
    converted: dict[str, tuple[str, str]] = {}
    new_nodes = []
    for node in graph.nodes:
        if node.type == 'subgraph':
            child, child_changed = convert_definition(WorkflowDefinition.model_validate(node.data['workflow']))
            node.data['workflow'] = child.model_dump(mode='json')
            changed |= child_changed
        if node.type != 'foreach':
            new_nodes.append(node)
            continue
        child, _ = convert_definition(WorkflowDefinition.model_validate(node.data['workflow']))
        if child.interface is None or not child.interface.inputs:
            message = f'ForEach {node.id}: child has no public file input'
            raise ValueError(message)
        batch_id = f'{node.id}_batch'
        suffix = 2
        while batch_id in used_ids:
            batch_id = f'{node.id}_batch_{suffix}'
            suffix += 1
        used_ids.add(batch_id)
        converted[node.id] = (batch_id, child.interface.inputs[0].id)
        collection = node.model_copy(deep=True)
        collection.type = 'collect_file_collection'
        collection.data = {
            'pattern': node.data.get('pattern', ''),
            'subfolder_prefix': node.data.get('subfolder_prefix', 'item'),
        }
        batch = node.model_copy(deep=True)
        batch.id = batch_id
        batch.type = 'subgraph'
        batch.data = {
            'name': node.data.get('name', 'ForEach'),
            'description': node.data.get('description', ''),
            'source_uid': node.data.get('source_uid'),
            'workflow': child.model_dump(mode='json'),
        }
        if batch.position is not None:
            batch.position.x += 240
        new_nodes.extend((collection, batch))
        changed = True
    if not converted:
        graph.nodes = new_nodes
        return graph, changed
    edge_ids = {edge.id for edge in graph.edges}
    new_edges = []
    for edge in graph.edges:
        copied = edge.model_copy(deep=True)
        if edge.source in converted:
            batch_id, _ = converted[edge.source]
            copied.source = batch_id
            copied.sourceHandle = f'{batch_id}-out-results_folder'
        if edge.target in converted and edge.targetHandle != f'{edge.target}-in-folder_path':
            batch_id, _ = converted[edge.target]
            copied.target = batch_id
            copied.targetHandle = edge.targetHandle.replace(f'{edge.target}-in-', f'{batch_id}-in-', 1)
        new_edges.append(copied)
    for collection_id, (batch_id, item_port_id) in converted.items():
        edge_id = f'{collection_id}-to-{batch_id}'
        suffix = 2
        while edge_id in edge_ids:
            edge_id = f'{collection_id}-to-{batch_id}-{suffix}'
            suffix += 1
        edge_ids.add(edge_id)
        new_edges.append(
            Edge(
                id=edge_id,
                source=collection_id,
                sourceHandle=f'{collection_id}-out-collection',
                target=batch_id,
                targetHandle=f'{batch_id}-in-{item_port_id}',
            )
        )
    if graph.interface is not None:
        for port in graph.interface.inputs:
            for target in port.targets:
                if target.node_id in converted and target.handle != 'folder_path':
                    target.node_id = converted[target.node_id][0]
        for port in graph.interface.outputs:
            if port.source.node_id in converted:
                port.source.node_id = converted[port.source.node_id][0]
    graph.nodes = new_nodes
    graph.edges = new_edges
    return graph, True


def main() -> None:
    """Convert saved workflow definitions in one transaction."""
    count = 0
    with Session(engine) as session, session.begin():
        unfinished = session.scalar(
            select(NodeRun.uid)
            .join(WorkflowRun, WorkflowRun.uid == NodeRun.run_uid)
            .where(NodeRun.node_type == 'foreach', WorkflowRun.settled.is_(False))
            .limit(1)
        )
        if unfinished is not None:
            message = 'Finish existing embedded ForEach runs before converting workflow definitions'
            raise RuntimeError(message)
        for workflow in session.scalars(select(Workflow)):
            converted, changed = convert_definition(workflow.workflow)
            if changed:
                validate_workflow(converted)
                workflow.workflow = converted
                count += 1
    print(f'Converted {count} saved workflow definitions')


if __name__ == '__main__':
    main()
