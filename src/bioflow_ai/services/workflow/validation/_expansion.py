"""Internal boundary expansion and structural rule support."""

from bioflow_ai.schemas.react_flow import Edge, Node, WorkflowDefinition
from bioflow_ai.services.workflow.subgraphs import MAX_SUBGRAPH_DEPTH, SubgraphError, definition_id
from bioflow_ai.services.workflow.validation.types import SourceLocation


def _fail(message: str) -> None:
    raise SubgraphError(message)


def _flatten_workflow(
    workflow: WorkflowDefinition,
    *,
    source_map: dict[str, SourceLocation] | None = None,
) -> WorkflowDefinition:
    """Resolve boundary bindings recursively without mutating the saved definition."""
    flat_nodes: list[Node] = []
    flat_edges: list[Edge] = []
    ids: set[str] = set()

    def visit(graph: WorkflowDefinition, path: tuple[str, ...], json_path: str = '') -> tuple[dict, dict]:
        if len(path) > MAX_SUBGRAPH_DEPTH:
            _fail('Subgraph nesting exceeds 32 levels')
        nodes = {node.id: node for node in graph.nodes}
        if len(nodes) != len(graph.nodes):
            _fail(f'{path}: duplicate node IDs')
        child_ports = {}
        for node_index, node in enumerate(graph.nodes):
            node_path = (*path, node.id)
            location = f'{json_path}/nodes/{node_index}'
            if source_map is not None:
                source_map[f'node:{definition_id(node_path)}'] = SourceLocation(
                    path=location, node_id=node.id, subgraph_path=list(path)
                )
            if node.type == 'subgraph':
                child = WorkflowDefinition.model_validate(node.data.get('workflow'))
                if child.interface is None:
                    _fail(f'{node_path}: subgraph has no interface definition')
                child_ports[node.id] = visit(child, node_path, location + '/data/workflow')
            else:
                uid = definition_id(node_path)
                if uid in ids:
                    _fail(f'Expanded node ID collision: {uid}')
                ids.add(uid)
                copied = node.model_copy(deep=True)
                copied.id = uid
                copied.data.pop('run_data', None)
                copied.data.pop('preview_data', None)
                flat_nodes.append(copied)

        def resolve(node_id: str, handle: str, direction: str) -> list[tuple[str, str]]:
            if node_id not in nodes or not handle:
                _fail(f'{path}: missing {direction} mapping {node_id}.{handle}')
            if nodes[node_id].type == 'subgraph':
                ports = child_ports[node_id][0 if direction == 'input' else 1]
                if handle not in ports:
                    _fail(f'{path}: unknown {direction} port {node_id}.{handle}')
                return ports[handle]
            if nodes[node_id].type in {'note', 'image_preview'} and direction == 'output':
                _fail(f'{path}: presentation nodes cannot be interface outputs')
            if nodes[node_id].type == 'note':
                _fail(f'{path}: notes cannot carry interface bindings')
            return [(definition_id((*path, node_id)), handle)]

        for edge_index, edge in enumerate(graph.edges):
            source_prefix, target_prefix = f'{edge.source}-out-', f'{edge.target}-in-'
            if not edge.sourceHandle.startswith(source_prefix) or not edge.targetHandle.startswith(target_prefix):
                _fail(f'{path}: invalid handles on edge {edge.id}')
            sources = resolve(edge.source, edge.sourceHandle.removeprefix(source_prefix), 'output')
            targets = resolve(edge.target, edge.targetHandle.removeprefix(target_prefix), 'input')
            for source, source_key in sources:
                for target, target_key in targets:
                    expanded_id = f'expanded-{len(flat_edges)}'
                    if source_map is not None:
                        source_map[f'edge:{expanded_id}'] = SourceLocation(
                            path=f'{json_path}/edges/{edge_index}', edge_id=edge.id, subgraph_path=list(path)
                        )
                    flat_edges.append(
                        Edge(
                            id=expanded_id,
                            source=source,
                            sourceHandle=f'{source}-out-{source_key}',
                            target=target,
                            targetHandle=f'{target}-in-{target_key}',
                        )
                    )
        inputs, outputs = {}, {}
        if graph.interface is not None:
            for port in graph.interface.inputs:
                targets = [
                    binding for target in port.targets for binding in resolve(target.node_id, target.handle, 'input')
                ]
                if len(set(targets)) != len(targets):
                    _fail(f'{path}: duplicate targets on input {port.name}')
                inputs[port.id] = targets
            for port in graph.interface.outputs:
                outputs[port.id] = resolve(port.source.node_id, port.source.handle, 'output')
        return inputs, outputs

    visit(workflow, ())
    return WorkflowDefinition(nodes=flat_nodes, edges=flat_edges)
