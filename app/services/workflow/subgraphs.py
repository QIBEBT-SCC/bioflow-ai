"""Validation, expansion and presentation helpers for snapshot subgraphs."""

from copy import deepcopy
from typing import Any
from urllib.parse import quote

from sqlalchemy.orm import Session

from app.crud.tools import tool as tool_crud
from app.schemas.react_flow import Edge, Node, WorkflowDefinition
from app.schemas.tools import ToolNodePublic
from llm.utils.node_schema import PROCESSORS, RESOURCES

MAX_SUBGRAPH_DEPTH = 32


class SubgraphError(ValueError):
    """A subgraph interface cannot be resolved."""


def definition_id(path: tuple[str, ...]) -> str:
    """Keep legacy root IDs; namespace descendants without delimiter collisions."""
    return path[0] if len(path) == 1 else '@subgraph/' + '/'.join(quote(part, safe='') for part in path)


def _fail(message: str) -> None:
    raise SubgraphError(message)


def flatten_workflow(workflow: WorkflowDefinition, *, require_inputs: bool = False) -> WorkflowDefinition:
    """Resolve boundary bindings recursively without mutating the saved definition."""
    flat_nodes: list[Node] = []
    flat_edges: list[Edge] = []
    ids: set[str] = set()
    required_targets: list[tuple[str, list[tuple[str, str]]]] = []

    def visit(graph: WorkflowDefinition, path: tuple[str, ...]) -> tuple[dict, dict]:
        if len(path) > MAX_SUBGRAPH_DEPTH:
            _fail('Subgraph nesting exceeds 32 levels')
        nodes = {node.id: node for node in graph.nodes}
        if len(nodes) != len(graph.nodes):
            _fail(f'{path}: duplicate node IDs')
        child_ports = {}
        for node in graph.nodes:
            node_path = (*path, node.id)
            if node.type == 'subgraph':
                child = WorkflowDefinition.model_validate(node.data.get('workflow'))
                if child.interface is None:
                    _fail(f'{node_path}: subgraph has no interface definition')
                child_ports[node.id] = visit(child, node_path)
            else:
                uid = definition_id(node_path)
                if uid in ids:
                    _fail(f'Expanded node ID collision: {uid}')
                ids.add(uid)
                copied = node.model_copy(deep=True)
                copied.id = uid
                copied.data.pop('run_data', None)
                flat_nodes.append(copied)

        def resolve(node_id: str, handle: str, direction: str) -> list[tuple[str, str]]:
            if node_id not in nodes or not handle:
                _fail(f'{path}: missing {direction} mapping {node_id}.{handle}')
            if nodes[node_id].type == 'subgraph':
                ports = child_ports[node_id][0 if direction == 'input' else 1]
                if handle not in ports:
                    _fail(f'{path}: unknown {direction} port {node_id}.{handle}')
                return ports[handle]
            if nodes[node_id].type == 'note':
                _fail(f'{path}: notes cannot carry interface bindings')
            return [(definition_id((*path, node_id)), handle)]

        for edge in graph.edges:
            source_prefix, target_prefix = f'{edge.source}-out-', f'{edge.target}-in-'
            if not edge.sourceHandle.startswith(source_prefix) or not edge.targetHandle.startswith(target_prefix):
                _fail(f'{path}: invalid handles on edge {edge.id}')
            sources = resolve(edge.source, edge.sourceHandle.removeprefix(source_prefix), 'output')
            targets = resolve(edge.target, edge.targetHandle.removeprefix(target_prefix), 'input')
            for source, source_key in sources:
                for target, target_key in targets:
                    flat_edges.append(
                        Edge(
                            id=f'expanded-{len(flat_edges)}',
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
                required_targets.append((f'{"/".join(path) or "root"}.{port.name}', targets))
            for port in graph.interface.outputs:
                outputs[port.id] = resolve(port.source.node_id, port.source.handle, 'output')
        return inputs, outputs

    visit(workflow, ())
    if require_inputs:
        bound = {(edge.target, edge.targetHandle.removeprefix(f'{edge.target}-in-')) for edge in flat_edges}
        for name, targets in required_targets:
            if any(target not in bound for target in targets):
                _fail(f'Missing subgraph input: {name}')
    return WorkflowDefinition(nodes=flat_nodes, edges=flat_edges)


def aggregate_status(statuses: list[str]) -> str:
    """Aggregate executable descendants; failure remains visible during other work."""
    for status in ('failed', 'blocked'):
        if status in statuses:
            return status
    if not statuses or all(status == 'succeeded' for status in statuses):
        return 'succeeded'
    if 'running' in statuses or 'succeeded' in statuses:
        return 'running'
    return next((status for status in ('queued', 'ready', 'pending') if status in statuses), 'pending')


def inject_run_data(workflow: WorkflowDefinition, by_definition: dict[str, dict[str, Any]]) -> list[dict]:
    """Decorate the immutable run snapshot, including all nested descendants."""

    def visit(graph: dict, path: tuple[str, ...]) -> list[str]:
        statuses = []
        for node in graph['nodes']:
            node_path = (*path, node['id'])
            node['data'].pop('run_data', None)
            if node['type'] == 'subgraph' and definition_id(node_path) not in by_definition:
                descendants = visit(node['data']['workflow'], node_path)
                node['data']['run_data'] = {'status': aggregate_status(descendants)}
                statuses.extend(descendants)
            elif data := by_definition.get(definition_id(node_path)):
                node['data']['run_data'] = data
                statuses.append(data['status'])
        return statuses

    graph = deepcopy(workflow.model_dump(mode='json'))
    visit(graph, ())
    return graph['nodes']


def validate_interface_handles(graph: WorkflowDefinition, session: Session) -> None:
    """Reject interfaces referencing removed tool ports or invalid built-in handles."""
    nodes = {node.id: node for node in graph.nodes}
    if graph.interface:
        bindings = [(target, 'input') for port in graph.interface.inputs for target in port.targets]
        bindings.extend((port.source, 'output') for port in graph.interface.outputs)
        for target, direction in bindings:
            node = nodes.get(target.node_id)
            if node is None or node.type == 'subgraph':
                continue  # Structural validation resolves subgraph ports and missing nodes.
            names: set[str] = set()
            if node.type == 'tool':
                tool = tool_crud.get(session, node.data.get('tool_uid', ''))
                if tool is not None:
                    public = ToolNodePublic.model_validate(tool)
                    names = {handle.name for handle in getattr(public, f'{direction}_handles')}
            elif node.type == 'foreach':
                child = WorkflowDefinition.model_validate(node.data['workflow'])
                names = (
                    {'collection'}
                    | {port.id for port in child.interface.inputs if port.id != node.data['item_port_id']}
                    if direction == 'input' and child.interface is not None
                    else {'results_folder'}
                    if direction == 'output'
                    else set()
                )
            elif node.type in {'code_R', 'code_python', 'code_bash', 'downstream_summary'}:
                names = (
                    {'input_files'}
                    if direction == 'input'
                    else ({'output_folder'} if node.type != 'downstream_summary' else set())
                )
            else:
                schema = PROCESSORS.get(node.type) or RESOURCES.get(node.type.removeprefix('resource_'))
                if schema:
                    names = {handle.name for handle in schema[f'{direction}_handles']}
            if target.handle not in names:
                message = f'Node {node.id}: unknown exposed {direction} handle {target.handle!r}'
                raise SubgraphError(message)
    for node in graph.nodes:
        if node.type in {'subgraph', 'foreach'}:
            validate_interface_handles(WorkflowDefinition.model_validate(node.data['workflow']), session)
