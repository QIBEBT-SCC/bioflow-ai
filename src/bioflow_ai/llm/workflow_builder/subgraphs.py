"""Editing projections for reusable subgraphs and their batch boundaries."""

from collections import Counter
from typing import Any

from bioflow_ai.llm.states import BuildEdge, BuildNode
from bioflow_ai.llm.workflow_builder.node_ids import canonicalize_build_graph
from bioflow_ai.llm.workflow_builder.serialization import build_workflow_definition
from bioflow_ai.schemas.react_flow import (
    InterfaceInput,
    InterfaceOutput,
    PortTarget,
    WorkflowDefinition,
    WorkflowInterface,
)
from bioflow_ai.schemas.tools import Handle

INPUT_NODE_ID = '@subgraph_inputs'


def input_source(ports: list[dict[str, str]]) -> BuildNode:
    """Represent declared interface inputs as one editing-only upstream node."""
    return BuildNode(
        id=INPUT_NODE_ID,
        name='Subgraph Inputs',
        type='subgraph_inputs',
        input_handles=[],
        output_handles=[Handle(name=port['id'], description=port['description']) for port in ports],
        data={'ports': ports},
    )


def input_ports(nodes: list[BuildNode]) -> list[dict[str, str]]:
    """Return the current stable input declarations from the virtual source."""
    node = next((item for item in nodes if item.id == INPUT_NODE_ID), None)
    return list(node.data.get('ports', [])) if node is not None else []


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


def validate_editing_edges(nodes: list[BuildNode], edges: list[BuildEdge]) -> list[str]:
    """Report invalid references and handle use in the complete proposed graph."""
    errors: list[str] = []
    try:
        by_id = {node.id: node for node in effective_nodes(nodes, edges)}
    except ValueError as error:
        return [str(error)]
    if len(by_id) != len(nodes):
        errors.append('Duplicate node IDs')
    bindings: Counter[tuple[str, str]] = Counter()
    collection_outputs: Counter[str] = Counter()
    for edge in edges:
        source, target = by_id.get(edge.source_node), by_id.get(edge.target_node or '')
        if source is None or target is None:
            errors.append(f'{edge.source_node} -> {edge.target_node}: missing node')
            continue
        if edge.source_handle not in {handle.name for handle in source.output_handles}:
            errors.append(f'{source.id}: invalid output {edge.source_handle!r}')
        if edge.target_handle not in {handle.name for handle in target.input_handles}:
            errors.append(f'{target.id}: invalid input {edge.target_handle!r}')
        if source.type == 'collect_file_collection' and (
            edge.source_handle != 'collection' or target.type != 'subgraph'
        ):
            errors.append(f'{source.id}.collection must connect to a subgraph input')
        if source.type == 'collect_file_collection' and edge.source_handle == 'collection':
            collection_outputs[source.id] += 1
        bindings[(target.id, edge.target_handle)] += 1
    for (target_id, handle), count in bindings.items():
        if count > 1 and by_id[target_id].type in {'tool', 'subgraph'}:
            errors.append(f'{target_id}.{handle}: more than one input edge')
    for node_id, count in collection_outputs.items():
        if count > 1:
            errors.append(f'{node_id}.collection: connect to exactly one subgraph input')
    return errors


def build_subgraph_definition(
    nodes: list[BuildNode],
    edges: list[BuildEdge],
    outputs: list[dict[str, Any]],
    *,
    foreach: bool,
) -> WorkflowDefinition:
    """Replace virtual inputs with a real interface and canonicalize real node IDs."""
    real_nodes = [node for node in nodes if node.id != INPUT_NODE_ID]
    if not real_nodes:
        message = 'Subgraph needs at least one internal node'
        raise ValueError(message)
    problems = validate_editing_edges(nodes, edges)
    if problems:
        raise ValueError('; '.join(problems))

    ports = input_ports(nodes)
    declared = {port['id'] for port in ports}
    boundary_edges = [edge for edge in edges if edge.source_node == INPUT_NODE_ID]
    if any(edge.source_handle not in declared for edge in boundary_edges):
        message = 'Subgraph input edge refers to an undeclared port'
        raise ValueError(message)
    unused = declared - {edge.source_handle for edge in boundary_edges}
    if unused:
        message = f'Unused subgraph inputs: {sorted(unused)}'
        raise ValueError(message)
    if foreach and not ports:
        message = 'ForEach subgraph needs a current-file input'
        raise ValueError(message)
    if foreach and not outputs:
        message = 'ForEach subgraph needs at least one output'
        raise ValueError(message)

    by_id = {node.id: node for node in real_nodes}
    for node in real_nodes:
        incoming = [edge for edge in edges if edge.target_node == node.id]
        missing = {handle.name for handle in node.input_handles} - {edge.target_handle for edge in incoming}
        if missing:
            message = f'{node.id}: missing {node.type} inputs {sorted(missing)}'
            raise ValueError(message)
        if node.type in {'code_bash', 'code_python', 'code_R'}:
            invalid_sources = [
                edge.source_node
                for edge in incoming
                if getattr(by_id.get(edge.source_node), 'type', None) != 'bind_param'
            ]
            if invalid_sources:
                message = f'{node.id}: CODE inputs must come from bind_param nodes, not {invalid_sources}'
                raise ValueError(message)

    internal_edges = [edge for edge in edges if edge.source_node != INPUT_NODE_ID]
    canonical_nodes, canonical_edges, mapping = canonicalize_build_graph(real_nodes, internal_edges)
    inputs = [
        InterfaceInput(
            id=port['id'],
            name=port['name'],
            description=port['description'],
            targets=[
                PortTarget(node_id=mapping[edge.target_node], handle=edge.target_handle)
                for edge in boundary_edges
                if edge.source_handle == port['id'] and edge.target_node in mapping
            ],
        )
        for port in ports
    ]
    exposed = []
    for output in outputs:
        source = next((node for node in real_nodes if node.id == output['source_node']), None)
        if source is None or output['source_handle'] not in {handle.name for handle in source.output_handles}:
            message = f'Invalid subgraph output source {output["source_node"]}.{output["source_handle"]}'
            raise ValueError(message)
        exposed.append(
            InterfaceOutput(
                id=output['id'],
                name=output['name'],
                description=output['description'],
                source=PortTarget(node_id=mapping[source.id], handle=output['source_handle']),
            )
        )
    interface = WorkflowInterface(inputs=inputs, outputs=exposed)
    return build_workflow_definition(canonical_nodes, canonical_edges, interface)


def remap_interface(interface: WorkflowInterface | None, mapping: dict[str, str]) -> WorkflowInterface | None:
    """Keep boundary bindings aligned when the editor renumbers graph nodes."""
    if interface is None:
        return None
    result = interface.model_copy(deep=True)
    for port in result.inputs:
        for target in port.targets:
            target.node_id = mapping.get(target.node_id, target.node_id)
    for port in result.outputs:
        port.source.node_id = mapping.get(port.source.node_id, port.source.node_id)
    return result
