"""Editing projections for reusable subgraphs and their batch boundaries."""

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
from bioflow_ai.services.workflow.ports import effective_nodes

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


def build_subgraph_definition(
    nodes: list[BuildNode],
    edges: list[BuildEdge],
    outputs: list[dict[str, Any]],
) -> WorkflowDefinition:
    """Replace virtual inputs with a real interface and canonicalize real node IDs."""
    real_nodes = [node for node in nodes if node.id != INPUT_NODE_ID]
    ports = input_ports(nodes)
    boundary_edges = [edge for edge in edges if edge.source_node == INPUT_NODE_ID]
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
    exposed = [
        InterfaceOutput(
            id=output['id'],
            name=output['name'],
            description=output['description'],
            source=PortTarget(node_id=mapping[output['source_node']], handle=output['source_handle']),
        )
        for output in outputs
    ]
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


__all__ = ['effective_nodes']
