"""Serialization helpers for workflow-builder state."""

from app.models.workflow import ExecutionScope
from app.schemas.react_flow import Edge, Node, Position, WorkflowDefinition
from llm.states import BuildEdge, BuildNode
from llm.workflow_builder.drafts import WorkflowScope


def to_execution_scope(scope: WorkflowScope) -> ExecutionScope:
    """Convert the LLM-facing scope string to the database enum."""
    return ExecutionScope.PROJECT_LEVEL if scope == 'project_level' else ExecutionScope.SAMPLE_LEVEL


def format_execution_scope(scope: ExecutionScope | int) -> str:
    """Format a database workflow scope that may round-trip as an int."""
    try:
        return ExecutionScope(scope).name
    except ValueError:
        return f'UNKNOWN({scope})'


def build_workflow_definition(exist_nodes: list[BuildNode], exist_edges: list[BuildEdge]) -> WorkflowDefinition:
    """Convert builder nodes and edges into the ReactFlow workflow definition."""
    nodes = []
    for n in exist_nodes:
        if n.type in {'resource_ncbi_genome', 'resource_GRCh38', 'resource_GRCm39'}:
            required_index = list(dict.fromkeys(e.source_handle for e in exist_edges if e.source_node == n.id))
            node = Node(
                id=n.id,
                type=n.type,
                data={'required_index': required_index},
                position=Position(x=n.x * 500, y=n.y * 400),
                dragHandle='.nodeDragable',
                zIndex=-10 if n.type == 'note' else 20,
            )
        else:
            node = Node(
                id=n.id,
                type=n.type,
                data=n.data,
                position=Position(x=n.x * 500, y=n.y * 400),
                dragHandle='.nodeDragable',
                zIndex=-10 if n.type == 'note' else 20,
            )
        nodes.append(node)

    edges = []
    for e in exist_edges:
        source_handle = f'{e.source_node}-out-{e.source_handle}'
        target_handle = f'{e.target_node}-in-{e.target_handle}'
        edge = Edge(
            id=f'xy-edge__{source_handle}-{target_handle}',
            animated=True,
            source=e.source_node,
            # pyrefly: ignore [bad-argument-type]
            target=e.target_node,
            sourceHandle=source_handle,
            targetHandle=target_handle,
        )
        edges.append(edge)

    return WorkflowDefinition(nodes=nodes, edges=edges)
