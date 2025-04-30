from graphlib import TopologicalSorter
from typing import List, Any
from pydantic import BaseModel, Field


class Position(BaseModel):
    x: float
    y: float


class NodeBase(BaseModel):
    id: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class Node(NodeBase):
    position: Position


class Edge(BaseModel):
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class WorkflowDefinition(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

    def get_graph_dict(self) -> list[NodeBase]:
        graph: dict[str, set[str]] = {}
        node_map = {}
        edge_map: dict[str, dict[str, tuple[str, str]]] = {}
        task_list = []
        for edge in self.edges:
            if edge.target in graph:
                graph[edge.target].add(edge.source)
                edge_map[edge.target][edge.targetHandle] = (
                    edge.sourceHandle.split('-out-')[0],
                    edge.sourceHandle.split('-out-')[-1],
                )
            else:
                graph[edge.target] = {edge.source}
                edge_map[edge.target] = {
                    edge.targetHandle.split('-')[-1]: (
                        edge.sourceHandle.split('-out-')[0],
                        edge.sourceHandle.split('-out-')[-1],
                    )
                }

        for node in self.nodes:
            if node.id in edge_map:
                node.data['kwargs'] = edge_map[node.id]
            node_map[node.id] = NodeBase.model_validate(node.model_dump())

        sorter = TopologicalSorter(graph)
        node_list = tuple(sorter.static_order())

        for node_id in node_list:
            task_list.append(node_map[node_id])

        return task_list
