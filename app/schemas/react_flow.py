from typing import Optional
from pydantic import BaseModel, Field


class Position(BaseModel):
    x: float
    y: float


class NodeData(BaseModel):
    tool_uid: Optional[str] = None
    args: str


class NodeBase(BaseModel):
    id: str
    type: str
    data: NodeData


class Node(NodeBase):
    position: Position
    dragHandle: str
    zIndex: int


class EdgeBase(BaseModel):
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class Edge(EdgeBase):
    id: str
    animated: bool


class WorkflowDefinition(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class TaskInput(BaseModel):
    source_uid: str
    source_k: str


class TaskNode(BaseModel):
    id: str
    type: str
    tool_id: Optional[int] = None
    args: str
    kwarg: dict[str, TaskInput] = Field(default_factory=dict)
