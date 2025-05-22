from typing import Optional
from pydantic import BaseModel, Field


class Position(BaseModel):
    x: float
    y: float


class Data(BaseModel):
    tool_uid: Optional[str] = None
    args: str


class NodeBase(BaseModel):
    id: str
    type: str
    data: Data


class Node(NodeBase):
    position: Position


class EdgeBase(BaseModel):
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class Edge(EdgeBase):
    id: str
    animated: bool


class WorkflowDefinition(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


class TaskInput(BaseModel):
    source_uid: str
    source_k: str


class TaskNode(BaseModel):
    id: str
    type: str
    tool_uid: Optional[str] = None
    args: str
    kwarg: dict[str, TaskInput] = Field(default_factory=dict)
