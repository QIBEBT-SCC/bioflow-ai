from typing import List, Any
from pydantic import BaseModel, Field


class Position(BaseModel):
    x: float
    y: float


class Measured(BaseModel):
    width: float
    height: float


class Node(BaseModel):
    id: str
    type: str
    position: Position
    data: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class WorkflowDefinition(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
