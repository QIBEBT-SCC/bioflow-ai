from typing import List
from pydantic import BaseModel


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
    measured: Measured


class Edge(BaseModel):
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class WorkflowDefinition(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
