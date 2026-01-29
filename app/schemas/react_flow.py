from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Position(BaseModel):
    x: float
    y: float


class NodeData(BaseModel):
    resource_uid: Optional[str] = None
    args: str | dict[str, Any]


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
    animated: Optional[bool] = True


class WorkflowDefinition(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class TaskInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_uid: str
    source_k: str


class TaskNode(BaseModel):
    id: str
    type: str
    resource_uid: Optional[str] = None
    args: str | dict[str, Any]
    kwarg: dict[str, TaskInput | list[TaskInput]] = Field(default_factory=dict)
