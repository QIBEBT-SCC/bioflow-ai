from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Position(BaseModel):
    x: float
    y: float


class NodeBase(BaseModel):
    id: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class Node(NodeBase):
    position: Position
    dragHandle: str  # noqa: N815
    zIndex: int  # noqa: N815


class EdgeBase(BaseModel):
    source: str
    sourceHandle: str  # noqa: N815
    target: str
    targetHandle: str  # noqa: N815


class Edge(EdgeBase):
    id: str
    animated: bool | None = True


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
    data: dict[str, Any]
    kwarg: dict[str, TaskInput | list[TaskInput]] = Field(default_factory=dict)
