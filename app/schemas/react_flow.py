from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Position(BaseModel):
    x: float
    y: float


class NodeBase(BaseModel):
    id: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class Node(NodeBase):
    @model_validator(mode='after')
    def validate_subgraph(self) -> 'Node':
        """Validate embedded definitions while retaining ordinary node-specific data."""
        if self.type == 'subgraph':
            self.data = SubgraphData.model_validate(self.data).model_dump(mode='json')
        return self

    position: Position | None = None
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


class PortTarget(BaseModel):
    node_id: str = Field(min_length=1)
    handle: str = Field(min_length=1)


class InterfaceInput(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    targets: list[PortTarget] = Field(min_length=1)


class InterfaceOutput(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source: PortTarget


class InterfaceParameter(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    node_path: list[str] = Field(min_length=1)
    field: str = Field(min_length=1)


class InterfacePositions(BaseModel):
    inputs: Position
    outputs: Position


class WorkflowInterface(BaseModel):
    positions: InterfacePositions | None = None
    inputs: list[InterfaceInput] = Field(default_factory=list)
    outputs: list[InterfaceOutput] = Field(default_factory=list)
    parameters: list[InterfaceParameter] = Field(default_factory=list)

    @model_validator(mode='after')
    def unique_ids(self) -> 'WorkflowInterface':
        """Port identities are independent of their editable display names."""
        for ports in (self.inputs, self.outputs, self.parameters):
            if len({port.id for port in ports}) != len(ports):
                message = 'Duplicate interface ID'
                raise ValueError(message)
        return self


class WorkflowDefinition(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    interface: WorkflowInterface | None = None


class TaskInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_uid: str
    source_k: str


class TaskNode(BaseModel):
    id: str
    type: str
    data: dict[str, Any]
    kwarg: dict[str, TaskInput | list[TaskInput]] = Field(default_factory=dict)


class SubgraphData(BaseModel):
    model_config = ConfigDict(extra='allow')

    name: str = 'Subgraph'
    source_uid: str | None = None
    workflow: WorkflowDefinition
