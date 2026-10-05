"""Transport-independent values used while editing workflow graphs."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from bioflow_ai.schemas.tools import Handle


class BuildNode(BaseModel):
    id: str
    name: str
    type: str
    input_handles: list[Handle]
    output_handles: list[Handle]
    data: dict[str, Any] = Field(default_factory=dict)
    command: Optional[str] = None


class BuildEdge(BaseModel):
    source_node: str = Field(description='Node ID as data source (not tool or database ID)')
    source_handle: str = Field(description='The name value of the corresponding handle on the source node')
    target_node: Optional[str] = Field(
        description='The ID of the target node (i.e., this node). It will be automatically assigned by the program, just leave it blank.',
        default=None,
    )
    target_handle: str = Field(description='The name value of the corresponding handle on the target node')
