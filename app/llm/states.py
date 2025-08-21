import operator
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
from pydantic import BaseModel, Field

from app.models.tools import ImageConfig
from app.schemas.react_flow import NodeData
from app.schemas.tools import Handle


# =====================create tool agent state==========================
#
class SearchImageState(AgentStatePydantic):
    image_id: Optional[int] = None
    image: Optional[ImageConfig] = None


class GetDocState(AgentStatePydantic):
    image_id: Optional[int] = None
    image: Optional[ImageConfig] = None

    last_output: Optional[str] = None
    doc_id: Optional[int] = None


class TestRunState(AgentStatePydantic):
    image: Optional[ImageConfig] = None
    container_id: Optional[str] = None


class SelectGroupState(AgentStatePydantic):
    group_id: Optional[int] = None


class CreateToolState(AgentStatePydantic):
    image_id: Optional[int] = None
    image: Optional[ImageConfig] = None

    doc_id: Optional[int] = None
    doc: Optional[str] = None

    checked: bool = False

    group_id: Optional[int] = None
    tool_id: Optional[int] = None


class CreateToolOutput(BaseModel):
    messages: list[BaseMessage]
    tool_id: int


# =====================reproduction agent state==========================
#
class ToolsOutput(BaseModel):
    """Bioinformatics tools used in the literature"""

    tools: list[str] = Field(description='All tools used in the paper', default_factory=list)
    workflow_structure: str = Field(description="Workflow's Mermaid flowchart code. The flowchart's layout is LR")


class BuildNode(BaseModel):
    id: str
    type: str
    input_handles: list[Handle]
    output_handles: list[Handle]
    data: NodeData
    command: Optional[str] = None
    x: int
    y: int


class BuildEdge(BaseModel):
    source_node: str = Field(description='Node ID as data source (not tool or database ID)')
    source_handle: str = Field(description='The name value of the corresponding handle on the source node')
    target_node: Optional[str] = Field(
        description='The ID of the target node (i.e., this node). It will be automatically assigned by the program, just leave it blank.',
        default=None,
    )
    target_handle: str = Field(description='The name value of the corresponding handle on the target node')


class ReproductionState(AgentStatePydantic):
    paper_content: str
    paper_tools: Optional[ToolsOutput] = None
    checked_tools: Annotated[list[int], operator.add] = []
    exist_nodes: Annotated[list[BuildNode], operator.add] = []
    exist_edges: Annotated[list[BuildEdge], operator.add] = []

    price: Annotated[float, operator.add] = 0.0
