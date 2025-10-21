import operator
import uuid
from typing import Annotated, Optional, Any

from langchain_core.messages import BaseMessage
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
from pydantic import BaseModel, Field

from app.models.tools import ImageConfig, FileMount, ParamDefine
from app.schemas.react_flow import NodeData
from app.schemas.tools import Handle
from llm.utils.uid_mapper import BidirectionalMapper


# =====================tool generator state==========================
class ToolGeneratorState(AgentStatePydantic):
    name: Optional[str] = None
    description: Optional[str] = None
    file_mounts: list[FileMount] = Field(default_factory=list)
    command_template: Optional[str] = None
    dynamic_params: list[ParamDefine] = Field(default_factory=list)
    static_params: Optional[str] = None
    mkdir_output: Optional[bool] = None
    use_temp_dir: Optional[bool] = None


# =====================create tool agent state==========================
class SearchImageState(AgentStatePydantic):
    id_mapper: Optional[BidirectionalMapper] = None

    image_uid: Optional[uuid.UUID] = None
    image: Optional[ImageConfig] = None

    github_url: Optional[str] = None


class GetDocState(AgentStatePydantic):
    id_mapper: Optional[BidirectionalMapper] = None

    image_uid: Optional[uuid.UUID] = None
    image: Optional[ImageConfig] = None

    last_output: Optional[str] = None
    doc_uid: Optional[uuid.UUID] = None


class SelectGroupState(AgentStatePydantic):
    group_id: Optional[int] = None


class CreateToolState(AgentStatePydantic):
    image_uid: Optional[uuid.UUID] = None
    image: Optional[ImageConfig] = None

    doc_uid: Optional[uuid.UUID] = None
    doc: Optional[str] = None

    command: Optional[str] = None

    github_url: Optional[str] = None

    group_id: Optional[int] = None
    tool_uid: Optional[uuid.UUID] = None

    json_response: dict[str, Any] = {}
    id_mapper: Optional[BidirectionalMapper] = None


class CreateToolOutput(BaseModel):
    messages: list[BaseMessage]

    id_mapper: Optional[BidirectionalMapper]
    json_response: dict[str, Any]


# =====================extra workflow state==========================
class ExtractOutput1(BaseModel):
    """A brief summary of the literature (requirements) and the extraction of tools mentioned in the text"""

    overall_summary: str = Field(
        description='The main conclusions/findings and general analytical direction/methods of this literature.'
    )
    major_finding: str = Field(
        description='Summary of the main findings/conclusions of the literature. Leave blank if it is not a complete research paper'
    )
    tools: list[str] = Field(description='All tools used in the paper', default_factory=list)
    r_packages: list[str] = Field(description='All R packages used in the analysis', default_factory=list)
    python_packages: list[str] = Field(description='All Python libraries used in the analysis', default_factory=list)
    unknown:list[str] = Field(description='')


class PaperWorkflow(BaseModel):
    """Details of each workflow"""

    name: str = Field(description='Name for this workflow')
    workflow_summary: str = Field(description='A brief description of the bioinformatics role of this workflow')
    conclusion_support: Optional[str] = Field(
        description='(Only for complete articles) How do the results of this workflow support the final conclusions of the article?',
        default=None,
    )
    workflow_detail: str = Field(description='Detailed workflow described in a list format')
    mermaid_code: str = Field(description="Workflow's Mermaid flowchart code. The flowchart's layout is LR")


class ExtractOutput2(BaseModel):
    """The workflows described in the user-provided materials"""

    workflows: list[PaperWorkflow] = Field(description='Workflow List', default_factory=list)


class ExtraWorkflowState(AgentStatePydantic):
    analyse_step: Annotated[int, operator.add] = 0
    formatted_msg: Optional[str] = None

    tools: list[str] = Field(default_factory=list)
    r_packages: list[str] = Field(default_factory=list)
    python_packages: list[str] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)


# =====================reproduction agent state==========================
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
    analyse_step: Annotated[int, operator.add] = 0
    formatted_msg: Optional[str] = None
    paper_tools: Optional[ExtractOutput2] = None

    id_mapper: Optional[BidirectionalMapper] = None

    exist_nodes: Annotated[list[BuildNode], operator.add] = []
    exist_edges: Annotated[list[BuildEdge], operator.add] = []

    workflow_uid: Optional[uuid.UUID] = None

    price: Annotated[float, operator.add] = 0.0
