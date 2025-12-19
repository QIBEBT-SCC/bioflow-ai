import operator
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from app.models.tools import FileMount, ImageConfig, ParamDefine
from app.schemas.react_flow import NodeData
from app.schemas.tools import Handle
from llm.utils.uid_mapper import BidirectionalMapper


# =====================tool generator state==========================
class ToolGeneratorState(MessagesState, total=False):
    name: str
    description: str
    file_mounts: list[FileMount]
    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str
    modifiable_static_params: str


# =====================create tool agent state==========================
class SearchImageState(MessagesState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID
    image: ImageConfig

    name: str
    description: str
    homepage: str
    paper_link: str


class GetDocState(MessagesState):
    id_mapper: NotRequired[BidirectionalMapper]

    image_uid: uuid.UUID
    image: ImageConfig

    doc_uid: NotRequired[uuid.UUID]


class SelectGroupState(MessagesState):
    group_id: NotRequired[int]


@dataclass
class CreateToolContext:
    auto_save: bool = True

    save_message: bool = False
    dataset_file: str = 'test_run.parquet'
    test_run_id: Optional[str] = None


class CreateToolState(MessagesState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID
    image: ImageConfig

    doc_uid: uuid.UUID

    group_id: int

    tool_uid: str
    json_response: dict[str, Any]


class CreateToolOutput(BaseModel):
    messages: list[BaseMessage]

    id_mapper: Optional[BidirectionalMapper]
    json_response: dict[str, Any]


# =====================test run agent state==========================
class TestRunState(MessagesState):
    image: ImageConfig
    container_id: NotRequired[str]

    checked: NotRequired[bool]


@dataclass
class TestRunContext:
    folder_name: str

    save_message: bool = False
    test_run_id: Optional[str] = None


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
    unknown: list[str] = Field(description='')


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


class ExtraWorkflowState(MessagesState):
    analyse_step: Annotated[int, operator.add]
    formatted_msg: Optional[str]

    tools: list[str]
    r_packages: list[str]
    python_packages: list[str]
    unknown: list[str]


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


class ReproductionState(MessagesState, total=False):
    id_mapper: BidirectionalMapper

    exist_nodes: Annotated[list[BuildNode], operator.add]
    exist_edges: Annotated[list[BuildEdge], operator.add]

    workflow_uid: uuid.UUID
    price: Annotated[float, operator.add]


@dataclass
class ReproductionContext:
    user_id: int
