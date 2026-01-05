import operator
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Generic, NotRequired, Optional, TypedDict, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from app.models.tools import FileMount, ImageConfig, ParamDefine
from app.schemas.react_flow import NodeData
from app.schemas.tools import Handle
from llm.utils.uid_mapper import BidirectionalMapper

ParsedT = TypeVar('ParsedT', bound=BaseModel | None, default=None)


class StructuredResponse(TypedDict, Generic[ParsedT]):
    raw: AIMessage
    parsed: ParsedT
    parsing_error: NotRequired[Any]


@dataclass
class UserContext:
    user_id: int


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
class CreateToolContext(UserContext):
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
class TestRunContext(UserContext):
    folder_name: str

    save_message: bool = False
    test_run_id: Optional[str] = None


# =====================extra workflow state==========================
class DesignState(MessagesState, total=False):
    task: str
    paper_id: str

    workflow_description: str

    workflow_uids: list[uuid.UUID]


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


class BuildSchema(MessagesState, total=False):
    paper_id: str

    id_mapper: BidirectionalMapper
    exist_nodes: Annotated[list[BuildNode], operator.add]
    exist_edges: Annotated[list[BuildEdge], operator.add]

    workflow_uids: Annotated[list[uuid.UUID], operator.add]
    price: Annotated[float, operator.add]
