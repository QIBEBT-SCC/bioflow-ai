import operator
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Generic, Literal, NotRequired, Optional, TypedDict, TypeVar

from langchain.agents import AgentState
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field

from app.models.tools import FileMount, ImageConfig, ParamDefine
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
class ToolGeneratorState(AgentState, total=False):
    name: str
    description: str
    file_mounts: list[FileMount]
    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str
    modifiable_static_params: str


# =====================create tool agent state==========================
class SearchImageState(AgentState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID
    image: ImageConfig

    name: str
    description: str
    homepage: str
    paper_link: str


class GetDocState(AgentState):
    id_mapper: NotRequired[BidirectionalMapper]

    image_uid: uuid.UUID
    image: ImageConfig

    doc_uid: NotRequired[uuid.UUID]


class SelectGroupState(AgentState):
    group_id: NotRequired[int]


@dataclass
class CreateToolContext(UserContext):
    auto_save: bool = True

    save_message: bool = False
    dataset_file: str = 'test_run.parquet'
    test_run_id: Optional[str] = None


class CreateToolState(AgentState, total=False):
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
class TestRunState(AgentState):
    image: ImageConfig
    container_id: NotRequired[str]

    checked: NotRequired[bool]


@dataclass
class TestRunContext(UserContext):
    folder_name: str

    save_message: bool = False
    test_run_id: Optional[str] = None


# =====================extra workflow state==========================
class Todo(BaseModel):
    index: Optional[int] = None
    content: str
    status: Literal['pending', 'in_progress', 'completed']


class DesignState(AgentState, total=False):
    task: str
    plan_update: bool

    # 文献复现使用
    paper_id: str
    paper_summary: str

    # 暂存
    todos: list[Todo]

    # 生成节点输入用
    workflow_description: str

    # 生成接节点输出
    workflow_uids: list[uuid.UUID]


# =====================reproduction agent state==========================
class BuildNode(BaseModel):
    id: str
    name: str
    type: str
    input_handles: list[Handle]
    output_handles: list[Handle]
    data: dict[str, Any] = Field(default_factory=dict)
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


class GlobalMarker(BaseModel):
    mark_name: str
    description: str


class ManagerSchema(AgentState, total=False):
    paper_id: str

    id_mapper: BidirectionalMapper
    exist_nodes: Annotated[list[BuildNode], operator.add]
    exist_edges: Annotated[list[BuildEdge], operator.add]

    global_marks: Annotated[list[GlobalMarker], operator.add]


class BuildSchema(AgentState, total=False):
    paper_id: str

    todos: list[Todo]

    exist_nodes: list[BuildNode]
    exist_edges: list[BuildEdge]

    global_marks: list[GlobalMarker]

    workflow_uids: Annotated[list[uuid.UUID], operator.add]


@dataclass
class BuildContext(UserContext):
    has_reference: bool


# =====================summary agent state==========================
@dataclass
class FolderContext(UserContext):
    root_path: str
    allowed_files: list[str]
