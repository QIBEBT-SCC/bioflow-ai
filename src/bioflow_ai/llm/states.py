import operator
import uuid
from typing import Annotated, Any, Literal, NotRequired, Optional

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from bioflow_ai.llm.tools.test_run_report import TestRunReport
from bioflow_ai.llm.utils.uid_mapper import BidirectionalMapper
from bioflow_ai.schemas.react_flow import WorkflowInterface
from bioflow_ai.schemas.tool_config import FileMount, ImageConfig, ParamDefine
from bioflow_ai.schemas.tools import Handle


# =====================create tool agent state==========================
class CreateToolState(AgentState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID
    image: ImageConfig

    test_run_report: TestRunReport
    doc_uid: uuid.UUID
    help_command: str

    group_id: int

    name: str
    description: str
    file_mounts: list[FileMount]
    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str
    modifiable_static_params: str

    tool_uid: str
    json_response: dict[str, Any]

    return_to_parent: bool
    parent_request: str


class CreateToolOutput(BaseModel):
    messages: list[BaseMessage]

    id_mapper: Optional[BidirectionalMapper]
    tool_uid: Optional[str] = None
    json_response: dict[str, Any]
    return_to_parent: bool = False
    parent_request: str | None = None


# =====================test run agent state==========================
class TestRunState(AgentState):
    test_run_report: NotRequired[TestRunReport]
    updated: NotRequired[bool]
    checked: NotRequired[bool]


# =====================tool generator state==========================
class ToolGeneratorState(AgentState, total=False):
    image: ImageConfig
    test_run_report: TestRunReport

    name: str
    description: str
    file_mounts: list[FileMount]
    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str
    modifiable_static_params: str

    validated: bool


# =====================sample manager state==========================
class SampleState(AgentState, total=False):
    base_path: str
    sample_maps: Annotated[dict[str, str], operator.or_]  # {sample_name: sample_id}

    summary: str


# =====================workflow diagnosis state==========================
class WorkflowDiagnosisState(AgentState, total=False):
    diagnosis_path: str
    diagnosis_written: bool
    source_run_uid: str
    workflow_uid: str | None
    updated: bool


# =====================unified planner state==========================
class PlannerState(AgentState, total=False):
    """聚合三种规划场景 (自主设计/工具扩展/文献复现) 所需的全部可选状态字段.

    所有字段均为可选: 不同场景只会写入其中一部分, 后续处理时按需读取.
    """

    task: str
    plan: str

    # 工具探索用
    tool_report: str

    # 文献复现用
    paper_id: str
    paper_context: str

    # 写入类工具标记
    updated: bool

    # 透传给下游 workflow_builder
    workflow_uids: list[uuid.UUID]
    new_tools: list[uuid.UUID]
    new_dbs: list[int]


# =====================build workflow agent state==========================
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


class WorkflowDraft(BaseModel):
    draft_id: str
    name: str
    description: str = ''
    execution_scope: Literal['sample_level', 'project_level'] = 'sample_level'
    nodes: list[BuildNode] = Field(default_factory=list)
    edges: list[BuildEdge] = Field(default_factory=list)
    interface: WorkflowInterface | None = None
    next_node_index: int = Field(default=1, ge=1)


class BuildState(AgentState, total=False):
    paper_id: str

    previous_summary: str

    id_mapper: BidirectionalMapper
    exist_nodes: Annotated[list[BuildNode], operator.add]
    exist_edges: Annotated[list[BuildEdge], operator.add]
    next_node_index: int

    active_workflow_id: str | None
    workflow_drafts: dict[str, WorkflowDraft]

    workflow_uids: Annotated[list[uuid.UUID], operator.add]
    new_subgraphs: Annotated[list[uuid.UUID], operator.add]
    new_tools: Annotated[list[uuid.UUID], operator.add]
    new_dbs: Annotated[list[int], operator.add]


class SubgraphBuildState(BuildState, total=False):
    subgraph_outputs: list[dict[str, Any]]
    subgraph_outputs_set: bool
    subgraph_uid: uuid.UUID
    subgraph_creation_id: str
    subgraph_purpose: Literal['ordinary', 'foreach']
    return_to_parent: bool
    parent_request: str


class WorkflowFixState(BuildState, total=False):
    diagnosis_run_uid: str | None
    target_workflow_uid: str
    source_subgraph_path: list[str]
    diagnosis_report: str | None
    original_workflow: dict[str, Any]
    original_node_id_mapping: dict[str, str]
    subgraph_stack: list[dict[str, Any]]
    subgraph_source_baselines: dict[str, dict[str, Any]]
    changed_subgraph_paths: list[list[str]]
    changed_workflow_uids: list[str]
    fix_completed: bool
    workflow_changed: bool
    update_path: str | None
    updated: bool
