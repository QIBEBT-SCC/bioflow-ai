import operator
import uuid
from typing import Annotated, Any, Literal, NotRequired, Optional

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from bioflow_ai.infrastructure.containers import ContainerMount
from bioflow_ai.llm.tools.test_run_report import TestRunReport
from bioflow_ai.schemas.react_flow import WorkflowInterface
from bioflow_ai.schemas.tool_config import FileMount, ImageConfig, ParamDefine, ToolType
from bioflow_ai.schemas.workflow_build import BuildEdge, BuildNode
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.utils.uid_mapper import BidirectionalMapper


def union_uids(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Merge UID lists in first-seen order without duplicates."""
    return list(dict.fromkeys([*(existing or []), *(new or [])]))


# =====================create tool agent state==========================
class CreateToolState(AgentState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID
    image: ImageConfig

    test_run_report: TestRunReport
    doc_uid: uuid.UUID
    help_command: str

    group_id: int
    tag_ids: list[int]

    name: str
    tool_type: ToolType
    description: str
    file_mounts: list[FileMount]
    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str
    modifiable_static_params: str
    validated: bool

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
    test_run_mounts: NotRequired[list[ContainerMount]]
    updated: NotRequired[bool]
    checked: NotRequired[bool]


# =====================tool generator state==========================
class ToolGeneratorState(AgentState, total=False):
    image: ImageConfig
    test_run_report: TestRunReport
    id_mapper: BidirectionalMapper
    validation_runs: dict[str, ToolOutput]
    execution_count: int
    node_execution_failures: dict[str, int]

    name: str
    tool_type: ToolType
    description: str
    file_mounts: list[FileMount]
    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str
    modifiable_static_params: str

    validated: bool


# =====================project planner state==========================
class ProjectPlannerState(AgentState, total=False):
    # 样本注册工具使用
    base_path: str
    sample_maps: Annotated[dict[str, str], operator.or_]  # {sample_name: sample_id}

    # 文件工具未显式指定工作目录时的默认容器目录, 缺省为 /work
    default_dir: Literal['/app/data', '/work']

    # web_run 使用
    web_refs: Annotated[dict[str, dict], operator.or_]


# =====================workflow assistant state==========================
class WorkflowAssistantState(AgentState, total=False):
    diagnosis_path: str


# =====================build workflow agent state==========================
class WorkflowDraft(BaseModel):
    draft_id: str
    name: str
    description: str = ''
    execution_scope: Literal['sample_level', 'project_level'] = 'sample_level'
    nodes: list[BuildNode] = Field(default_factory=list)
    edges: list[BuildEdge] = Field(default_factory=list)
    interface: WorkflowInterface | None = None
    next_node_index: int = Field(default=1, ge=1)


class UserInputState(AgentState, total=False):
    """Checkpointed corrections that survive ordinary message-history cleanup."""

    user_input_cursor: int
    user_input_acked: int
    user_input_messages: list[HumanMessage]


class BuildState(UserInputState, total=False):
    id_mapper: BidirectionalMapper
    # Every tool UID matched by this workflow's searches, shown or not; used to spot concurrent creations.
    seen_tool_uids: Annotated[list[str], union_uids]
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
    target_workflow_uid: str
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
