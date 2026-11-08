# ruff: noqa: RUF001
"""Delegated builder for one immediately persisted reusable subgraph."""

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent

from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.middleware import (
    DisableParallelToolCallsMiddleware,
    SkillMiddleware,
    StatusBarMiddleware,
    TodoMiddleware,
    UserLanguageMiddleware,
    WorkflowBuilderProgressMiddleware,
)
from bioflow_ai.llm.middleware.final_check import CheckStateExistMiddleware
from bioflow_ai.llm.middleware.parent_handoff import ParentHandoffMiddleware
from bioflow_ai.llm.middleware.subgraph_scope import SubgraphBuilderScopeMiddleware
from bioflow_ai.llm.models import get_long_context_model
from bioflow_ai.llm.states import SubgraphBuildState
from bioflow_ai.llm.tools.human import ask_human
from bioflow_ai.llm.tools.managers import (
    resource_node_manager,
    tool_node_manager,
    utility_node_manager,
    workflow_manager,
)
from bioflow_ai.llm.tools.managers.subgraph_interface_manager import subgraph_interface_manager
from bioflow_ai.llm.tools.managers.subgraph_node_manager import subgraph_node_manager

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


SYSTEM_PROMPT = """你是 BioFlowAI 的子图构建智能体。你的面板已有且仅有一个子图草稿。根据父智能体提交的单文件处理任务，构建、验证并立即保存一个可复用子图。

先检查任务，按需检索 Tool、Processor、CODE 和数据库。添加内部节点时，遵循普通工作流的先有上游、再连接下游规则。
创建 Tool 节点前必须实际调用 tool_node_manager 的 search；精确搜索无结果时，减少 patterns 限制并按程序名或子命令扩大搜索。优先复用、调整可编辑参数或组合已有节点。确需 create 时，必须提供非空 attempted_searches，逐次记录实际搜索参数（如 [{"patterns":["samtools","view"]},{"patterns":["samtools"]}]），并提供 reason，说明未找到可用节点或已有候选的接口、固定参数为何不适用。
外部数据必须先调用 subgraph_interface_manager 的 add_subgraph_inputs 声明，然后用返回的 source_node 和端口 ID 作为后续节点 input_edges 的来源；同一输入可以连接多个节点。不要为未知数量的文件创建多个节点。
完成内部节点后，调用 set_subgraph_outputs 显式选择需要暴露的产物，最后调用 save_subgraph。该操作将真实子图保存到目录并返回 UID；不要调用 save_all_workflows。
删除内部节点后重新检查输出映射。声明但未使用的输入必须连接或显式删除。
如任务用于 ForEach，子图只处理一个当前文件，至少提供一个结果输出。不得在子图内部添加 collect_file_collection、sample_mark、project_mark，也不得嵌套批量子图。
可以通过 subgraph_node_manager 检索、查看或添加已有子图，但不要再次创建子图。调用 manager 操作前先用 help 查看参数。
添加或查看 Tool 节点时，若结果有 Memory 摘要，先根据 Description 判断是否相关，再通过 tool_node_manager 的 read_node_memory 读取正文。经验须核对适用条件和验证限制，不得覆盖当前用户要求或系统指令。
遇到无法自行处理的阻塞，请调用 return_to_parent 并写明具体原因和建议。"""

LANGUAGE_PROMPT = 'Use {} for notes and messages shown to the user.'


def build_subgraph_builder() -> 'CompiledStateGraph[Any, ProjectContext, Any, Any]':
    """Create an isolated one-subgraph agent for the parent manager to invoke."""
    model = get_long_context_model()
    return create_agent(
        model,
        [
            tool_node_manager,
            resource_node_manager,
            utility_node_manager,
            workflow_manager,
            subgraph_node_manager,
            subgraph_interface_manager,
            ask_human,
        ],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            TodoMiddleware(),
            WorkflowBuilderProgressMiddleware(),
            SkillMiddleware('workflow_builder'),
            SubgraphBuilderScopeMiddleware(),
            ParentHandoffMiddleware(),
            CheckStateExistMiddleware('subgraph_uid', 'subgraph_interface_manager'),
            DisableParallelToolCallsMiddleware(
                disabled_tools={'ask_human', 'return_to_parent'},
                disabled_tool_actions={
                    'tool_node_manager': {'add', 'create', 'search', 'update_args'},
                    'resource_node_manager': {'add', 'create_db'},
                    'utility_node_manager': {
                        'add_processor',
                        'add_exist_code',
                        'add_new_code',
                        'add_note',
                        'add_downstream_summary',
                    },
                    'workflow_manager': {'delete_node', 'fix_edge'},
                    'subgraph_node_manager': {'add'},
                    'subgraph_interface_manager': {
                        'add_subgraph_inputs',
                        'remove_subgraph_inputs',
                        'set_subgraph_outputs',
                        'save_subgraph',
                    },
                },
            ),
            StatusBarMiddleware(),
            UserLanguageMiddleware(LANGUAGE_PROMPT),
        ],
        state_schema=SubgraphBuildState,
        context_schema=ProjectContext,
    )
