# ruff: noqa: RUF001
"""Workflow repair agent that edits the current template using a diagnosis report."""

from typing import TYPE_CHECKING, Optional, cast

from langchain.agents import create_agent

from llm.context import WorkflowFixContext
from llm.models import get_long_context_model
from llm.states import WorkflowFixState
from llm.tools.managers import resource_node_manager, tool_node_manager, utility_node_manager, workflow_manager
from llm.tools.workflow_fix import finalize_workflow_fix
from llm.utils.middleware import (
    CheckStateExistMiddleware,
    DisableParallelToolCallsMiddleware,
    SaveMessagesMiddleware,
    SkillMiddleware,
    StatusBarMiddleware,
    TokenStatisticMiddleware,
    WorkflowBuilderProgressMiddleware,
    WorkflowFixGuardMiddleware,
)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from llm.llm_client import LLMMetadata


SYSTEM_PROMPT = """你是 BioFlow AI 的工作流修复智能体。系统已经为你加载了当前对话中同一 source run 的最新成功诊断报告，以及数据库里当前最新的工作流模板。诊断报告是修复参考，当前面板才是实际修改基础。

## 必须遵循的流程

1. 首先调用 `workflow_manager` 的 `view_workflow`，再对诊断涉及的节点调用 `view_node`。将报告建议与当前面板逐项比对，识别用户在诊断后已经手动完成的修改。
2. 只有机器摘要同时满足 `classification` 为 `workflow_structure` 或 `workflow_usage`，并且 `recommended_action` 为 `patch_workflow` 时，才允许修改工作流。
3. 只实施仍然必要的最小变更。可以添加或删除节点、更新 Tool 节点可修改参数、修复连线。
4. 不得创建、切换或删除草稿，不得修改工作流元数据，不得调用 `save_all_workflows`。当前只有一个已加载的 active draft。
5. 不支持直接修改当前已有 Resource/Processor 节点的参数或节点定义。如果必须这样做才能修复，停止修改并用 `finalize_workflow_fix(mode="no_change")` 说明人工操作。
6. 若分类是 `node_definition`，动作是 `manual_node_fix`，或属于数据、环境、证据不足问题，禁止通过改工作流绕过，直接以 no_change 完成并给出具体人工建议。
7. 修改连线时，先查看完整工作流；若要替换现有有效连线，调用 `workflow_manager(action="fix_edge")` 时传入完整目标边集合并设置 `replace_all=true`。
8. 结束前必须且只能通过 `finalize_workflow_fix` 提交：
   - 确有变更：`mode="save"`，在 summary 中用中文列出修改内容、原因和验证建议；工具会原地更新模板并写 UPDATE.md。
   - 当前已修好、无需变更、无法安全修改或禁止修改：`mode="no_change"`，具体说明原因和下一步；不会创建 UPDATE.md。

## 编辑器工具

- `tool_node_manager`: `search`、`add`、`update_args`、`create`。对任一 action 先用 `help` 查看参数。
- `resource_node_manager`: `view`、`view_marks`、`search_db`、`create_db`、`add`。没有已有 Resource 参数更新操作。
- `utility_node_manager`: `search`、`view`、`add_processor`、`add_exist_code`、`add_new_code`、`add_note`、`add_downstream_summary`。
- `workflow_manager`: 只使用 `view_workflow`、`view_node`、`delete_node`、`fix_edge`。对修改 action 先用 `help` 查看参数。
- `finalize_workflow_fix`: 最终保存或无修改结论。

节点 ID 是当前模板 ID。删除节点可能触发后续 ID 重排，之后必须重新查看工作流。添加节点前先确认来源节点和 handle；修复必须保持工作流可执行，不添加占位节点。完成最终工具调用后，在聊天中简要说明是否修改、主要内容和如何验证；no_change 时明确说明人工处理建议。
"""

FIX_DISABLED_TOOLS = {'finalize_workflow_fix'}
FIX_DISABLED_TOOL_ACTIONS = {
    'tool_node_manager': {'add', 'update_args', 'create'},
    'resource_node_manager': {'create_db', 'add'},
    'utility_node_manager': {
        'add_processor',
        'add_exist_code',
        'add_new_code',
        'add_note',
        'add_downstream_summary',
    },
    'workflow_manager': {'delete_node', 'fix_edge'},
}


def build_workflow_fixer(checkpointer: Optional['BaseCheckpointSaver'] = None) -> 'CompiledStateGraph':
    """Build a Fix agent around the existing workflow editor managers."""
    model = get_long_context_model()
    metadata = cast('LLMMetadata', model.metadata)
    kwargs = {'checkpointer': checkpointer} if checkpointer is not None else {}
    # pyrefly: ignore [no-matching-overload]
    return create_agent(
        model,
        [
            tool_node_manager,
            resource_node_manager,
            utility_node_manager,
            workflow_manager,
            finalize_workflow_fix,
        ],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            WorkflowFixGuardMiddleware(),
            WorkflowBuilderProgressMiddleware(),
            SkillMiddleware('workflow_builder'),
            CheckStateExistMiddleware('fix_completed', 'finalize_workflow_fix'),
            DisableParallelToolCallsMiddleware(
                disabled_tools=FIX_DISABLED_TOOLS,
                disabled_tool_actions=FIX_DISABLED_TOOL_ACTIONS,
            ),
            StatusBarMiddleware(),
            SaveMessagesMiddleware('workflow_fixer'),
            TokenStatisticMiddleware('workflow_fixer', **metadata),
        ],
        state_schema=WorkflowFixState,
        context_schema=WorkflowFixContext,
        **kwargs,
    )
