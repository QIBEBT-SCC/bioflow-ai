# ruff: noqa: RUF001
"""Read-only assistant for questions and diagnosis about one workflow run."""

from typing import TYPE_CHECKING, Any, Optional, cast

from langchain.agents import create_agent

from bioflow_ai.llm.context import WorkflowAssistantContext
from bioflow_ai.llm.middleware import (
    ChatAttachmentMiddleware,
    DisableParallelToolCallsMiddleware,
    SkillMiddleware,
    TokenStatisticMiddleware,
    ToolProgressMiddleware,
    UserLanguageMiddleware,
)
from bioflow_ai.llm.models import get_high_performance_model
from bioflow_ai.llm.states import WorkflowAssistantState
from bioflow_ai.llm.tools.container import run_evidence_command
from bioflow_ai.llm.tools.view_image import view_image
from bioflow_ai.llm.tools.workflow_diagnosis import workflow_reader, write_diagnosis

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from bioflow_ai.llm.llm_client import LLMMetadata

SYSTEM_PROMPT = """你是工作流助手，根据用户当前的问题，读取指定工作流运行的状态、配置、日志和输出，给出有证据支持的回答。普通问题按需检查后直接回答；用户请求诊断、排查故障或调查失败原因时，先加载诊断 Skill，再按 Skill 完成调查和报告。会话中的旧请求和报告只提供背景，不代表本轮也要诊断或重新写报告。

## 工具
- `workflow_reader`：按需读取运行保存的工作流快照和执行证据。`overview` 默认只返回失败节点的分页摘要；已知目标时用 `node` 读取节点配置、拼合后的工具命令模板、输入句柄说明及来源节点/源句柄、逐个输出句柄的已记录结果。`no_result` 表示没有结果记录，不能据此断定文件不存在；CODE 节点源码在 data 中。需要依赖关系时用 `overview(node_id=..., upstream_depth=1)` 查看目标和直接上游，必要时再扩大层数；也可不传 node_id，从所有失败节点向上游展开。范围内成功节点会保留，摘要中的节点可按需逐个调用 `node`。只有用户需要整体结构时才显式设置 `failed_only=false` 分页查看全部节点。普通子图用 `subgraph_path` 逐层进入；批量子图用 `children` 分页获取失败子运行，再用 `child_run_uid` 查看；这两种场景同样支持局部上游范围。有 `next_offset` 时继续翻页，否则已返回完毕。`node_id` 是所选子图层级的本地 ID，`upstream_depth` 按依赖边的层数计算。
- `run_command`：在一次性容器中检查一个只读来源：`run:current`、已验证的 `run:child:<uid>`、当前样本的 `sample:<tag>` 或项目共享文件 `proj:<keyword>`。证据挂载在 `/app/data/input`，可能是文件或目录；不要把宿主机绝对路径作为 source。可使用 find、file、fd、rg、jq、yq、mlr、seqkit、head、tail 等工具，每次只检查一个来源。
- `write_diagnosis`：仅在用户请求诊断时提交 `DIAGNOSIS.md`，普通问答和报告解释不调用。
- `view_image`：需要查看输出图表、截图或其他图片时使用。source 与 run_command 相同；path 是来源目录内的相对路径，来源本身为图片文件时使用 '.'。支持 PNG、JPEG、WebP、TIFF 和 SVG；多页 TIFF 默认首页，可用从 1 开始的 page 选择页面。结果含直接可见的视觉输入；不能把文件路径当成已读取图片。文件变化或需要重新核对细节时再次读取。

## 回答与边界
本轮存在 current_node_reference 时，“这个节点”指向该引用；先按 node_id 与 subgraph_path 调用 workflow_reader(action="node") 获取当前证据。附件名称仅用于显示，不代表结果快照。工具用途依据节点详情中的 name、description 和命令模板回答，工具定义缺失时说明限制。历史消息中的节点附件不会自动成为本轮选中节点。
围绕当前指定运行回答，区分事实与推断，不把未运行或上游阻塞判为节点自身失败。检查失败、超时、截断或证据不足时说明限制。仅可通过 `write_diagnosis` 写指定报告，其他工作流、文件和数据库均只读；不调用网络、不安装软件、不搜索节点库，不声称使用不存在的编辑工具。已有报告是历史资料，涉及当前状态时按需重新读取运行证据。
"""

LANGUAGE_PROMPT = """## 语言
回答和诊断报告一律使用 {}。"""


def build_workflow_assistant(
    checkpointer: Optional['BaseCheckpointSaver'] = None,
) -> 'CompiledStateGraph[Any, WorkflowAssistantContext, Any, Any]':
    """Build the workflow run assistant with optional diagnosis skills."""
    model = get_high_performance_model()
    metadata = cast('LLMMetadata', model.metadata)
    kwargs = {'checkpointer': checkpointer} if checkpointer is not None else {}
    return create_agent(
        model,
        [workflow_reader, run_evidence_command, view_image, write_diagnosis],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ToolProgressMiddleware(),
            SkillMiddleware('workflow_assistant'),
            DisableParallelToolCallsMiddleware(disabled_tools={'write_diagnosis'}),
            TokenStatisticMiddleware('workflow_assistant', **metadata),
            UserLanguageMiddleware(LANGUAGE_PROMPT),
            ChatAttachmentMiddleware(),
        ],
        state_schema=WorkflowAssistantState,
        context_schema=WorkflowAssistantContext,
        **kwargs,
    )
