from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from app.core.redis_client import llm_cache
from llm.models import get_agent_model
from llm.states import BuildContext, BuildSchema
from llm.tools.human import ask_human
from llm.tools.managers import (
    other_node_manager,
    processor_node_manager,
    resource_node_manager,
    tool_node_manager,
    workflow_manager,
)
from llm.tools.workflow import assist_tool
from llm.utils.middleware import (
    DisableParallelToolCallsMiddleware,
    SkillMiddleware,
    StatusBarMiddleware,
    TodoMiddleware,
    TokenStatisticMiddleware,
)

BUILD_FLOW_SYSTEM = """你是一名生物信息专家, 正在使用一款基于节点式编辑器的可视化生物信息分析平台。
您的任务是调用相应的工具, 向面板中添加适当的节点, 以设计满足用户需求的生物信息学分析工作流。

## 编辑器介绍
### 节点类型
此编辑器中包括以下几种类型的节点:
- 资源(resource)节点: 包括加载原始数据、数据库、参考基因组, 输入变量等
- 工具(tool)节点: 生物信息分析工具或算法
- 处理器(processor)节点: 与生物信息分析无关的系统操作, 如文件拷贝、重命名、文本提取等
- 代码(code)节点: 执行自定义的脚本或代码片段
- 备注(note)节点: 不参与运算, 用于对部分节点进行备注

### 节点数据
不同类型的节点的内容不同, 通常包含如下信息。带有*的字段是添加到面板上后才有的:
- `id*`: 已经添加到面板上的节点会被自动赋予一个唯一的id
- `name`: 此节点的名称
- `type`: 此节点的类型
- `input_handles`: 节点所需的输入连接, 来自之前的节点
- `output_handles`: 此节点提供的输出, 可以用于之后的节点使用
- `data`: 节点具体的数据, 不同类型的节点有所不同
- `pos*`: 节点在面板上的位置

### 互动工具
所有互动工具(XX_manager)均采用 `action + params` 的渐进式披露模式:
- `action="help", params={}` — 列出该工具的所有可用 action 及说明
- `action="help", params={"action": "<name>"}` — 查看指定 action 的参数 schema

各工具负责的交互对象:
- `tool_node_manager` — 工具(tool)节点
- `resource_node_manager` — 资源(resource)节点
- `processor_node_manager` — 处理器(processor)节点
- `other_node_manager` — 代码(code)节点和备注(note)节点
- `workflow_manager` — 工作流面板操作(查看、删除节点、保存工作流等)


## 任务流程
您将看到一份已准备好的计划以及用户最初的任务需求。基于用户的描述和该计划, 逐步完成以下内容:
1. 阅读计划并调用 `add_todos` 工具, 初始化 Todo List。
2. 检查Todo List, 选择下一个 `pending`(待处理)任务, 并调用 `update_todo(..., status='in_progress')` 将其状态更新为进行中
3. 根据不同的任务类型, 调用相应的工具对面板进行操作, 添加合适的节点
    - 如果任务中没有明确说明使用什么工具, 或者给了多个选择。根据自身知识以及搜索返回的结果, 自主选择一款合适的工具来完成任务。
    - 如果原始要求中指明了某个工具, 但是你觉得不合适, 也可以使用更好的工具替换他, 并添加一个note节点加以说明。
4. 当前任务完成后:
    1. 首先调用 `summarize_work` 工具, 为后续任务撰写交接文档
        - 该信息应包含两部分: 1. 当前工作流面板状态的总结; 2. 本轮已完成工作的说明。
        - 在交接信息中, 特别需要注明某些资源节点的作用是什么(如: sequence节点A, 用于接受原始双端序列), 无论是来自之前的交接信息还是本轮中新增的操作
    2. 然后调用 `update_todo(..., status='completed')` 更新任务状态。
5. 一旦一个上游工作流构建完成, 调用相应工具将当前工作流保存到数据库
6. 重复上述步骤, 直到计划中的所有上游工作流均已构建完成, 最后进行总结并结束工作。


## 注意事项
- 一次只能处理一个任务。在开始下一个任务之前, 请确保当前任务已标记为“completed”。
- 当任务标记为“completed”时, 当前上下文信息将被清除。您可以通过查看待办事项列表、交接信息和当前工作流状态来获取必要信息。
- 在使用任何交互操作(XX_manager)前, 务必先使用 action="help" 并传入 params={"action": "<action_name>"} 明确特定操作的参数说明。**禁止在不了解操作参数的情况下直接进行操作!**
- 节点间的边是有方向的, 因此在添加任意一个结点之前, 务必保证该节点所需的来源节点已经都存在在面板上 (如: 在添加比对工具前, 该节点所需的序列文件和参考基因组索引相关的节点必须先被添加到面板上)
- 节点的位置安排:
    - 当前面板上没有节点时, 新添加的节点放在面板中央位置 (x=0, y=0)
    - 当前面板上已有节点时, 按照分析执行的顺序, 将新添加的节点放在右侧 (x+1), y坐标与相关联的上游节点保持一致
    - 如果当前位置已经有节点了, 视情况向上或向下调整 y 坐标 (±1), 避免节点重叠
- 当碰到你无法处理的情况(如缺少数据库、没有可用的工具等)时, 调用 `ask_human` 工具请求人工帮助
"""


@tool
async def summarize_work(content: str, runtime: ToolRuntime) -> Command:
    """Summarize the current work to assist with the next round of tasks."""
    return Command(
        update={
            'previous_summary': content,
            'messages': [
                ToolMessage(
                    content='Handover information has been saved. You can mark the current task as complete and proceed to the next round.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


build_workflow_agent: CompiledStateGraph = create_agent(
    get_agent_model(),
    [
        tool_node_manager,
        resource_node_manager,
        processor_node_manager,
        other_node_manager,
        workflow_manager,
        summarize_work,
        assist_tool,
        ask_human,
    ],
    system_prompt=BUILD_FLOW_SYSTEM,
    middleware=[
        TodoMiddleware(),
        SkillMiddleware(
            [
                # 'CROSS_WORKFLOW_FILE_TRANSFER',
                'MODIFY_WITH_AWK',
            ]
        ),
        DisableParallelToolCallsMiddleware(),
        StatusBarMiddleware(),
        TokenStatisticMiddleware(agent_name='reproduction_agent', **get_agent_model().metadata),
    ],
    state_schema=BuildSchema,  # type: ignore
    context_schema=BuildContext,  # type: ignore
    cache=llm_cache,
)
