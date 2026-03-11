from typing import Literal

import nanoid
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite

from llm.agents.tool_manager import CreateToolAgent
from llm.models import get_agent_model
from llm.states import BuildContext, CreateToolContext, CreateToolState, ManagerSchema
from llm.tools.database import search_biodb, search_tool
from llm.tools.human import ask_human
from llm.tools.vector_store import get_paper_snippets, search_processors
from llm.tools.workflow import (
    add_code_node,
    add_note_node,
    add_processor_node,
    add_resource_node,
    add_tool_node,
    view_node_details,
    view_processor_node,
    view_resource_node,
    view_workflow,
)
from llm.utils.middleware import DisableParallelToolCallsMiddleware, TokenStatisticMiddleware
from llm.utils.middleware.dynamic_managers import ToolManagerMiddleware

RESOURCE_MANAGER_SYSTEM = """你是一名生物信息专家, 正在使用一款基于节点式编辑器的可视化生物信息分析平台。

## 编辑器说明
此编辑器中包括以下几种类型的节点:
- 资源节点: 包括加载原始数据、数据库、参考基因组, 输入变量等
- 工具节点: 生物信息分析工具或算法
- 处理器节点: 一些用于辅助数据处理的节点, 如格式转换、数据过滤、文件拷贝等
- 代码节点: 执行自定义的脚本或代码片段

你专门负责添加资源节点。当收到用户请求时, 你需要根据请求内容, 调用相应的工具来添加合适的资源节点。

## 资源节点定义
不同类型的资源节点封装的内容不同, 资源节点通常包含如下信息。带有星号的字段是添加到面板上后才有的:
- `id*`: 已经添加到面板上的节点会被自动赋予一个唯一的id
- `name`: 此工具节点的名称
- `output_handles`: 此节点提供的输出, 可以用于之后的节点使用
- `pos*`: 节点在面板上的位置

进一步细分, 我们有以下几种资源节点: 'sequence', 'file', 'db', 'genome', 'value_string'
- db: 封装某个生物信息数据库的信息, 如NCBI nr数据库
- genome: 来自NCBI的标准参考基因组数据库, 此节点需要一个前置的字符变量节点输入, 指定参考基因组的物种名称或TaxID
- sequence: 针对双端测序文件的资源节点
- file: 通用文件输入节点
- value_string: 封装一个字符串变量, 可以用于其他节点的值输入
- global_file: 这是一个特殊类型的节点, 除非用户直接说明, 否则你不应当使用它



## 任务流程
1. 首先, 使用`view_workflow`工具查看当前面板上的节点状态
2. 针对你要使用的资源节点, 调用 `view_resource_node` 工具, 查看该节点的详细信息和要求
3. 调用 `add_resource_node` 工具, 将选中的工具节点添加到面板上, 选择合适和位置并连接好节点输入
4. 针对文件输入和序列输入类型的节点, 添加一个note节点说明此节点的输入是什么
5. 无论任务成功与否, 总结你的操作并结束工作

## 注意事项
- 你是一个大的工作流构建团队中的工具专家, 你只需要添加当前要求你添加的对应资源节点就行
- 用户不知道你有哪些类型的节点, 所以他们的要求可能不是最优选择。比如用户不知道有专门的genome节点, 可能会要求添加多个file节点, 此时你应该自行选择最合适的节点并告知用户
- 节点的位置安排:
    - 当前面板上没有节点时, 新添加的节点放在面板中央位置 (x=0, y=0)
    - 当前面板上已有节点时, 按照分析执行的顺序, 将新添加的节点放在右侧 (x+1), y坐标与相关联的上游节点保持一致
    - 如果当前位置已经有节点了, 视情况向上或向下调整 y 坐标 (±1), 避免节点重叠
"""

PROCESSOR_MANAGER_SYSTEM = """你是一名生物信息专家, 正在使用一款基于节点式编辑器的可视化生物信息分析平台。

## 编辑器说明
此编辑器中包括以下几种类型的节点:
- 资源节点: 包括加载原始数据、数据库、参考基因组, 输入变量等
- 工具节点: 生物信息分析工具或算法
- 处理器节点: 一些用于辅助数据处理的节点, 如格式转换、数据过滤、文件拷贝等
- 代码节点: 执行自定义的脚本或代码片段

你专门负责添加处理器节点。当收到用户请求时, 你需要根据请求内容, 调用相应的工具来添加合适的处理器节点。

## 处理器节点定义
处理器节点封装的是一些辅助工作流运行的操作, 每一个节点包含如下信息。带有星号的字段是添加到面板上后才有的:
- `id*`: 已经添加到面板上的节点会被自动赋予一个唯一的id
- `name`: 此节点的名称
- `description`: 对于此节点功能的描述
- `input_handles`: 节点所需的输入连接, 来自之前的节点
- `output_handles`: 此节点提供的输出, 可以用于之后的节点使用
- `pos*`: 节点在面板上的位置

## 任务流程
1. 首先, 使用`view_workflow`工具查看当前面板上的节点状态
2. 根据你收到的要求, 调用 `search_processors` 搜索对应功能的处理器节点
3. 调用 `view_processor_node` 工具, 查看该节点的详细信息和要求
4. 调用 `add_processor_node` 工具, 将选中的工具节点添加到面板上, 选择合适和位置并连接好节点输入
6. 无论任务成功与否, 总结你的操作并结束工作

## 注意事项
- 你是一个大的工作流构建团队中的工具专家, 你只需要添加当前要求你添加的对应处理器节点就行
- 节点的位置安排:
    - 当前面板上没有节点时, 新添加的节点放在面板中央位置 (x=0, y=0)
    - 当前面板上已有节点时, 按照分析执行的顺序, 将新添加的节点放在右侧 (x+1), y坐标与相关联的上游节点保持一致
    - 如果当前位置已经有节点了, 视情况向上或向下调整 y 坐标 (±1), 避免节点重叠
- 由于中间处理情况比较复杂, 如果确定数据库中没有找到相关处理器节点时, 调用 `add_code_node` 工具直接添加一个bash类型的代码节点来实现该功能
"""

CODE_MANAGER_SYSTEM = """你是一名生物信息专家, 正在使用一款基于节点式编辑器的可视化生物信息分析平台。

## 编辑器说明
此编辑器中包括以下几种类型的节点:
- 资源节点: 包括加载原始数据、数据库、参考基因组, 输入变量等
- 工具节点: 生物信息分析工具或算法
- 处理器节点: 一些用于辅助数据处理的节点, 如格式转换、数据过滤、文件拷贝等
- 代码节点: 执行自定义的脚本或代码片段

你专门负责添加代码节点。当收到用户请求时, 你需要根据请求内容, 调用相应的工具来添加合适的代码节点。

## 代码节点节点定义
代码节点节点封装的是python、R或BASH脚本, 每一个节点包含如下信息。带有星号的字段是添加到面板上后才有的:
- `id*`: 已经添加到面板上的节点会被自动赋予一个唯一的id
- `name`: 此节点的名称
- `input_handles`: 代码节点只有一个输入input_files, 只要是连接到此输入的文件, 都会被复制到代码工作空间的raw_data文件夹下
- `output_handles`: 代码节点只有一个输出output_files, 映射的是代码工作空间下的output文件夹。
- `pos*`: 节点在面板上的位置

## 任务流程
1. 首先, 使用`view_workflow`工具查看当前面板上的节点状态
2. 如果需要, 调用 `get_paper_snippets` 工具以进一步了解原文中对于这里分析代码的需求
3. 调用 `add_code_node` 工具, 添加代码节点。注意你只是负责添加节点, 因此只需要通过`program_request`参数传入详细的代码需求描述即可, 具体的代码写作会由后续的代码编写工具完成
4. 无论任务成功与否, 总结你的操作并结束工作

## 注意事项
- 你是一个大的工作流构建团队中的工具专家, 你只需要添加当前要求你添加的对应代码节点就行
- 节点的位置安排:
    - 当前面板上没有节点时, 新添加的节点放在面板中央位置 (x=0, y=0)
    - 当前面板上已有节点时, 按照分析执行的顺序, 将新添加的节点放在右侧 (x+1), y坐标与相关联的上游节点保持一致
    - 如果当前位置已经有节点了, 视情况向上或向下调整 y 坐标  (±1), 避免节点重叠
"""


@tool(parse_docstring=True)
async def create_tool_node(request: str, runtime: ToolRuntime[BuildContext]) -> Command:
    """Create new tool node.

    **This tool prohibits parallel calls! You are only allowed to request the creation of one tool at a time!**

    Args:
        request: Node creation requirements. Format: "Using XXX (tool name) to XXX (specific task description, including what you want to do with this tool, what the tool's input is, and what the main expected output is)"
    """
    writer = get_stream_writer()
    writer({'info': f'Creating new tool: {request}'})

    sub_agent = CreateToolAgent().build()
    agent_input: CreateToolState = {'messages': [HumanMessage(content=f'<Require>\n{request}\n</Require>')]}
    run_id = nanoid.generate(size=8)
    context = CreateToolContext(user_id=runtime.context.user_id, auto_save=True, test_run_id=run_id)
    response = await sub_agent.ainvoke(agent_input, context=context)  # type: ignore

    return Command(
        update={
            'id_mapper': response['id_mapper'],
            'messages': [ToolMessage(response['messages'][-1].content, tool_call_id=runtime.tool_call_id)],
        }
    )


@tool('need_other_node', parse_docstring=True)
async def tool_need_other_node(
    node_type: Literal['resource', 'processor', 'code'], requirement: str, runtime: ToolRuntime[BuildContext]
) -> Command:
    """Call this tool to request adding a resource node, processor node, or code node.

    Args:
        node_type: Type of node to add, can be `resource`, `processor`, or `code`
        requirement: Specific requirements for the node to be added, as brief and precise as possible. For example, "add an NCBI nr database node in DIAMOND format at (x=1, y=2)"
    """
    exist_nodes = runtime.state.get('exist_nodes', [])
    exist_edges = runtime.state.get('exist_edges', [])
    global_marks = runtime.state.get('global_marks', [])

    sub_agent_input: ManagerSchema = {
        'messages': [HumanMessage(content=requirement)],
        'exist_nodes': exist_nodes,
        'exist_edges': exist_edges,
        'global_marks': global_marks,
        'paper_id': runtime.state['paper_id'],
    }
    sub_agent_context = BuildContext(user_id=runtime.context.user_id, has_reference=runtime.context.has_reference)

    match node_type:
        case 'resource':
            response = await resource_manager_agent.ainvoke(sub_agent_input, context=sub_agent_context)  # type: ignore
        case 'processor':
            response = await processor_manager_agent.ainvoke(sub_agent_input, context=sub_agent_context)  # type: ignore
        case 'code':
            response = await code_manager_agent.ainvoke(sub_agent_input, context=sub_agent_context)  # type: ignore
        case _:
            return f'Unknown node_type: {node_type}'

    return Command(
        update={
            'messages': [ToolMessage(content=response['messages'][-1].content, tool_call_id=runtime.tool_call_id)],
            'exist_nodes': Overwrite(response['exist_nodes']),
            'exist_edges': Overwrite(response['exist_edges']),
            'global_marks': Overwrite(response['global_marks']),
        }
    )


tool_manager_agent: CompiledStateGraph = create_agent(
    get_agent_model(),
    [
        get_paper_snippets,
        view_workflow,
        view_node_details,
        search_tool,
        create_tool_node,
        add_tool_node,
        add_note_node,
        tool_need_other_node,
        ask_human,
    ],
    middleware=[
        ToolManagerMiddleware(),
        DisableParallelToolCallsMiddleware(
            disabled_tools={
                'search_tool',
                'create_tool_node',
                'add_tool_node',
                'tool_need_other_node',
                'ask_human',
            }
        ),
        TokenStatisticMiddleware('tool_manager', **get_agent_model().metadata),
    ],
    state_schema=ManagerSchema,  # type: ignore
    context_schema=BuildContext,  # type: ignore
)

resource_manager_agent: CompiledStateGraph = create_agent(
    get_agent_model(),
    [
        view_workflow,
        view_node_details,
        view_resource_node,
        search_biodb,
        add_resource_node,
        add_note_node,
        ask_human,
    ],
    system_prompt=RESOURCE_MANAGER_SYSTEM,
    middleware=[
        DisableParallelToolCallsMiddleware(disabled_tools={'add_resource_node', 'add_note_node', 'ask_human'}),
        TokenStatisticMiddleware('resource_manager', **get_agent_model().metadata),
    ],
    state_schema=ManagerSchema,  # type: ignore
    context_schema=BuildContext,  # type: ignore
)


processor_manager_agent: CompiledStateGraph = create_agent(
    get_agent_model(),
    [
        view_workflow,
        view_node_details,
        search_processors,
        view_processor_node,
        add_processor_node,
        add_code_node,
    ],
    system_prompt=PROCESSOR_MANAGER_SYSTEM,
    middleware=[
        DisableParallelToolCallsMiddleware(disabled_tools={'add_processor_node', 'add_code_node'}),
        TokenStatisticMiddleware('processor_manager', **get_agent_model().metadata),
    ],
    state_schema=ManagerSchema,  # type: ignore
    context_schema=BuildContext,  # type: ignore
)

code_manager_agent: CompiledStateGraph = create_agent(
    get_agent_model(),
    [
        get_paper_snippets,
        view_node_details,
        view_workflow,
        add_code_node,
    ],
    system_prompt=CODE_MANAGER_SYSTEM,
    middleware=[
        DisableParallelToolCallsMiddleware(disabled_tools={'add_code_node'}),
        TokenStatisticMiddleware('code_manager', **get_agent_model().metadata),
    ],
    state_schema=ManagerSchema,  # type: ignore
    context_schema=BuildContext,  # type: ignore
)
