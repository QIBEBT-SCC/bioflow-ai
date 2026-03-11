from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from app.core.redis_client import llm_cache
from llm.agents.reproduction.mamagers import (
    code_manager_agent,
    processor_manager_agent,
    resource_manager_agent,
    tool_manager_agent,
)
from llm.models import get_agent_model
from llm.states import BuildContext, BuildSchema, UserContext
from llm.tools.workflow import assist_tool, save_workflow, view_workflow
from llm.utils.middleware import (
    DisableParallelToolCallsMiddleware,
    InitedTodoMiddleware,
    SkillMiddleware,
    StatusBarMiddleware,
    TodoMiddleware,
    TokenStatisticMiddleware,
)

BUILD_FLOW_SYSTEM = """You are a bioinformatics expert using a visual bioinformatics analysis platform based on a node-based editor.
Your task is to call the corresponding tools to add appropriate nodes to the panel to design a bioinformatics analysis workflow that meets user needs.

## Task Flow
You will see a prepared plan and the user's description of the tasks. Based on the user's description and the plan, gradually complete the following:
1. Read the plan and call the `add_todos` tool to initialize the todo list.
2. Check the Todo list, select the next `pending` task and call `update_todo(..., status='in_progress')` to update its status.
3. According to different task types, call the corresponding manager tools to prepare the required resources or tools.
    - For every workflow, you need to first understand what is used as input data for this workflow, and then call the `call_resource_manager` tool to request adding resource nodes.
    - If it is a task to process or analyze data, call the `call_tool_manager` tool to request adding corresponding bioinformatics tool nodes.
    - Once an upstream workflow is built, call the `save_workflow` tool to save the current workflow to the database.
4. After the current task is completed, call `update_todo(..., status='completed')` to update the task status.
5. Repeat the above steps until all upstream workflows in the plan have been built, then summarize and end your work.

## Editor Description
This editor includes the following types of nodes:
- Resource nodes: including loading raw data, databases, reference genomes, etc.
- Tool nodes: bioinformatics analysis tools or algorithms.
- Processor nodes: some nodes used to assist data processing, such as format conversion, data filtering, file copying, etc.
- Code nodes: execute custom bash scripts or R, Python code.

You cannot directly operate nodes; you can only request to add nodes by calling the corresponding manager tools. You need to pass the specific requirements of the current task to the appropriate manager tools so that they can correctly add the required nodes.
Generally, you ONLY need to decide to add Resource nodes, Processor nodes, and Code nodes when reading initial data or in special scenarios within other SKILLs; in other cases, tool_manager will automatically add other types of nodes required by the tool.

## Notes
- Only one task can be processed at a time. Ensure that the current task has been marked as `completed` before starting the next task.
- When a task is marked as `completed`, the current context information will be cleared. You can obtain necessary information by viewing the todo list and the current workflow status.
- To allow the manager to focus solely on adding nodes, you need to tell them in detail what needs to be done, as only you can see the current task list and workflow status.
"""


@tool
async def call_tool_manager(request: str, runtime: ToolRuntime[BuildContext, BuildSchema]) -> Command:
    """Tool manager, call to add tool nodes."""
    exist_nodes = runtime.state.get('exist_nodes', [])
    exist_edges = runtime.state.get('exist_edges', [])
    global_marks = runtime.state.get('global_marks', [])

    response = await tool_manager_agent.ainvoke(
        {  # type: ignore
            'paper_id': runtime.state.get('paper_id', None),
            'exist_nodes': exist_nodes,
            'exist_edges': exist_edges,
            'global_marks': global_marks,
            'messages': [HumanMessage(content=request)],
        },
        context=BuildContext(user_id=runtime.context.user_id, has_reference=runtime.context.has_reference),  # type: ignore
    )

    return Command(
        update={
            'messages': [ToolMessage(content=response['messages'][-1].content, tool_call_id=runtime.tool_call_id)],
            'exist_nodes': response['exist_nodes'],
            'exist_edges': response['exist_edges'],
            'global_marks': response['global_marks'],
        }
    )


@tool
async def call_resource_manager(request: str, runtime: ToolRuntime[BuildContext, BuildSchema]) -> Command:
    """Resource manager, call to add resource nodes."""
    exist_nodes = runtime.state.get('exist_nodes', [])
    exist_edges = runtime.state.get('exist_edges', [])
    global_marks = runtime.state.get('global_marks', [])

    response = await resource_manager_agent.ainvoke(
        {  # type: ignore
            'paper_id': runtime.state.get('paper_id', None),
            'exist_nodes': exist_nodes,
            'exist_edges': exist_edges,
            'global_marks': global_marks,
            'messages': [HumanMessage(content=request)],
        },
        context=UserContext(user_id=runtime.context.user_id),  # type: ignore
    )
    return Command(
        update={
            'messages': [ToolMessage(content=response['messages'][-1].content, tool_call_id=runtime.tool_call_id)],
            'exist_nodes': response['exist_nodes'],
            'exist_edges': response['exist_edges'],
            'global_marks': response['global_marks'],
        }
    )


@tool
async def call_processor_manager(request: str, runtime: ToolRuntime[BuildContext, BuildSchema]) -> Command:
    """Processor manager, call to add processor nodes."""
    exist_nodes = runtime.state.get('exist_nodes', [])
    exist_edges = runtime.state.get('exist_edges', [])
    global_marks = runtime.state.get('global_marks', [])

    response = await processor_manager_agent.ainvoke(
        {  # type: ignore
            'paper_id': runtime.state.get('paper_id', None),
            'exist_nodes': exist_nodes,
            'exist_edges': exist_edges,
            'global_marks': global_marks,
            'messages': [HumanMessage(content=request)],
        },
        context=UserContext(user_id=runtime.context.user_id),  # type: ignore
    )
    return Command(
        update={
            'messages': [ToolMessage(content=response['messages'][-1].content, tool_call_id=runtime.tool_call_id)],
            'exist_nodes': response['exist_nodes'],
            'exist_edges': response['exist_edges'],
            'global_marks': response['global_marks'],
        }
    )


@tool
async def call_code_manager(request: str, runtime: ToolRuntime[BuildContext, BuildSchema]) -> Command:
    """Code manager, call to add code nodes."""
    exist_nodes = runtime.state.get('exist_nodes', [])
    exist_edges = runtime.state.get('exist_edges', [])
    global_marks = runtime.state.get('global_marks', [])

    response = await code_manager_agent.ainvoke(
        {  # type: ignore
            'paper_id': runtime.state.get('paper_id', None),
            'exist_nodes': exist_nodes,
            'exist_edges': exist_edges,
            'global_marks': global_marks,
            'messages': [HumanMessage(content=request)],
        },
        context=UserContext(user_id=runtime.context.user_id),  # type: ignore
    )
    return Command(
        update={
            'messages': [ToolMessage(content=response['messages'][-1].content, tool_call_id=runtime.tool_call_id)],
            'exist_nodes': response['exist_nodes'],
            'exist_edges': response['exist_edges'],
            'global_marks': response['global_marks'],
        }
    )


build_workflow_agent: CompiledStateGraph = create_agent(
    get_agent_model(),
    [
        view_workflow,
        call_tool_manager,
        call_resource_manager,
        call_processor_manager,
        call_code_manager,
        assist_tool,
        save_workflow,
    ],
    system_prompt=BUILD_FLOW_SYSTEM,
    middleware=[
        TodoMiddleware(),
        SkillMiddleware(['CROSS_WORKFLOW_FILE_TRANSFER', 'MODIFY_WITH_AWK']),
        DisableParallelToolCallsMiddleware(),
        StatusBarMiddleware(),
        TokenStatisticMiddleware(agent_name='reproduction_agent', **get_agent_model().metadata),
    ],
    state_schema=BuildSchema,  # type: ignore
    context_schema=BuildContext,  # type: ignore
    cache=llm_cache,
)
