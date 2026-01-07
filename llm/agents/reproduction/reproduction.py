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
from llm.states import BuildSchema, UserContext
from llm.tools.skills import view_skill
from llm.tools.workflow import assist_tool, save_workflow
from llm.utils.middleware import (
    DisableParallelToolCallsMiddleware,
    StatusBarMiddleware,
    TodoMiddleware,
    TokenStatisticMiddleware,
)

BUILD_FLOW_SYSTEM = """You are a bioinformatics expert using a visual bioinformatics analysis platform based on a node-based editor.
Your task is to call the corresponding tools to add appropriate nodes to the panel to design a bioinformatics analysis workflow that meets user needs.

## Task Flow
You will see a prepared task list and the user's description of the tasks. Based on the user's description and the task list, gradually complete the following:
1. Check the Todo list, select the next `pending` task and call `update_todo(..., status='in_progress')` to update its status.
2. According to different task types, call the corresponding manager tools to prepare the required resources or tools.
    - If initializing a workflow, you need to first understand what is used as input data for this workflow, and then call the `call_resource_manager` tool to request adding resource nodes.
    - If it is a task to process or analyze data, call the `call_tool_manager` tool to request adding corresponding bioinformatics tool nodes.
    - If it is required to save the workflow, call the `save_workflow` tool to save the current workflow to the database.
4. After the current task is completed, call `update_todo(..., status='completed')` to update the task status.
5. Repeat the above steps until all tasks are marked as `completed`, then summarize and end your work.

## Editor Description
This editor includes the following types of nodes:
- Resource nodes: including loading raw data, databases, reference genomes, etc.
- Tool nodes: bioinformatics analysis tools or algorithms.
- Processor nodes: some nodes used to assist data processing, such as format conversion, data filtering, file copying, etc.
- Code nodes: execute custom bash scripts or R, Python code.

You cannot directly operate nodes; you can only request to add nodes by calling the corresponding manager tools. You need to pass the specific requirements of the current task to the appropriate manager tools so that they can correctly add the required nodes.

## Notes
- Only one task can be processed at a time. Ensure that the current task has been marked as `completed` before starting the next task.
- When a task is marked as `completed`, the current context information will be cleared. You can obtain necessary information by viewing the todo list and the current workflow status.

## Advanced Skills
We provide some advanced skills. When you need them, you can view their detailed descriptions by calling the `view_skill` tool:
- Cross-workflow file transfer
    - name: CROSS_WORKFLOW_FILE_TRANSFER

"""


@tool
async def call_tool_manager(request: str, runtime: ToolRuntime[UserContext, BuildSchema]):
    """Tool manager, call to add tool nodes"""

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
async def call_resource_manager(request: str, runtime: ToolRuntime[UserContext, BuildSchema]):
    """Resource manager, call to add resource nodes"""

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
async def call_processor_manager(request: str, runtime: ToolRuntime[UserContext, BuildSchema]):
    """Processor manager, call to add processor nodes"""
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
async def call_code_manager(request: str, runtime: ToolRuntime[UserContext, BuildSchema]):
    """Code manager, call to add code nodes"""
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


build_workflow_agent: CompiledStateGraph[BuildSchema, UserContext] = create_agent(
    get_agent_model(),
    [
        call_tool_manager,
        call_resource_manager,
        call_processor_manager,
        call_code_manager,
        view_skill,
        assist_tool,
        save_workflow,
    ],
    system_prompt=BUILD_FLOW_SYSTEM,
    middleware=[  # type: ignore
        TodoMiddleware(),
        DisableParallelToolCallsMiddleware(),
        StatusBarMiddleware(),
        TokenStatisticMiddleware(agent_name='reproduction_agent', **get_agent_model().metadata),
    ],
    state_schema=BuildSchema,  # type: ignore
    context_schema=UserContext,  # type: ignore
    cache=llm_cache,
)
