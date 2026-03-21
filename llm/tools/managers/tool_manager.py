from typing import Any

import nanoid
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from sqlmodel import Session

import app.crud.tools.tool as tool_crud
from app.core.session import engine
from app.schemas.tools import ToolNodePublic
from llm.agents.tool_manager import CreateToolAgent
from llm.states import BuildContext, BuildEdge, BuildNode, CreateToolContext, CreateToolState
from llm.utils.formatters import (
    format_validation_error,
    manager_help_command,
    node_formatter,
    search_nodes_formatter,
)
from llm.utils.retrievers import retrieve_tools
from llm.utils.uid_mapper import BidirectionalMapper, generate_char_index

# ---- Schemas ----


class HelpSchema(BaseModel):
    action: str = Field(description='The action name to get detailed schema for')


class NameSearchSchema(BaseModel):
    """Search for tool nodes by name."""

    query: str = Field(description='Name of the tool to search for')


class FunctionSearchSchema(BaseModel):
    """Search for tool nodes by function."""

    query: str = Field(description='Tool functionality to be queried described in natural language')


class AddToolSchema(BaseModel):
    """Add tool nodes to the workflow by tool ID."""

    selected_id: int = Field(description='The tool ID of the corresponding node source')
    input_edges: list[BuildEdge] = Field(
        default_factory=list,
        description='Input edge connection list for this node. Each input handle allows only one edge to be connected.',
    )
    pos_x: int = Field(description="The x-coordinate of the node's layout on the panel, only positive integer")
    pos_y: int = Field(description="The y-coordinate of the node's layout on the panel")


class CreateToolSchema(BaseModel):
    """Create a new tool node."""

    request: str = Field(
        description='Node creation requirements. Format: "Using XXX (tool name) to XXX (specific task description, including what you want to do with this tool, what the input is, and what the main expected output is)"'
    )


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'search_by_name': NameSearchSchema,
    # 'search_by_function': FunctionSearchSchema,
    'add': AddToolSchema,
    'create': CreateToolSchema,
}

HELP_MESSAGE = """## Actions
tool_node_manager 中包括了以下操作:
- `search_by_name`: 基于名称搜索工具
- `add`: 向面板添加一个工具节点
- `create`: 新建一个工具节点 (当没有满足需求的节点时)

在使用对应操作前, 务必使用 action="help" 并传入 params={"action": "<action_name>"} 以获取特定操作的参数说明
(e.g. tool_node_manager(action="help" params={"action": "search_by_name"}))

## 说明
怎样判断搜索到的工具节点是否合适:
- 满足需求
    - 找到了可用的工具, 且其功能完美符合用户需求
    - 找到了可用工具, 虽然参数设置不完全符合用户需求, 但是可以通过增加或调整可修改参数来实现
- 不满足需求
    - 搜索工具返回结果为空
    - 搜索工具返回了结果, 但是其中完全没有符合用户要求的工具
    - 找到了可用工具, 但是需要实现的需求与工具的不可修改参数(args_static)冲突
    - 找到了可用工具, 但是需要实现的功能与当前节点的输入不一致 (需要额外的文件或者不需要某些输入)
"""


@tool(parse_docstring=True)
async def tool_node_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[BuildContext]) -> Command:
    """Tool manager, execute operations related to tool nodes.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    match action:
        case 'help':
            logger.info(f'Tool Manager(action="{action} {params}")')
            return manager_help_command(params, HelpSchema, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)
        case 'search_by_name':
            try:
                validate_param = NameSearchSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('search_by_name', e, NameSearchSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            logger.info(f'Tool Manager(action="{action}" query="{validate_param.query}")')
            return _search_tool_by_name(validate_param.query, runtime)

        case 'search_by_function':
            try:
                validate_param = FunctionSearchSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('search_by_function', e, FunctionSearchSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            logger.info(f'Tool Manager(action="{action}" query="{validate_param.query}")')
            return await _search_tools_by_function(validate_param.query, runtime)

        case 'add':
            try:
                validate_param = AddToolSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('add', e, AddToolSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            logger.info(f'Tool Manager(action="{action}" selected_id="{validate_param.selected_id}")')
            return _add_tool_node(
                validate_param.selected_id,
                validate_param.input_edges,
                validate_param.pos_x,
                validate_param.pos_y,
                runtime,
            )

        case 'create':
            try:
                validate_param = CreateToolSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('create', e, CreateToolSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            logger.info(f'Tool Manager(action="{action}" request="{validate_param.request}")')
            writer = get_stream_writer()
            writer({'info': f'Creating new tool: {validate_param.request}'})

            sub_agent = CreateToolAgent().build()
            agent_input: CreateToolState = {
                'messages': [HumanMessage(content=f'<Require>\n{validate_param.request}\n</Require>')]
            }
            run_id = nanoid.generate(size=8)
            context = CreateToolContext(user_id=runtime.context.user_id, auto_save=True, test_run_id=run_id)
            response = await sub_agent.ainvoke(agent_input, context=context)  # type: ignore

            return Command(
                update={
                    'id_mapper': response['id_mapper'],
                    'messages': [ToolMessage(response['messages'][-1].content, tool_call_id=runtime.tool_call_id)],
                }
            )

        case _:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'Unknown action: "{action}". Use action="help" with params={{}} to see available actions.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )


def _search_tool_by_name(query: str, runtime: ToolRuntime[BuildContext]) -> Command:
    with Session(engine) as session:
        exist_tools = tool_crud.get_by_name_fuzzy(session, query, 15, 0)
        mappers = BidirectionalMapper(datas=exist_tools)

    if exist_tools:
        tool_nodes = []
        for exist_tool in exist_tools:
            try:
                tool_nodes.append(ToolNodePublic.model_validate(exist_tool.model_dump()))
            except TypeError:
                logger.warning(f'Failed to parse tool "{exist_tool.name}"({exist_tool.uid})')

        return Command(
            update={
                'id_mapper': mappers,
                'messages': [
                    ToolMessage(
                        content=(
                            f'Based on the keywords, the following tool nodes are available '
                            f'(each tool node has an additional <command> field indicating the command executed):\n'
                            f'{search_nodes_formatter(tool_nodes, mappers)}\n'
                            f'Note that tool_id is only valid for this search; if search is called again, previous IDs become invalid.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=(
                        'Based on the keywords you provided, no results found. '
                        'Consider using new search keywords or use action="create" to create a new tool node.'
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


async def _search_tools_by_function(query: str, runtime: ToolRuntime[BuildContext]) -> Command:
    tool_nodes, mappers = await retrieve_tools(query)

    if not tool_nodes:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Based on your search query, no corresponding results found',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            'id_mapper': mappers,
            'messages': [
                ToolMessage(
                    content=search_nodes_formatter(tool_nodes, mappers),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _add_tool_node(
    selected_id: int,
    input_edges: list[BuildEdge],
    pos_x: int,
    pos_y: int,
    runtime: ToolRuntime[BuildContext],
) -> Command:
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)
    tool_uid = runtime.state['id_mapper'].get_uid_by_index(selected_id)

    with Session(engine) as session:
        db_tool = tool_crud.get(session, tool_uid)
        db_node = ToolNodePublic.model_validate(db_tool.model_dump())  # type: ignore[union-attr]

        new_node = BuildNode(
            id=node_id,
            name=db_node.name,
            type='tool',
            input_handles=db_node.input_handles,
            output_handles=db_node.output_handles,
            data={'tool_uid': tool_uid, 'args': db_node.modifiable_static_params},
            command=db_node.complete_command,
            x=pos_x,
            y=pos_y,
        )

    for edge in input_edges:
        edge.target_node = node_id

    return Command(
        update={
            'exist_nodes': [new_node],
            'exist_edges': input_edges,
            'messages': [
                ToolMessage(
                    content=f'add new tool node:\n{node_formatter(new_node)}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
