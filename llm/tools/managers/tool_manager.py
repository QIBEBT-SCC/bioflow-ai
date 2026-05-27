from typing import Any, cast

import nanoid
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from sqlmodel import Session

import app.crud.tools.tool as tool_crud
from app.core.session import engine
from app.schemas.tools import ToolNodePublic
from app.utils.input_format import build_command_template
from llm.agents.tool_agent import CreateToolAgent
from llm.prebuild.chains import check_update_args
from llm.states import BuildContext, BuildEdge, BuildNode, CreateToolContext, CreateToolState
from llm.utils.edge_validation import check_edge_handles, check_tool_node_edges
from llm.utils.formatters import (
    format_validation_error,
    manager_help_command,
    node_formatter,
    search_nodes_formatter,
)
from llm.utils.retrievers import retrieve_tools
from llm.utils.uid_mapper import BidirectionalMapper, generate_char_index, merge_mappers

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


class UpdateArgsSchema(BaseModel):
    """Update the editable parameters of the specified tool node that has been added to the panel."""

    node_id: str = Field(description='Tool Node ID on the panel to be deleted')
    new_args: str = Field(description='New parameters to update')
    reason: str = Field(description='The reason you made this parameter modification')


class CreateToolSchema(BaseModel):
    """Create a new tool node."""

    request: str = Field(
        description='Node creation requirements. Format: "Using XXX (tool name) to XXX (specific task description, including what you want to do with this tool, what the input is, and what the main expected output is)"'
    )


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'search_by_name': NameSearchSchema,
    # 'search_by_function': FunctionSearchSchema,
    'add': AddToolSchema,
    'update_args': UpdateArgsSchema,
    'create': CreateToolSchema,
}

HELP_MESSAGE = """## Actions
The tool_node_manager includes the following actions:
- `search_by_name`: Search for a tool by name
- `add`: Add a tool node to the panel
- `update_args`: Update node parameters
- `create`: Create a new tool node (when no existing node meets the requirements)

Before using any action, always use action="help" and pass params={"action": "<action_name>"} to obtain parameter descriptions for the specific action.
(e.g., tool_node_manager(action="help" params={"action": "search_by_name"}))

## Explanation
How to determine whether a found tool node is suitable:
- Meets requirements
    - A usable tool is found, and its functionality perfectly matches user needs.
    - A usable tool is found, though its parameter settings do not fully match user needs; however, this can be resolved by adding or adjusting modifiable parameters (args_editable).
- Does not meet requirements
    - The search tool returns no results.
    - The search tool returns results, but none of them match user requirements.
    - A usable tool is found, but the required functionality conflicts with the tool's non-modifiable parameters (args_static).
    - A usable tool is found, but the required functionality is inconsistent with the current node's input (additional files are needed or certain inputs are unnecessary).
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
            return manager_help_command(params, HelpSchema, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)  # type: ignore
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
            return _add_tool_node(
                validate_param.selected_id,
                validate_param.input_edges,
                validate_param.pos_x,
                validate_param.pos_y,
                runtime,
            )

        case 'update_args':
            try:
                validate_param = UpdateArgsSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('update_args', e, UpdateArgsSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return await _update_tool_node(
                validate_param.node_id,
                validate_param.new_args,
                validate_param.reason,
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
        new_mapper = BidirectionalMapper(datas=exist_tools)

    if exist_tools:
        exist_mapper = cast('BidirectionalMapper', runtime.state.get('id_mapper'))
        merged_mapper = merge_mappers(exist_mapper, new_mapper)
        tool_nodes = []
        for exist_tool in exist_tools:
            try:
                tool_nodes.append(ToolNodePublic.model_validate(exist_tool.model_dump()))
            except TypeError:
                logger.warning(f'Failed to parse tool "{exist_tool.name}"({exist_tool.uid})')

        return Command(
            update={
                'id_mapper': merged_mapper,
                'messages': [
                    ToolMessage(
                        content=(
                            f'Based on the keywords, the following tool nodes are available:\n'
                            f'{search_nodes_formatter(tool_nodes, merged_mapper)}'
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
    tool_nodes, new_mapper = await retrieve_tools(query)

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

    exist_mapper = cast('BidirectionalMapper', runtime.state.get('id_mapper'))
    merged_mapper = merge_mappers(exist_mapper, new_mapper)  # type: ignore

    return Command(
        update={
            'id_mapper': merged_mapper,
            'messages': [
                ToolMessage(
                    content=search_nodes_formatter(tool_nodes, merged_mapper),
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
    tool_uid = runtime.state['id_mapper'].get_uid_by_index(selected_id)

    with Session(engine) as session:
        db_tool = tool_crud.get(session, tool_uid)
        if db_tool:
            db_node = ToolNodePublic.model_validate(db_tool.model_dump())
        else:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content='The tool corresponding to the ID was not found in the database. Please verify that the provided ID is correct, or search again to confirm whether the ID has changed.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    if error := check_edge_handles(input_edges, db_node.input_handles, exist_nodes):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})
    if error := check_tool_node_edges(input_edges, db_node.input_handles):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})

    node_id = generate_char_index(len(exist_nodes) + 1)
    new_node = BuildNode(
        id=node_id,
        name=db_node.name,
        type='tool',
        input_handles=db_node.input_handles,
        output_handles=db_node.output_handles,
        data={'tool_uid': tool_uid, 'modifiable_params': db_node.modifiable_static_params},
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


async def _update_tool_node(node_id: str, new_args: str, reason: str, runtime: ToolRuntime[BuildContext]) -> Command:
    exist_nodes = cast('list[BuildNode]', runtime.state.get('exist_nodes'))
    tool_node: BuildNode | None = next((n for n in exist_nodes if n.id == node_id), None)

    if tool_node is None:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='The provided `node_id` is not a valid panel node ID.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    if tool_node.type != 'tool':
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='This node is not a tool node; please check if the ID you provided is correct.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    tool_uid = tool_node.data['tool_uid']
    with Session(engine) as session:
        db_tool = tool_crud.get(session, tool_uid)
        tool_doc = db_tool.help_doc.content

    origin_command = tool_node.command
    if db_tool.modifiable_static_params:
        origin_command = origin_command.replace(
            db_tool.modifiable_static_params, f'[{db_tool.modifiable_static_params}]'
        )
    if db_tool.immutable_static_params:
        origin_command = origin_command.replace(db_tool.immutable_static_params, f'({db_tool.immutable_static_params})')
    check_response = await check_update_args(tool_doc, origin_command, new_args, reason, runtime.context.user_id)

    if check_response.approve:
        new_command = build_command_template(
            db_tool.command_template,
            db_tool.dynamic_params,
            db_tool.immutable_static_params,
            new_args,
            db_tool.file_mounts,
        )
        new_nodes = [
            n.model_copy(update={'data': {'tool_uid': tool_uid, 'modifiable_params': new_args}, 'command': new_command})
            if n.id == node_id
            else n
            for n in exist_nodes
        ]

        return Command(
            update={
                'exist_nodes': Overwrite(new_nodes),
                'messages': [
                    ToolMessage(
                        content='The modifiable parameters of the tool node have been updated.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    return Command(
        update={'messages': [ToolMessage(content=check_response.feedback, tool_call_id=runtime.tool_call_id)]}
    )
