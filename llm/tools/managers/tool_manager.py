import uuid
from typing import Any, cast

import nanoid
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from pydantic import BaseModel, Field, ValidationError
from sqlmodel import Session

import app.crud.tools.tool as tool_crud
from app.core.session import engine
from app.schemas.tools import ToolNodePublic
from app.utils.input_format import build_command_template
from llm.agents.tool_agent import CreateToolAgent
from llm.context import CreateToolContext, ProjectContext
from llm.prebuild.chains import check_update_args
from llm.states import BuildEdge, BuildNode, CreateToolState
from llm.utils.formatters import (
    format_validation_error,
    manager_help_command,
    node_formatter,
    search_nodes_formatter,
)
from llm.utils.uid_mapper import BidirectionalMapper, HasUid, generate_char_index, merge_mappers
from llm.workflow_builder.validation import check_edge_handles, check_tool_node_edges


class SearchToolSchema(BaseModel):
    """Search for tool nodes by name."""

    name: str = Field(description='Name of the tool to search for, e.g. `samtools`')


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

    node_id: str = Field(description='Tool Node ID on the panel to be updated')
    new_args: str = Field(description='New parameters to update')
    reason: str = Field(
        description=(
            'The reason for this parameter modification, including the intended parameter semantics and expected '
            'value type'
        )
    )


class CreateToolSchema(BaseModel):
    """Create a new tool node."""

    name: str = Field(description='The naming of a new node must be distinguishable from existing nodes.')
    request: str = Field(
        description=(
            'Node creation requirements. '
            'Format: "Using XXX (tool name) to XXX (specific task description, including what you want to do '
            'with this tool, what new input handle is required or what new output handle is expected)"'
            'Do not guess specific parameters or subcommands to avoid misleading.'
        )
    )


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'search': SearchToolSchema,
    'add': AddToolSchema,
    'update_args': UpdateArgsSchema,
    'create': CreateToolSchema,
}

HELP_MESSAGE = (
    'Specify the action to inspect, for example: tool_node_manager(action="help", params={"action": "search"}).'
)


class InvalidToolDefinitionError(RuntimeError):
    """Raised when a stored tool cannot be converted into a workflow node."""


@tool(parse_docstring=True)
async def tool_node_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[ProjectContext]) -> Command:
    """Tool manager, execute operations related to tool nodes.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    match action:
        case 'help':
            return manager_help_command(params, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)  # type: ignore
        case 'search':
            try:
                validate_param = SearchToolSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('search_tool', e, SearchToolSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _search_tool(validate_param.name, runtime)

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
            return await _create_tool(validate_param.name, validate_param.request, runtime)

        case _:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'Unknown action: "{action}". Check the workflow-builder system prompt for available actions.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )


async def _create_tool(name: str, request: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    writer = get_stream_writer()
    writer({'info': f'Creating new tool: {request}'})

    sub_agent = CreateToolAgent().build()
    agent_input: CreateToolState = {
        'messages': [HumanMessage(content=f'Create a new tool node named `{name}`\n<Require>\n{request}\n</Require>')]
    }
    run_id = nanoid.generate(size=8)
    context = CreateToolContext(
        user_id=runtime.context.user_id,
        auto_save=True,
        skip_search=True,
        test_run_id=run_id,
    )
    # pyrefly: ignore [no-matching-overload]
    response = await sub_agent.ainvoke(agent_input, context=context)

    new_mapper = response['id_mapper']
    exist_mapper = cast('BidirectionalMapper', runtime.state.get('id_mapper'))
    merged_mapper = merge_mappers(exist_mapper, new_mapper)
    tool_uid = response['tool_uid']

    with Session(engine) as session:
        db_tool = tool_crud.get(session, tool_uid)
        if db_tool is None:
            err_msg = f'Newly created tool with UID {tool_uid} could not be loaded from the database'
            raise LookupError(err_msg)
        tool_node = ToolNodePublic.model_validate(db_tool)

    return Command(
        update={
            'new_tools': [tool_uid],
            'id_mapper': merged_mapper,
            'messages': [
                ToolMessage(
                    content=(
                        'New tool node has been saved to the database for addition to the workflow:\n'
                        f'{search_nodes_formatter([tool_node], merged_mapper)}'
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _search_tool(name: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    with Session(engine) as session:
        exist_tools = tool_crud.get_by_name_fuzzy(session, name, 15, 0)
        new_mapper = BidirectionalMapper(datas=cast('list[HasUid]', list(exist_tools)))

    if exist_tools:
        exist_mapper = cast('BidirectionalMapper', runtime.state.get('id_mapper'))
        merged_mapper = merge_mappers(exist_mapper, new_mapper)
        tool_nodes = []
        for exist_tool in exist_tools:
            try:
                tool_nodes.append(ToolNodePublic.model_validate(exist_tool))
            except Exception as error:
                msg = f'Invalid tool definition "{exist_tool.name}" ({exist_tool.uid}): {error}'
                raise InvalidToolDefinitionError(msg) from error

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
                        'Consider first searching with relatively concise keywords (e.g., STAR), then gradually adding restrictions. '
                        'If you confirm that the desired tool cannot be found, use action="create" to create a new tool node.'
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


def _add_tool_node(
    selected_id: int,
    input_edges: list[BuildEdge],
    pos_x: int,
    pos_y: int,
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    mapper = runtime.state.get('id_mapper')
    tool_uid = mapper.get_uid_by_index(selected_id) if isinstance(mapper, BidirectionalMapper) else None

    # Search result IDs are transient indexes maintained by ``id_mapper``.  The model can
    # occasionally reuse an ID from an earlier turn, in which case the mapper returns an
    # empty string.  Never send that value to PostgreSQL's UUID column: apart from avoiding
    # an unhandled DataError, the response gives the agent a concrete recovery action.
    try:
        tool_uid = str(uuid.UUID(tool_uid)) if tool_uid else None
    except (AttributeError, TypeError, ValueError):
        tool_uid = None

    if tool_uid is None:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=(
                            f'Tool ID {selected_id} is not available in the current search results. '
                            'Call `tool_node_manager` with `action="search"` to search again, then use '
                            'the ID returned by that latest search result to add the node.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    with Session(engine) as session:
        db_tool = tool_crud.get(session, tool_uid)
        if db_tool:
            db_node = ToolNodePublic.model_validate(db_tool)
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


async def _update_tool_node(node_id: str, new_args: str, reason: str, runtime: ToolRuntime[ProjectContext]) -> Command:
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
        if db_tool is None:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content='The information corresponding to the node is not found in the database. This may be due to data corruption or other reasons. '
                            'Check whether the provided ID is correct, or consider performing a new search to refresh the database information.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )
        # pyrefly: ignore [missing-attribute]
        tool_doc: str = db_tool.help_doc.content

    origin_command = tool_node.command
    if origin_command is None:
        err_msg = 'Tool command is None.'
        raise ValueError(err_msg)

    if db_tool.modifiable_static_params:
        # pyrefly: ignore [missing-attribute]
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
