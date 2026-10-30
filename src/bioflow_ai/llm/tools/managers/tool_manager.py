import uuid
from typing import Any, cast

import nanoid
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy.orm import Session

import bioflow_ai.crud.tools.tool as tool_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.infrastructure.redis.operations import open_catalog_generation_guard
from bioflow_ai.llm.agents.tool_agent import CreateToolAgent
from bioflow_ai.llm.context import CreateToolContext, ProjectContext
from bioflow_ai.llm.middleware.frontend_progress import ProgressStatus
from bioflow_ai.llm.middleware.limit import MaxRetriesError
from bioflow_ai.llm.states import BuildEdge, BuildNode, CreateToolState
from bioflow_ai.llm.workflow_builder.graphs import workflow_document
from bioflow_ai.llm.workflow_builder.node_ids import allocate_node_id
from bioflow_ai.schemas.tools import ToolNodePublic
from bioflow_ai.services.tools.catalog import check_update_args
from bioflow_ai.services.tools.input_format import build_command_template
from bioflow_ai.services.workflow.validation import validate_workflow
from bioflow_ai.services.workflow.validation.adapters import bind_candidate_edges
from bioflow_ai.utils.formatters import (
    format_validation_error,
    manager_help_command,
    node_formatter,
    search_nodes_formatter,
)
from bioflow_ai.utils.llm.session import workflow_thread_id
from bioflow_ai.utils.uid_mapper import BidirectionalMapper, HasUid, merge_mappers

SEARCH_RESULT_LIMIT = 15


class SearchToolSchema(BaseModel):
    """Search tool nodes by their complete command, like chaining `rg -i` over every node's command line."""

    patterns: list[str] = Field(
        min_length=1,
        description=(
            'Case-insensitive regular expressions matched against the complete command of each tool node; '
            'a node is returned only when every pattern matches. Use one pattern per command fragment so that '
            'option order does not matter, e.g. `["samtools sort"]`, `["gatk", "HaplotypeCaller"]` or '
            '`["emapper", "-m cache"]`. Start with the program and subcommand, then add mode-defining options '
            'to narrow down the results.'
        ),
    )

    @field_validator('patterns')
    @classmethod
    def drop_blank_patterns(cls, patterns: list[str]) -> list[str]:
        """Remove blank patterns and require at least one meaningful pattern."""
        patterns = [pattern.strip() for pattern in patterns if pattern.strip()]
        if not patterns:
            msg = 'At least one non-blank pattern is required'
            raise ValueError(msg)
        return patterns


class AddToolSchema(BaseModel):
    """Add tool nodes to the workflow by tool ID."""

    selected_id: int = Field(description='The tool ID of the corresponding node source')
    input_edges: list[BuildEdge] = Field(
        default_factory=list,
        description='Input edge connection list for this node. Each input handle allows only one edge to be connected.',
    )


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
            # ty: ignore[invalid-argument-type]
            return manager_help_command(params, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)
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
            return await _search_tool(validate_param.patterns, runtime)

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
    wait_call_id = f'tool-generation-wait:{runtime.tool_call_id}'
    waited = False

    def publish_wait(status: ProgressStatus) -> None:
        writer({'agent_tool': {'tool_call_id': wait_call_id, 'status': status, 'tool': 'wait_tool_generation'}})

    def on_wait() -> None:
        nonlocal waited
        waited = True
        publish_wait('running')

    thread_id = workflow_thread_id(runtime.config)
    async with (
        open_catalog_generation_guard('tool', thread_id, on_wait=on_wait) as guard,
        guard.creation() as permit,
    ):
        if waited:
            publish_wait('completed')
        if not permit.allowed:
            return _stale_tool_search_command(permit.reason, runtime)

        sub_agent = CreateToolAgent().build()
        agent_input: CreateToolState = {
            'messages': [
                HumanMessage(content=f'Create a new tool node named `{name}`\n<Require>\n{request}\n</Require>')
            ]
        }
        run_id = nanoid.generate(size=8)
        context = CreateToolContext(
            user_id=runtime.context.user_id,
            language=runtime.context.language,
            project_id=runtime.context.project_id,
            auto_save=True,
            skip_search=True,
            test_run_id=run_id,
            delegate_human_requests=True,
        )
        try:
            response = await sub_agent.ainvoke(agent_input, context=context)
        except MaxRetriesError:
            logger.warning('Tool node creation exceeded the validation retry limit: {}', name)
            return _tool_creation_failed_command(name, request, runtime)

    if response.get('return_to_parent'):
        return _tool_creation_handoff_command(name, response['parent_request'], runtime)

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


def _tool_creation_failed_command(name: str, request: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    warning = (
        f'Tool node `{name}` could not be created because its generated configuration did not pass automated '
        f'validation within the allowed attempts. The unmet requirement was: {request}'
    )
    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=warning,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _tool_creation_handoff_command(
    name: str,
    parent_request: str,
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=(
                        f'Tool node `{name}` could not continue from the original request. No tool node was created.\n'
                        f'<handoff_context>\n{parent_request}\n</handoff_context>\n'
                        'Use the verified findings above to replan the workflow. If they identify alternative tools or '
                        'an atomic decomposition, search for and use or create those nodes separately, with Bash nodes '
                        'for simple glue operations where appropriate. Call `ask_human` only when a user choice or '
                        'restated requirement is still necessary. Do not retry the unchanged creation request.'
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _stale_tool_search_command(reason: str | None, runtime: ToolRuntime[ProjectContext]) -> Command:
    detail = (
        'No catalog search has been recorded for this workflow.'
        if reason == 'missing_search'
        else ('The tool catalog changed after the previous search.')
    )
    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=(
                        f'{detail} Call `tool_node_manager` with `action="search"` again before creating a tool node.'
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


async def _search_tool(patterns: list[str], runtime: ToolRuntime[ProjectContext]) -> Command:
    writer = get_stream_writer()
    thread_id = workflow_thread_id(runtime.config)
    async with (
        open_catalog_generation_guard(
            'tool',
            thread_id,
            on_wait=lambda: writer(
                {'info': 'Waiting for another tool node generation task to finish before searching...'}
            ),
        ) as guard,
        guard.search(),
    ):
        with Session(engine) as session:
            exist_tools, total = tool_crud.search_by_command(session, patterns, SEARCH_RESULT_LIMIT)
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

        truncated_hint = (
            f'\n\n{total} tool nodes matched; only the first {len(tool_nodes)} are shown. '
            'Add more patterns to narrow down the results.'
            if total > len(tool_nodes)
            else ''
        )
        return Command(
            update={
                'id_mapper': merged_mapper,
                'messages': [
                    ToolMessage(
                        content=(
                            f'The following tool nodes have commands matching all patterns:\n'
                            f'{search_nodes_formatter(tool_nodes, merged_mapper)}{truncated_hint}'
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
                        'No tool node command matches all patterns. '
                        'Remove some patterns or search for the program name alone (e.g., ["STAR"]), '
                        'then add restrictions gradually. '
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

    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    node_id, next_node_index = allocate_node_id(exist_nodes, runtime.state.get('next_node_index'))
    new_node = BuildNode(
        id=node_id,
        name='Tool',
        type='tool',
        input_handles=[],
        output_handles=[],
        data={'tool_uid': tool_uid},
    )
    input_edges = bind_candidate_edges(node_id, input_edges)
    document = workflow_document(
        runtime.state,
        nodes=[*exist_nodes, new_node],
        edges=[*runtime.state.get('exist_edges', []), *input_edges],
        project_id=runtime.context.project_id,
    )
    result = validate_workflow(document)
    if not result.valid:
        return Command(update={'messages': [ToolMessage(content=result.message, tool_call_id=runtime.tool_call_id)]})
    db_node = (document.catalog or {})[str(uuid.UUID(tool_uid))].to_node()
    new_node = new_node.model_copy(
        update={
            'name': db_node.name,
            'input_handles': db_node.input_handles,
            'output_handles': db_node.output_handles,
            'data': {'tool_uid': tool_uid, 'modifiable_params': db_node.modifiable_static_params},
            'command': db_node.complete_command,
        }
    )

    return Command(
        update={
            'exist_nodes': [new_node],
            'exist_edges': input_edges,
            'next_node_index': next_node_index,
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
        # ty: ignore[unresolved-attribute]
        tool_doc: str = db_tool.help_doc.content

    origin_command = tool_node.command
    if origin_command is None:
        err_msg = 'Tool command is None.'
        raise ValueError(err_msg)

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

        result = validate_workflow(
            workflow_document(
                runtime.state,
                nodes=new_nodes,
                project_id=runtime.context.project_id,
            )
        )
        if not result.valid:
            return Command(
                update={'messages': [ToolMessage(content=result.message, tool_call_id=runtime.tool_call_id)]}
            )

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
