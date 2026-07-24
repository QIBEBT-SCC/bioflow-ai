from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from app.schemas.tools import Handle
from llm.context import ProjectContext
from llm.prebuild.code_node import (
    build_code_inputs,
    generate_bash_code,
    generate_python_code,
)
from llm.states import BuildEdge, BuildNode
from llm.utils.edge_validation import check_code_node_edges, check_edge_handles
from llm.utils.formatters import format_validation_error, manager_help_command, node_formatter
from llm.utils.uid_mapper import generate_char_index


class HelpSchema(BaseModel):
    action: str = Field(description='The action name to get detailed schema for')


class AddCodeSchema(BaseModel):
    """Add a bash code node to the workflow."""

    request: str = Field(
        min_length=1,
        description=(
            'Describe the exact transformation the Bash node must perform, including expected deterministic output '
            'file names. Do not write the script; a dedicated coding model generates it.'
        ),
    )
    input_edges: list[BuildEdge] = Field(
        default_factory=list,
        description='Code nodes have one `input_files` handle; connect files required by the code to this handle.',
    )
    pos_x: int = Field(description="The x-coordinate of the node's layout on the panel, only positive integer")
    pos_y: int = Field(description="The y-coordinate of the node's layout on the panel")


class AddPythonCodeSchema(BaseModel):
    """Add a simple single-file Python code node to the workflow."""

    request: str = Field(
        min_length=1,
        description=(
            'Describe the exact deterministic transformation the Python node must perform, including expected output '
            'file names and any required libraries. Do not write the script; a dedicated coding model generates it.'
        ),
    )
    input_edges: list[BuildEdge] = Field(
        default_factory=list,
        description=(
            'Python code nodes have one `input_files` handle. Connect each required file through a separate '
            '`bind_param` processor node.'
        ),
    )
    pos_x: int = Field(description="The x-coordinate of the node's layout on the panel, only positive integer")
    pos_y: int = Field(description="The y-coordinate of the node's layout on the panel")


class AddNoteSchema(BaseModel):
    """Add a note node to the workflow."""

    note_content: str = Field(description='Specific content of the notes (as brief and precise as possible)')
    pos_x: int = Field(description="The x-coordinate of the node's layout on the panel, only positive integer")
    pos_y: int = Field(description="The y-coordinate of the node's layout on the panel")


class AddSummarySchema(BaseModel):
    """Add a downstream summary node."""

    prompt: str = Field(description='Guidance for downstream analysis requirements.')
    input_edges: list[BuildEdge] = Field(
        default_factory=list,
        description='The node has one `input_files` handle; connect every file required for downstream analysis.',
    )
    pos_x: int = Field(description="The x-coordinate of the node's layout on the panel, only positive integer")
    pos_y: int = Field(description="The y-coordinate of the node's layout on the panel")


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'add_note': AddNoteSchema,
    'add_code': AddCodeSchema,
    'add_python_code': AddPythonCodeSchema,
    'add_downstream_summary': AddSummarySchema,
}

HELP_MESSAGE = (
    'Call other_node_manager with action="help" and params={"action": "<action_name>"} '
    'to obtain the parameter schema for a specific action.'
)


@tool(parse_docstring=True)
async def other_node_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[ProjectContext]) -> Command:
    """Code and Note manager, execute operations related to code nodes and note nodes.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    match action:
        case 'help':
            # pyrefly: ignore [bad-argument-type]
            return manager_help_command(params, HelpSchema, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)

        case 'add_code':
            try:
                validate_param = AddCodeSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('add_code', e, AddCodeSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return await _add_code(
                validate_param.request,
                validate_param.pos_x,
                validate_param.pos_y,
                validate_param.input_edges,
                runtime,
            )

        case 'add_python_code':
            try:
                validate_param = AddPythonCodeSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('add_python_code', e, AddPythonCodeSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return await _add_python_code(
                validate_param.request,
                validate_param.pos_x,
                validate_param.pos_y,
                validate_param.input_edges,
                runtime,
            )

        case 'add_note':
            try:
                validate_param = AddNoteSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('add_note', e, AddNoteSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _add_note_node(validate_param.note_content, validate_param.pos_x, validate_param.pos_y, runtime)

        case 'add_downstream_summary':
            try:
                validate_param = AddSummarySchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('add_downstream_summary', e, AddSummarySchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _add_downstream_node(
                validate_param.prompt,
                validate_param.pos_x,
                validate_param.pos_y,
                validate_param.input_edges,
                runtime,
            )
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


async def _add_code(
    request: str,
    pos_x: int,
    pos_y: int,
    input_edges: list[BuildEdge],
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    code_input_handles = [Handle(name='input_files', description='The files required by this code')]

    if error := check_code_node_edges(input_edges, exist_nodes):
        return Command(update={'messages': [ToolMessage(content=error, tool_call_id=runtime.tool_call_id)]})

    if error := check_edge_handles(input_edges, code_input_handles, exist_nodes):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})

    try:
        code_inputs = build_code_inputs(input_edges, exist_nodes, runtime.state.get('exist_edges', []))
        generated = await generate_bash_code(request, code_inputs, runtime.context.user_id)
    except Exception as error:
        logger.exception('Bash code-node generation failed')
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'Error: Bash code generation failed: {error}',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    node_id = generate_char_index(len(exist_nodes) + 1)

    new_node = BuildNode(
        id=node_id,
        name='Bash Code',
        type='code_bash',
        input_handles=[Handle(name='input_files', description='The files required by this code')],
        output_handles=[Handle(name='output_folder', description='The files generated by this code')],
        data={'code': generated.code},
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
                    content=f'add new code node:\n{node_formatter(new_node)}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


async def _add_python_code(
    request: str,
    pos_x: int,
    pos_y: int,
    input_edges: list[BuildEdge],
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    code_input_handles = [Handle(name='input_files', description='The files required by this Python code')]

    if error := check_code_node_edges(input_edges, exist_nodes):
        return Command(update={'messages': [ToolMessage(content=error, tool_call_id=runtime.tool_call_id)]})

    if error := check_edge_handles(input_edges, code_input_handles, exist_nodes):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})

    try:
        code_inputs = build_code_inputs(input_edges, exist_nodes, runtime.state.get('exist_edges', []))
        generated = await generate_python_code(request, code_inputs, runtime.context.user_id)
    except Exception as error:
        logger.exception('Python code-node generation failed')
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'Error: Python code generation failed: {error}',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    node_id = generate_char_index(len(exist_nodes) + 1)
    new_node = BuildNode(
        id=node_id,
        name='Python Code',
        type='code_python',
        input_handles=code_input_handles,
        output_handles=[Handle(name='output_folder', description='The files generated by this Python code')],
        data={'code': generated.code, 'dependencies': generated.dependencies},
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
                    content=f'add new Python code node:\n{node_formatter(new_node)}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

def _add_note_node(note_content: str, pos_x: int, pos_y: int, runtime: ToolRuntime[ProjectContext]) -> Command:
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)

    new_node = BuildNode(
        id=node_id,
        name='note',
        type='note',
        input_handles=[],
        output_handles=[],
        data={'content': note_content},
        x=pos_x,
        y=pos_y,
    )

    return Command(
        update={
            'exist_nodes': [new_node],
            'messages': [
                ToolMessage(
                    content=f'add new note:\n{node_formatter(new_node)}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _add_downstream_node(
    prompt: str,
    pos_x: int,
    pos_y: int,
    input_edges: list[BuildEdge],
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    input_handles = [Handle(name='input_files', description='The files required for downstream analyse')]

    if error := check_edge_handles(input_edges, input_handles, exist_nodes):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})

    if not input_edges:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Error: The `downstream_summary` node must connect the files to be used in downstream analysis to `input_files`. Currently, no files are connected.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    node_id = generate_char_index(len(exist_nodes) + 1)
    new_node = BuildNode(
        id=node_id,
        name='downstream_summary',
        type='downstream_summary',
        input_handles=input_handles,
        output_handles=[],
        data={'prompt': prompt},
        x=pos_x,
        y=pos_y,
    )

    node_name_map = {node.id: node.name for node in exist_nodes}
    used_files = [f'- {node_name_map[e.source_node]}: {e.source_handle}' for e in input_edges]

    return Command(
        update={
            'exist_nodes': [new_node],
            'exist_edges': input_edges,
            'messages': [
                ToolMessage(
                    content=f'Add downstream node with these files:\n{"\n".join(used_files)}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
