from typing import Any, Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from app.schemas.tools import Handle
from llm.states import BuildContext, BuildEdge, BuildNode
from llm.utils.edge_validation import check_edge_handles
from llm.utils.formatters import format_validation_error, manager_help_command, node_formatter
from llm.utils.uid_mapper import generate_char_index

# ---- Schemas ----


class HelpSchema(BaseModel):
    action: str = Field(description='The action name to get detailed schema for')


class AddCodeSchema(BaseModel):
    """Add a code node to the workflow."""

    language: Literal['R', 'python', 'bash'] = Field(description='Program language (R, python, linux bash)')
    pos_x: int = Field(description="The x-coordinate of the node's layout on the panel, only positive integer")
    pos_y: int = Field(description="The y-coordinate of the node's layout on the panel")
    program_request: str = Field(
        description='What functionality needs to be implemented in the code. Must be accurate, as subsequent work will directly base code writing on this description.'
    )
    input_edges: list[BuildEdge] = Field(
        default_factory=list,
        description='Code node only has one edge `files`; connect files required by the code to this handle.',
    )
    code: Optional[str] = Field(default=None, description='Specific code content; required only when language is bash.')


class AddNoteSchema(BaseModel):
    """Add a note node to the workflow."""

    note_content: str = Field(description='Specific content of the notes (as brief and precise as possible)')
    pos_x: int = Field(description="The x-coordinate of the node's layout on the panel, only positive integer")
    pos_y: int = Field(description="The y-coordinate of the node's layout on the panel")


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'add_note': AddNoteSchema,
    'add_code': AddCodeSchema,
}

HELP_MESSAGE = """other_node_manager contains the following available actions:
- `add_code`: Add a code node (R/python/bash) to the workflow panel
- `add_note`: Add a note node to the workflow panel

Use action="help" with params={"action": "<action_name>"} to get the parameter schema for a specific action.
(e.g. other_node_manager(action="help" params={"action": "add_note"}))
"""


@tool(parse_docstring=True)
async def other_node_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[BuildContext]) -> Command:
    """Code and Note manager, execute operations related to code nodes and note nodes.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    logger.info(f'Other Manager(action="{action} {params}")')
    match action:
        case 'help':
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
            return _add_code(
                validate_param.language,
                validate_param.pos_x,
                validate_param.pos_y,
                validate_param.program_request,
                validate_param.input_edges,
                runtime,
                validate_param.code,
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


def _add_code(
    language: Literal['R', 'python', 'bash'],
    pos_x: int,
    pos_y: int,
    program_request: str,
    input_edges: list[BuildEdge],
    runtime: ToolRuntime[BuildContext],
    code: Optional[str] = None,
) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    code_input_handles = [Handle(name='input_files', description='The files required by this code')]

    if language == 'bash' and input_edges:
        node_type_map = {node.id: node.type for node in exist_nodes}
        invalid_edges = [e for e in input_edges if node_type_map.get(e.source_node) != 'bind_param']
        if invalid_edges:
            invalid_node_ids = [e.source_node for e in invalid_edges]
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=(
                                f'Error: The input of a bash code node must come from `bind_param` processor nodes. '
                                f'The following source nodes are not `bind_param`: {invalid_node_ids}. '
                                f'Please refer to the USING_BASH_NODE skill for the correct usage pattern.'
                            ),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

    if error := check_edge_handles(input_edges, code_input_handles, exist_nodes):
        return Command(
            update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]}
        )

    node_id = generate_char_index(len(exist_nodes) + 1)

    new_node = BuildNode(
        id=node_id,
        name=f'{language} Code',
        type=f'code_{language}',
        input_handles=[Handle(name='input_files', description='The files required by this code')],
        output_handles=[Handle(name='output_folder', description='The files generated by this code')],
        data={'description': program_request, 'code': code},
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


def _add_note_node(note_content: str, pos_x: int, pos_y: int, runtime: ToolRuntime[BuildContext]) -> Command:
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
