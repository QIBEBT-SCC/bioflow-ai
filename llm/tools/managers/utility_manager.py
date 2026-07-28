import json
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

import app.crud.code as code_crud
from app.core.session import engine
from app.schemas.code import CodeCreate, CodeNodeType
from app.schemas.tools import Handle
from app.services import code_catalog
from app.services.code_catalog import CODE_NODE_KIND
from llm.context import ProjectContext
from llm.prebuild.code_node import (
    build_code_inputs,
    generate_bash_code,
    generate_python_code,
)
from llm.states import BuildEdge, BuildNode
from llm.utils.formatters import format_validation_error, manager_help_command, node_formatter
from llm.utils.node_schema import PROCESSORS
from llm.utils.retrievers import retrieve_auxiliary_nodes
from llm.utils.uid_mapper import generate_char_index
from llm.workflow_builder.validation import (
    check_code_node_edges,
    check_edge_handles,
    validate_mark_storage_conflict,
    validate_node_addition,
)

# ---- Schemas ----


class SearchSchema(BaseModel):
    """Search reusable CODE definitions and processors by functionality."""

    query: str = Field(description='Briefly describe the CODE node, processor, or function to be searched for')


class ViewSchema(BaseModel):
    """View one processor or reusable CODE definition returned by search."""

    node_id: str = Field(description='Processor name or reusable CODE UID returned by search')


class AddProcessorSchema(BaseModel):
    """Add a processor node to the workflow."""

    processor_name: str = Field(description='Processor name')
    arguments: dict[str, Any] | None = Field(default=None, description='Parameters of the processor node')
    input_edges: list[BuildEdge] = Field(default_factory=list, description='Input edge connection list for this node')
    pos_x: int = Field(description="The x-coordinate of the node's layout on the panel, only positive integer")
    pos_y: int = Field(description="The y-coordinate of the node's layout on the panel")


class AddExistCodeSchema(BaseModel):
    """Add an existing reusable CODE definition returned by search."""

    code_uid: str = Field(description='UID of the reusable CODE definition returned by search')
    input_edges: list[BuildEdge] = Field(
        default_factory=list,
        description='Connect each required bind_param output to the CODE node input_files handle.',
    )
    pos_x: int = Field(description="The x-coordinate of the node's layout on the panel, only positive integer")
    pos_y: int = Field(description="The y-coordinate of the node's layout on the panel")


class AddNewCodeSchema(BaseModel):
    """Generate, persist, and add a new Bash or Python CODE node."""

    code_type: Literal['bash', 'python'] = Field(description='CODE implementation language')
    request: str = Field(
        min_length=1,
        description=(
            'Describe the exact deterministic transformation the CODE node must perform, including expected output '
            'file names and, for Python, any required libraries. Do not write the code; a dedicated coding model '
            'generates it.'
        ),
    )
    input_edges: list[BuildEdge] = Field(
        default_factory=list,
        description=(
            'CODE nodes have one `input_files` handle. Connect each required file through a separate '
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
    'search': SearchSchema,
    'view': ViewSchema,
    'add_processor': AddProcessorSchema,
    'add_exist_code': AddExistCodeSchema,
    'add_new_code': AddNewCodeSchema,
    'add_note': AddNoteSchema,
    'add_downstream_summary': AddSummarySchema,
}

HELP_MESSAGE = (
    'Call utility_node_manager with action="help" and params={"action": "<action_name>"} '
    'to obtain the parameter schema for a specific action.'
)


@tool(parse_docstring=True)
async def utility_node_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[ProjectContext]) -> Command:
    """Manage processors, reusable or generated CODE nodes, notes, and summaries.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    match action:
        case 'help':
            # pyrefly: ignore [bad-argument-type]
            return manager_help_command(params, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)

        case 'search':
            try:
                validate_param = SearchSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('search', e, SearchSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return await _search_nodes(validate_param.query, runtime)

        case 'view':
            try:
                validate_param = ViewSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('view', e, ViewSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _view_node(validate_param.node_id, runtime)

        case 'add_processor':
            try:
                validate_param = AddProcessorSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('add_processor', e, AddProcessorSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _add_processor(
                validate_param.processor_name,
                validate_param.arguments,
                validate_param.input_edges,
                validate_param.pos_x,
                validate_param.pos_y,
                runtime,
            )

        case 'add_exist_code':
            try:
                validate_param = AddExistCodeSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('add_exist_code', e, AddExistCodeSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _add_existing_code(
                validate_param.code_uid,
                validate_param.input_edges,
                validate_param.pos_x,
                validate_param.pos_y,
                runtime,
            )

        case 'add_new_code':
            try:
                validate_param = AddNewCodeSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('add_new_code', e, AddNewCodeSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return await _add_new_code(
                validate_param.code_type,
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


async def _search_nodes(query: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    results = await retrieve_auxiliary_nodes(query)

    if not results:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Based on your search query, no corresponding results found',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    processors: list[dict[str, Any]] = []
    codes: list[dict[str, Any]] = []
    for result in results:
        kind = result.metadata.get('kind')
        if kind == CODE_NODE_KIND or 'code_uid' in result.metadata:
            codes.append(
                {
                    'code_uid': result.metadata['code_uid'],
                    'node_type': result.metadata['node_type'],
                    'description': result.page_content,
                }
            )
        else:
            processors.append(
                {
                    'name': result.metadata['name'],
                    'description': result.page_content,
                }
            )

    content = f'<UtilityNodes>\n{json.dumps({"processors": processors, "codes": codes})}\n</UtilityNodes>'
    return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})


def _view_node(node_id: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    if node_id in PROCESSORS:
        processor_schema = PROCESSORS[node_id]
        node_obj = {
            'kind': 'processor',
            'name': node_id,
            'type': node_id,
            'description': processor_schema.get('description', ''),
            'input_handles': [
                {'name': h.name, 'description': h.description} for h in processor_schema['input_handles']
            ],
            'output_handles': [
                {'name': h.name, 'description': h.description} for h in processor_schema['output_handles']
            ],
        }
        return Command(
            update={'messages': [ToolMessage(content=json.dumps(node_obj), tool_call_id=runtime.tool_call_id)]}
        )

    with Session(engine) as session:
        try:
            code = code_crud.get(session, node_id)
        except ValueError:
            code = None

        if code is None:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'Error: No processor or reusable CODE definition found for {node_id}.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        try:
            node_type = CodeNodeType(code.node_type)
        except ValueError:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'Error: Reusable CODE definition {node_id} has unsupported type {code.node_type}.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        language = 'bash' if node_type is CodeNodeType.BASH else 'python'
        skill_name = 'USING_BASH_NODE' if node_type is CodeNodeType.BASH else 'USING_PYTHON_NODE'
        code_content = code.code if code.code.endswith('\n') else f'{code.code}\n'
        content = (
            f'Code:\n\n'
            f'```{language}\n{code_content}```\n\n'
            f'If this node meets the requirement, refer to the **{skill_name}** skill to add the CODE node.'
        )
    return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})


def _add_processor(
    processor_name: str,
    arguments: dict[str, Any] | None,
    input_edges: list[BuildEdge],
    pos_x: int,
    pos_y: int,
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    node_id = generate_char_index(len(exist_nodes) + 1)

    if processor_name not in PROCESSORS:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Error: Unknown processor name.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    processor_schema = PROCESSORS[processor_name]['args']
    if arguments is None:
        arguments = {}
    try:
        processor_schema.model_validate(arguments)
    except ValidationError as e:
        error_details = []
        for error in e.errors():
            field = ' -> '.join(str(loc) for loc in error['loc'])
            msg = error['msg']
            error_details.append(f'  - {field}: {msg}')
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f"Error: Invalid arguments for processor '{processor_name}':\n"
                        + '\n'.join(error_details),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    if error := validate_node_addition(
        runtime.state,
        node_type=processor_name,
        arguments=arguments,
    ):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})

    if error := validate_mark_storage_conflict(processor_name, arguments, runtime):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})

    if error := check_edge_handles(input_edges, PROCESSORS[processor_name]['input_handles'], exist_nodes):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})

    # Call custom edge_check if defined in schema
    if edge_check := PROCESSORS[processor_name].get('edge_check'):  # noqa: SIM102
        if error := edge_check(input_edges, PROCESSORS[processor_name]['input_handles']):
            return Command(
                update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]}
            )

    new_node = BuildNode(
        id=node_id,
        name=PROCESSORS[processor_name]['name'],
        type=processor_name,
        input_handles=PROCESSORS[processor_name]['input_handles'],
        output_handles=PROCESSORS[processor_name]['output_handles'],
        data=arguments,
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
                    content=f'add new processor node:\n{node_formatter(new_node)}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _add_existing_code(
    code_uid: str,
    input_edges: list[BuildEdge],
    pos_x: int,
    pos_y: int,
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    input_handles = [Handle(name='input_files', description='The files required by this code')]

    if error := check_code_node_edges(input_edges, exist_nodes):
        return Command(update={'messages': [ToolMessage(content=error, tool_call_id=runtime.tool_call_id)]})
    if error := check_edge_handles(input_edges, input_handles, exist_nodes):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})

    with Session(engine) as session:
        try:
            code = code_crud.get(session, code_uid)
        except ValueError:
            code = None
        if code is None:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'Error: No reusable CODE definition found for UID {code_uid}.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        try:
            node_type = CodeNodeType(code.node_type)
        except ValueError:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'Error: Reusable CODE definition {code_uid} has unsupported type {code.node_type}.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        code_description = code.description
        code_content = code.code
        dependencies = list(code.dependencies)

    node_id = generate_char_index(len(exist_nodes) + 1)
    code_name = 'Bash Code' if node_type is CodeNodeType.BASH else 'Python Code'
    node_data: dict[str, Any] = {
        'code': code_content,
        'description': code_description,
    }
    if node_type is CodeNodeType.PYTHON:
        node_data['dependencies'] = dependencies

    new_node = BuildNode(
        id=node_id,
        name=code_name,
        type=node_type.value,
        input_handles=input_handles,
        output_handles=[Handle(name='output_folder', description='The files generated by this code')],
        data=node_data,
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
                    content=f'add reusable code node:\n{node_formatter(new_node)}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


async def _store_generated_code(
    *,
    name: str,
    description: str,
    node_type: CodeNodeType,
    code: str,
    dependencies: list[str] | None = None,
) -> None:
    """Persist generated code and add its description to the shared vector index."""
    with Session(engine) as session:
        await code_catalog.create_code(
            session,
            CodeCreate(
                name=name,
                description=description,
                node_type=node_type,
                code=code,
                dependencies=dependencies or [],
            ),
        )


async def _add_new_code(
    code_type: Literal['bash', 'python'],
    request: str,
    pos_x: int,
    pos_y: int,
    input_edges: list[BuildEdge],
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    """Generate and add a CODE node in the requested language."""
    if code_type == 'bash':
        return await _add_generated_bash_code(request, pos_x, pos_y, input_edges, runtime)
    return await _add_generated_python_code(request, pos_x, pos_y, input_edges, runtime)


async def _add_generated_bash_code(
    request: str,
    pos_x: int,
    pos_y: int,
    input_edges: list[BuildEdge],
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    """Generate, persist, index, and add one Bash CODE node."""
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    input_handles = [Handle(name='input_files', description='The files required by this code')]

    if error := check_code_node_edges(input_edges, exist_nodes):
        return Command(update={'messages': [ToolMessage(content=error, tool_call_id=runtime.tool_call_id)]})
    if error := check_edge_handles(input_edges, input_handles, exist_nodes):
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

    if generated.reject:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=(
                            'This request is not the responsibility of the Bash code node. '
                            'Read the relevant documentation and use a Python code node or tool node to implement the requirements.'
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    try:
        await _store_generated_code(
            name=generated.name,
            description=request,
            node_type=CodeNodeType.BASH,
            code=generated.code,
        )
    except Exception as error:
        logger.exception('Failed to persist generated Bash code node')
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'Error: Bash code was generated but could not be saved and indexed: {error}',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    node_id = generate_char_index(len(exist_nodes) + 1)
    new_node = BuildNode(
        id=node_id,
        name='Bash Code',
        type=CodeNodeType.BASH.value,
        input_handles=input_handles,
        output_handles=[Handle(name='output_folder', description='The files generated by this code')],
        data={
            'code': generated.code,
            'description': request,
        },
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


async def _add_generated_python_code(
    request: str,
    pos_x: int,
    pos_y: int,
    input_edges: list[BuildEdge],
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    """Generate, persist, index, and add one Python CODE node."""
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    input_handles = [Handle(name='input_files', description='The files required by this Python code')]

    if error := check_code_node_edges(input_edges, exist_nodes):
        return Command(update={'messages': [ToolMessage(content=error, tool_call_id=runtime.tool_call_id)]})
    if error := check_edge_handles(input_edges, input_handles, exist_nodes):
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

    try:
        await _store_generated_code(
            name=generated.name,
            description=request,
            node_type=CodeNodeType.PYTHON,
            code=generated.code,
            dependencies=generated.dependencies,
        )
    except Exception as error:
        logger.exception('Failed to persist generated Python code node')
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'Error: Python code was generated but could not be saved and indexed: {error}',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    node_id = generate_char_index(len(exist_nodes) + 1)
    new_node = BuildNode(
        id=node_id,
        name='Python Code',
        type=CodeNodeType.PYTHON.value,
        input_handles=input_handles,
        output_handles=[Handle(name='output_folder', description='The files generated by this Python code')],
        data={
            'code': generated.code,
            'description': request,
            'dependencies': generated.dependencies,
        },
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
    """Add one note node."""
    node_id = generate_char_index(len(runtime.state.get('exist_nodes', [])) + 1)
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
    """Add one downstream-summary node."""
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
    used_files = [f'- {node_name_map[edge.source_node]}: {edge.source_handle}' for edge in input_edges]
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
