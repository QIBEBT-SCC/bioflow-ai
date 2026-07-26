import json
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError
from sqlmodel import Session

from app.core.session import engine
from app.crud import project_file_mapping as project_file_mapping_crud
from app.crud.resource import sample as sample_crud
from llm.context import ProjectContext
from llm.states import BuildEdge, BuildNode
from llm.utils.formatters import format_validation_error, manager_help_command, node_formatter
from llm.utils.node_schema import PROCESSORS
from llm.utils.retrievers import retrieve_processor
from llm.utils.uid_mapper import generate_char_index
from llm.workflow_builder.validation import validate_node_addition, check_edge_handles


# ---- Schemas ----


class ViewProcessorSchema(BaseModel):
    """View details of a specific processor node type."""

    processor_name: str = Field(description='The name of the processor node type to view details for')


class SearchProcessorSchema(BaseModel):
    """Search for processor nodes by functionality."""

    query: str = Field(description='Briefly describe the processor or function to be searched for')


class AddProcessorSchema(BaseModel):
    """Add a processor node to the workflow."""

    processor_name: str = Field(description='Processor name')
    arguments: dict[str, Any] | None = Field(default=None, description='Parameters of the processor node')
    input_edges: list[BuildEdge] = Field(default_factory=list, description='Input edge connection list for this node')
    pos_x: int = Field(description="The x-coordinate of the node's layout on the panel, only positive integer")
    pos_y: int = Field(description="The y-coordinate of the node's layout on the panel")


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'search_processor': SearchProcessorSchema,
    'view_processor': ViewProcessorSchema,
    'add_processor': AddProcessorSchema,
}

HELP_MESSAGE = (
    'Call processor_node_manager with action="help" and params={"action": "<action_name>"} '
    'to obtain the parameter schema for a specific action.'
)


@tool(parse_docstring=True)
async def processor_node_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[ProjectContext]) -> Command:
    """Processor node manager, execute operations related to processor nodes.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    match action:
        case 'help':
            # pyrefly: ignore [bad-argument-type]
            return manager_help_command(params, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)

        case 'view_processor':
            try:
                validate_param = ViewProcessorSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('view_processor', e, ViewProcessorSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _view_processor(validate_param.processor_name, runtime)

        case 'search_processor':
            try:
                validate_param = SearchProcessorSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('search_processor', e, SearchProcessorSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return await _search_processor(validate_param.query, runtime)

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


def _view_processor(node_name: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    if node_name not in PROCESSORS:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'Error: Unknown processor type: {node_name}.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    processor_schema = PROCESSORS[node_name]
    node_obj = {
        'name': node_name,
        'type': node_name,
        'description': processor_schema.get('description', ''),
        'input_handles': [{'name': h.name, 'description': h.description} for h in processor_schema['input_handles']],
        'output_handles': [{'name': h.name, 'description': h.description} for h in processor_schema['output_handles']],
    }
    return Command(update={'messages': [ToolMessage(content=json.dumps(node_obj), tool_call_id=runtime.tool_call_id)]})


async def _search_processor(query: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    results = await retrieve_processor(query)

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

    processors_content = json.dumps([{'name': p.metadata['name'], 'description': p.page_content} for p in results])
    content = f'<Processors>\n{processors_content}\n</Processors>'
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

    if error := _validate_mark_storage_conflict(processor_name, arguments, runtime):
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


def _validate_mark_storage_conflict(
    processor_name: str,
    arguments: dict[str, Any],
    runtime: ToolRuntime[ProjectContext],
) -> str | None:
    """Reject mark names that are already occupied in the project file system."""
    if processor_name not in {'project_mark', 'sample_mark'}:
        return None

    mark_name = str(arguments['mark_name'])
    with Session(engine) as session:
        if processor_name == 'project_mark':
            mapping = project_file_mapping_crud.get_by_keyword(
                session,
                runtime.context.project_id,
                mark_name,
            )
            if mapping is None:
                return None
            source = 'dynamic mark' if mapping.is_dynamic else 'initial project resource'
            return f'Cannot add mark `{mark_name}` because `proj:{mark_name}` already exists as a {source}.'

        matches = sample_crud.get_files_by_tag_in_project(
            session,
            runtime.context.project_id,
            mark_name,
        )
        if not matches:
            return None
        sample_names = ', '.join(sample_name for _, sample_name in matches)
        return (
            f'Cannot add mark `{mark_name}` because `sample:{mark_name}` already exists '
            f'on project sample(s): {sample_names}.'
        )
