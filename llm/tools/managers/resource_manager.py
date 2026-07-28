import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

import app.crud.resource.bio_db as db_crud
from app.core.session import engine
from app.crud import project_file_mapping as project_file_mapping_crud
from app.crud.resource import sample as sample_crud
from app.services.resource import bio_db as db_service
from llm.context import ProjectContext
from llm.states import BuildEdge, BuildNode
from llm.utils.formatters import format_marker_registry, format_validation_error, manager_help_command, node_formatter
from llm.utils.node_schema import RESOURCES
from llm.utils.uid_mapper import generate_char_index
from llm.workflow_builder.drafts import snapshot_active_workflow
from llm.workflow_builder.validation import check_edge_handles, validate_node_addition


class ViewResourceSchema(BaseModel):
    """View details of a specific resource node type."""

    resource_type: str = Field(description='The name of the resource node type to view details for')


class SearchDbSchema(BaseModel):
    """Search for biological databases by keyword."""

    query: str = Field(
        description='Keyword for search. Can be the tool that uses this database (e.g., Kraken2), or the name of this database (e.g., nr or eggnog).'
    )


class CreateDbSchema(BaseModel):
    """Add a new biological database to be downloaded."""

    name: str = Field(description='The name of the database to be added.')
    description: str = Field(
        description='A functional description of the database, including its application scenarios and the tools that may be used with it.'
    )
    query: str = Field(
        description='Specific requirement descriptions, the task of creating database is assigned to a sub-agent. You need to provide a detailed description of your usage scenario and the expected type of database.'
    )


class AddResourceSchema(BaseModel):
    """Add a resource node to the workflow."""

    resource_type: str = Field(description='Resource node type')
    input_edges: list[BuildEdge] = Field(default_factory=list, description='Input edge connection list for this node')
    arguments: Optional[dict[str, Any]] = Field(
        default=None,
        description='Parameters of the resource node. Not all resource nodes need parameters; depends on user instructions.',
    )


VALID_OPERATES: dict[str, type[BaseModel] | None] = {
    'view': ViewResourceSchema,
    'view_marks': None,
    'search_db': SearchDbSchema,
    'create_db': CreateDbSchema,
    'add': AddResourceSchema,
}

HELP_MESSAGE = (
    'Specify the action to inspect, for example: resource_node_manager(action="help", params={"action": "view"}).'
)


@tool(parse_docstring=True)
async def resource_node_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[ProjectContext]) -> Command:
    """Resource node manager, execute operations related to resource nodes.

    Resources include files, databases, variables, and so on.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    match action:
        case 'help':
            # pyrefly: ignore [bad-argument-type]
            return manager_help_command(params, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)

        case 'view':
            try:
                validate_param = ViewResourceSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('view', e, ViewResourceSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _view_resource(validate_param.resource_type, runtime)

        case 'view_marks':
            return _view_marks(runtime)

        case 'search_db':
            try:
                validate_param = SearchDbSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('search_db', e, SearchDbSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _search_db(validate_param.query, runtime)

        case 'create_db':
            try:
                validate_param = CreateDbSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('create_db', e, CreateDbSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _create_db(validate_param.name, validate_param.description, validate_param.query, runtime)

        case 'add':
            try:
                validate_param = AddResourceSchema.model_validate(params)
            except ValidationError as e:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=format_validation_error('add', e, AddResourceSchema),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
            return _add_resource(
                validate_param.resource_type,
                validate_param.input_edges,
                runtime,
                validate_param.arguments,
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


def _view_resource(resource_type: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    if resource_type == 'genome':
        content = (
            'The system currently includes the following nodes for reference genomes (available resource_type):\n'
            '- `ncbi_genome`: Loads reference genomes and annotation files for any species from NCBI RefSeq based on species name or ID.\n'
            '- `GRCh38`: Loads the human reference genome released by GENCODE.\n'
            '- `GRCm39`: Loads the mouse reference genome released by GENCODE.\n\n'
            'In addition to providing genome and annotation files, all these nodes also offer indexes for common alignment tools. '
            'Generally, you can use them directly rather than manually building indexes. '
            'Refer to the respective nodes for more details.'
        )
        return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})

    if resource_type not in RESOURCES:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'Error: Unknown resource type: {resource_type}.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    resource_schema = RESOURCES[resource_type]
    node_obj = {
        'name': resource_type,
        'type': resource_type,
        'description': resource_schema.get('description', ''),
        'input_handles': [{'name': h.name, 'description': h.description} for h in resource_schema['input_handles']],
        'output_handles': [{'name': h.name, 'description': h.description} for h in resource_schema['output_handles']],
    }
    return Command(update={'messages': [ToolMessage(content=json.dumps(node_obj), tool_call_id=runtime.tool_call_id)]})


def _view_marks(
    runtime: ToolRuntime[ProjectContext],
) -> Command:
    """Read persisted markers and markers planned by every builder draft."""
    with Session(engine) as session:
        project_marks = project_file_mapping_crud.get_all_by_project(
            session,
            runtime.context.project_id,
        )
        sample_marks = sample_crud.get_files_by_project(
            session,
            runtime.context.project_id,
        )

    drafts = snapshot_active_workflow(runtime.state)
    content = format_marker_registry(project_marks, sample_marks, list(drafts.values()))
    return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})


def _search_db(query: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    with Session(engine) as session:
        exist_dbs = db_crud.get_by_name_fuzzy(session, query, 10, offset=0)

    if exist_dbs:
        db_blocks = [
            (
                f'<db id="{db.id}">\n'
                f'  <name>{db.name}</name>\n'
                f'  <description>{db.description}</description>\n'
                f'  <path>{db.path}</path>\n'
                f'</db>'
            )
            if db.path
            else (
                f'<db id="{db.id}">\n'
                f'  <name>{db.name}</name>\n'
                f'  <description>{db.description}</description>\n'
                f'  <download_command>{db.download_command}</download_command>\n'
                f'</db>'
            )
            for db in exist_dbs
        ]
        content = f'Based on the keywords, the following db nodes are available:\n\n{"\n".join(db_blocks)}'
    else:
        content = (
            'Based on the keywords you provided, no results found. Use another keyword to search.'
            'Or consider use `create_db` action '
            'explaining that a database is needed here and how to download it.'
        )

    return Command(update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})


async def _create_db(name: str, description: str, query: str, runtime: ToolRuntime[ProjectContext]) -> Command:
    from llm.agents.workflow_agent.bio_db_creater import bio_db_creator

    try:
        response = await bio_db_creator.ainvoke(
            {
                'messages': [
                    HumanMessage(
                        content=f'Create a new database with:\n'
                        f'name: {name}\n'
                        f'description: {description}\n'
                        f'User query: {query}'
                    )
                ]
            }
        )
        content = response['messages'][-1].content
        new_db = response.get('new_db', None)
    except Exception as exc:
        new_db = None
        content = f'Failed to create biological database "{name}": {exc}'

    update: dict[str, Any] = {
        'messages': [
            ToolMessage(
                content=content,
                tool_call_id=runtime.tool_call_id,
            )
        ],
    }
    if new_db:
        update['new_dbs'] = [new_db]

    return Command(update=update)


def _add_resource(
    resource_type: str,
    input_edges: list[BuildEdge],
    runtime: ToolRuntime[ProjectContext],
    arguments: Optional[dict[str, Any]] = None,
) -> Command:
    exist_nodes: list[BuildNode] = runtime.state.get('exist_nodes', [])
    node_id = generate_char_index(len(exist_nodes) + 1)

    if resource_type not in RESOURCES:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'Error: Unknown resource type: {resource_type}.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    resource_schema = RESOURCES[resource_type]

    if arguments is None:
        arguments = {}

    try:
        resource_schema['args'].model_validate(arguments)
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
                        content=f"Error: Invalid arguments for resource '{resource_type}':\n"
                        + '\n'.join(error_details),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    if error := validate_node_addition(
        runtime.state,
        node_type=f'resource_{resource_type}',
        arguments=arguments,
    ):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})

    if error := check_edge_handles(input_edges, resource_schema['input_handles'], exist_nodes):
        return Command(update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]})

    # Call custom edge_check if defined in schema
    if edge_check := resource_schema.get('edge_check'):  # noqa: SIM102
        if error := edge_check(input_edges, resource_schema['input_handles']):
            return Command(
                update={'messages': [ToolMessage(content=f'Error: {error}', tool_call_id=runtime.tool_call_id)]}
            )

    node_data = {}
    match resource_type:
        case 'value_string':
            new_node = BuildNode(
                id=node_id,
                name=resource_schema['name'],
                type='value_string',
                input_handles=[],
                output_handles=resource_schema['output_handles'],
                data=arguments,
            )
            return Command(
                update={
                    'exist_nodes': [new_node],
                    'messages': [
                        ToolMessage(
                            content=f'add new resource node:\n{node_formatter(new_node)}',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        case 'genome':
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content='You should use a specific reference genome node type rather than "genome" as the value for `resource_type`.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )
        case 'ncbi_genome' | 'GRCh38' | 'GRCm39':
            node_data = {'required_index': []}
        case 'db':
            db_id = arguments.get('db_id', -1)
            try:
                with Session(engine) as session:
                    bio_db = db_service.get_bio_db(session, int(db_id))
                    node_data = {**arguments, 'db_name': bio_db.name}
            except Exception as exc:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                content=f'Error: Invalid biological database id "{db_id}": {exc}',
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
        case _:
            node_data = arguments

    new_node = BuildNode(
        id=node_id,
        name=resource_schema['name'],
        type=f'resource_{resource_type}',
        input_handles=resource_schema['input_handles'],
        output_handles=resource_schema['output_handles'],
        data=node_data,
    )

    for edge in input_edges:
        edge.target_node = node_id

    return Command(
        update={
            'exist_nodes': [new_node],
            'exist_edges': input_edges,
            'messages': [
                ToolMessage(
                    content=f'add new resource node:\n{node_formatter(new_node)}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
