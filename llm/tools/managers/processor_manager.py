import json
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from llm.states import BuildContext, BuildEdge, BuildNode
from llm.utils.formatters import format_validation_error, manager_help_command, node_formatter
from llm.utils.node_schema import PROCESSORS
from llm.utils.retrievers import retrieve_processor
from llm.utils.uid_mapper import generate_char_index

# ---- Schemas ----


class HelpSchema(BaseModel):
    action: str = Field(description='The action name to get detailed schema for')


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

HELP_MESSAGE = """## Actions
processor_node_manager 中包括了以下操作:
- `view_processor`: 查看特定处理器节点的详细信息(输入输出,参数格式等)
- `search_processor`: 通过向量搜索按功能查找处理器节点
- `add_processor`: 向面板上添加一个处理器节点

在使用对应操作前, 务必使用 action="help" 并传入 params={"action": "<action_name>"} 以获取特定操作的参数说明
(e.g. processor_node_manager(action="help" params={"action": "view_processor"}))

## 说明

"""


@tool(parse_docstring=True)
async def processor_node_manager(action: str, params: dict[str, Any], runtime: ToolRuntime[BuildContext]) -> Command:
    """Processor node manager, execute operations related to processor nodes.

    Args:
        action: Action to be performed
        params: Parameters used in the action
    """
    match action:
        case 'help':
            logger.info(f'Processor Manager(action="{action} {params}")')
            return manager_help_command(params, HelpSchema, VALID_OPERATES, HELP_MESSAGE, runtime.tool_call_id)

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
            logger.info(f'Processor Manager(action="{action}" processor_name="{validate_param.processor_name}")')
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
            logger.info(f'Processor Manager(action="{action}" query="{validate_param.query}")')
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
            logger.info(f'Processor Manager(action="{action}" type="{validate_param.processor_name}")')
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
                            content=f'Unknown action: "{action}". Use action="help" with params={{}} to see available actions.',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )


def _view_processor(node_name: str, runtime: ToolRuntime[BuildContext]) -> Command:
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


async def _search_processor(query: str, runtime: ToolRuntime[BuildContext]) -> Command:
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
    runtime: ToolRuntime[BuildContext],
) -> Command:
    node_count = len(runtime.state.get('exist_nodes', []))
    node_id = generate_char_index(node_count + 1)

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
