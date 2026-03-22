import inspect
import json
from typing import Any, get_args

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, ValidationError

from app.schemas.tools import Handle, ToolNodePublic
from app.utils.input_format import build_command_template
from llm.states import BuildNode, GlobalMarker, ToolGeneratorState
from llm.utils.uid_mapper import BidirectionalMapper


def _extract_nested_model(annotation: Any) -> type[BaseModel] | None:
    """Extract a BaseModel subclass from an annotation, handling generics like list[X] or Optional[X]."""
    if annotation is None:
        return None
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        result = _extract_nested_model(arg)
        if result is not None:
            return result
    return None


def schema_to_description(schema_cls: type[BaseModel], indent: int = 0, _visited: set | None = None) -> str:
    """Generate a human-readable description of a Pydantic model's fields, recursively expanding nested models."""
    if _visited is None:
        _visited = set()
    if schema_cls in _visited:
        return ''
    _visited.add(schema_cls)

    pad = '  ' * indent
    lines = [f'{pad}Schema fields ({schema_cls.__name__}):']
    for field_name, field_info in schema_cls.model_fields.items():
        annotation = field_info.annotation
        desc = field_info.description or ''
        default = '' if field_info.is_required() else f' (optional, default: {field_info.default})'
        lines.append(f'{pad}  - {field_name} ({annotation}){default}: {desc}')
        nested = _extract_nested_model(annotation)
        if nested is not None:
            lines.append(schema_to_description(nested, indent + 2, _visited))
    return '\n'.join(lines)


def format_validation_error(action: str, e: ValidationError, schema_cls: type[BaseModel]) -> str:
    """Format a ValidationError into a structured, LLM-friendly message."""
    error_lines = []
    for err in e.errors():
        field = '.'.join(str(loc) for loc in err['loc'])
        msg = err['msg']
        actual = err.get('input', '<missing>')
        error_lines.append(f'  - {field}: {msg} (got: {actual!r})')
    errors_str = '\n'.join(error_lines)
    return f'Invalid params for "{action}":\n{errors_str}\n\n{schema_to_description(schema_cls)}'


def manager_help_command(
    params: dict[str, Any],
    help_cls: type[BaseModel],
    valid_operates: dict[str, type[BaseModel]],
    help_msg: str,
    tool_call_id: str,
) -> Command:
    """Automatically generate help information from Pydantic data classes."""
    if params:
        try:
            validate_param = help_cls.model_validate(params)
        except ValidationError as e:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=format_validation_error('help', e, help_cls),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        action_name = validate_param.action
        if action_name not in valid_operates:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'Unknown action: "{action_name}". Available: {list(valid_operates.keys())}',
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        schema_cls = valid_operates[action_name]
        content = (
            f'Action "{action_name}" takes no parameters.' if schema_cls is None else schema_to_description(schema_cls)
        )
        return Command(update={'messages': [ToolMessage(content=content, tool_call_id=tool_call_id)]})

    return Command(update={'messages': [ToolMessage(content=help_msg, tool_call_id=tool_call_id)]})


def tool_config_formatter(state: ToolGeneratorState) -> str:
    """返回新生成的工具信息."""
    complete_command = build_command_template(
        command_template=state['command_template'],
        dynamic_params=state['dynamic_params'],
        immutable_static_params=state['immutable_static_params'],
        modifiable_static_params=state['modifiable_static_params'],
        file_mounts=state['file_mounts'],
    )

    input_files = '\n'.join([f'- `{f.name}`: {f.description}' for f in state['file_mounts'] if f.file_type == 'INPUT'])
    output_files = '\n'.join(
        [f'- {f.name}: {f.file_path}. {f.description}' for f in state['file_mounts'] if f.file_type == 'OUTPUT']
    )

    return (
        f'## Info\n'
        f'name: {state["name"]}\n'
        f'description: {state["description"]}\n\n'
        f'## Command\n'
        f'```bash\n'
        f'{complete_command}\n'
        f'```\n\n'
        f'## Input Files\n'
        f'{input_files}\n\n'
        f'## Output Files\n'
        f'{output_files}'
    )


def handle_formatter(handles: list[Handle], space: int = 6) -> str:
    return '\n'.join(
        [
            f'{" " * space}<handle>\n'
            f'{" " * space}  <name>{h.name}</name>\n'
            f'{" " * space}  <description>{h.description}</description>\n'
            f'{" " * space}</handle>'
            for h in handles
        ]
    )


def node_formatter(node: BuildNode) -> str:
    """将节点详细信息以JSON字符串形式返回."""
    data_obj: dict[str, Any] = {
        'id': node.id,
        'name': node.name,
        'type': node.type,
    }
    if node.type == 'tool':
        data_obj['data'] = {
            'command': node.command,
            'tool_uid': node.data.get('tool_uid'),
            'args_editable': node.data.get('args'),
        }
    else:
        data_obj['data'] = node.data

    data_obj['input_handles'] = [{'name': h.name, 'description': h.description} for h in node.input_handles]
    data_obj['output_handles'] = [{'name': h.name, 'description': h.description} for h in node.output_handles]
    data_obj['pos'] = {'x': node.x, 'y': node.y}

    return json.dumps(data_obj)


def search_nodes_formatter(nodes: list[ToolNodePublic], mappers: BidirectionalMapper) -> str:
    """将数据库中搜索的节点信息以JSON字符串形式返回."""
    tool_blocks = []
    for t in nodes:
        try:
            tool_blocks.append(
                {
                    'id': mappers.get_index_by_uid(str(t.uid)),
                    'name': t.name,
                    'description': t.description,
                    'input_handles': [{'name': h.name, 'description': h.description} for h in t.input_handles],
                    'output_handles': [{'name': h.name, 'description': h.description} for h in t.output_handles],
                    'data': {
                        'command': t.complete_command,
                        'args_editable': t.modifiable_static_params,
                        'args_static': t.immutable_static_params,
                    },
                }
            )
        except TypeError:
            logger.warning(f'Failed to build block for tool "{t.name}"({t.uid})')

    return json.dumps(tool_blocks)


def global_mark_formatter(markers: list[GlobalMarker]) -> str:
    if not markers:
        return 'No Global Marks In Workflow.'
    return '\n'.join(
        [
            f'<GlobalMark>\n'
            f'  <mark_name>{m.mark_name}</mark_name>\n'
            f'  <description>{m.description}</description>\n'
            f'</GlobalMark>'
            for m in markers
        ]
    )
