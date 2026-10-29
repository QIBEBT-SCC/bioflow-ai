"""Shared text, JSON, schema, and workflow presentation helpers."""

from __future__ import annotations

import inspect
import json
import types
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from bioflow_ai.utils.jsonable import dumps_jsonable

if TYPE_CHECKING:
    from langgraph.types import Command

    from bioflow_ai.llm.states import BuildNode, WorkflowDraft
    from bioflow_ai.models.project import ProjectFileMapping
    from bioflow_ai.models.resource import SampleFiles
    from bioflow_ai.schemas.tools import Handle, ToolNodePublic
    from bioflow_ai.utils.uid_mapper import BidirectionalMapper

DEFAULT_FORMAT_MAX_CHARS = 50_000


def truncate_text(content: Any, limit: int, *, notice: str = '...') -> str:
    """Convert content to text and truncate it to a bounded preview."""
    text = content if isinstance(content, str) else str(content)
    if len(text) <= limit:
        return text
    return f'{text[:limit]}{notice}'


def format_text(content: Any, *, max_chars: int = DEFAULT_FORMAT_MAX_CHARS) -> str:
    """Format text with an explicit notice when content is omitted."""
    text = content if isinstance(content, str) else str(content)
    omitted = max(0, len(text) - max_chars)
    return truncate_text(text, max_chars, notice=f'\n... [truncated {omitted} characters]')


def format_json(
    value: Any,
    *,
    max_chars: int | None = DEFAULT_FORMAT_MAX_CHARS,
    indent: int | None = 2,
) -> str:
    """Format a safe JSON value, optionally bounding the result."""
    formatted = dumps_jsonable(value, indent=indent)
    if max_chars is None:
        return formatted
    return format_text(formatted, max_chars=max_chars)


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


def _format_annotation(annotation: Any) -> str:
    """Format a Python annotation as a concise readable type."""
    if annotation is Any:
        return 'any'
    if annotation is None or annotation is type(None):
        return 'null'
    if inspect.isclass(annotation):
        if issubclass(annotation, BaseModel):
            return 'object'
        return annotation.__name__

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return ' | '.join(json.dumps(arg, ensure_ascii=False) for arg in args)
    if origin in {Union, types.UnionType}:
        return ' | '.join(_format_annotation(arg) for arg in args)
    if origin is list:
        item_type = _format_annotation(args[0]) if args else 'any'
        return f'list[{item_type}]'
    if origin is dict:
        key_type = args[0] if args else Any
        value_type = args[1] if len(args) > 1 else Any
        return f'dict[{_format_annotation(key_type)}, {_format_annotation(value_type)}]'
    if origin is tuple:
        item_types = ', '.join('...' if arg is Ellipsis else _format_annotation(arg) for arg in args)
        return f'tuple[{item_types}]'
    if origin is set:
        item_type = _format_annotation(args[0]) if args else 'any'
        return f'set[{item_type}]'

    return str(annotation).removeprefix('typing.')


def _format_default(field_info: Any) -> str:
    """Format a field default without exposing Pydantic implementation details."""
    default_factory = field_info.default_factory
    if default_factory is not None:
        if default_factory is dict:
            return '{}'
        if default_factory in {list, set, tuple}:
            return '[]'
        return '<generated>'
    return json.dumps(field_info.default, ensure_ascii=False, default=str)


def schema_to_description(schema_cls: type[BaseModel], indent: int = 0, _visited: set | None = None) -> str:
    """Generate concise usage documentation from a Pydantic model."""
    if _visited is None:
        _visited = set()
    if schema_cls in _visited:
        return ''
    _visited.add(schema_cls)

    pad = '  ' * indent
    lines = []
    model_description = inspect.cleandoc(schema_cls.__doc__) if schema_cls.__doc__ else ''
    if model_description:
        lines.extend(f'{pad}{line}' for line in model_description.splitlines())
        if indent == 0:
            lines.append('')
    lines.append(f'{pad}{"Parameters" if indent == 0 else "Fields"}:')

    for field_name, field_info in schema_cls.model_fields.items():
        annotation = field_info.annotation
        desc = field_info.description or ''
        type_name = _format_annotation(annotation)
        requirement = 'required' if field_info.is_required() else f'optional, default={_format_default(field_info)}'
        lines.append(f'{pad}  - {field_name} ({type_name}, {requirement}): {desc}')

        nested = _extract_nested_model(annotation)
        if nested is not None:
            nested_description = schema_to_description(nested, indent + 2, _visited)
            if nested_description:
                lines.append(nested_description)

    return '\n'.join(lines)


def format_validation_error(action: str, e: ValidationError, schema_cls: type[BaseModel]) -> str:
    """Format a ValidationError into a structured readable message."""
    error_lines = []
    for err in e.errors():
        field = '.'.join(str(loc) for loc in err['loc'])
        msg = err['msg']
        actual = err.get('input', '<missing>')
        error_lines.append(f'  - {field}: {msg} (got: {actual!r})')
    errors_str = '\n'.join(error_lines)
    return (
        f'## Validation errors\n'
        f'Invalid params for "{action}":\n'
        f'{errors_str}\n\n'
        f'## Correct usage\n'
        f'{schema_to_description(schema_cls)}'
    )


def format_code_inputs(inputs: list[dict[str, Any]]) -> str:
    """Describe bound input parameters and ForEach manifests in readable Markdown."""
    if not inputs:
        return 'No upstream input parameters are bound to this script.'
    parameters = [
        f'- `{item["parameter"]}`: {item["description"]} '
        f'(from `{item["source_node"]}`, output `{item["source_handle"]}`).'
        for item in inputs
    ]
    sections = ['This script expects the following input parameters:\n\n' + '\n'.join(parameters)]
    for item in inputs:
        if not (input_format := item.get('format_contract')):
            continue
        schema = input_format['schema']
        fields = ', '.join(
            f'`{field}` ({description.replace("<", "").replace(">", "")})'
            for field, description in schema['items'][0].items()
            if field != 'outputs'
        )
        note = (
            f'The input for `{item["parameter"]}` is a ForEach results directory containing '
            f'`{input_format["manifest_file"]}`. The manifest has version `{schema["version"]}` and an `items` list. '
            f'Each item contains {fields}, and `outputs`.'
        )
        ports = '; '.join(
            f'`{port["id"]}` ({port["name"]}): {port["description"]}' for port in input_format['output_ports']
        )
        if ports:
            note += f' Exported outputs: {ports}.'
        note += ' ' + input_format['path_rule']
        sections.append(note)
    return '\n\n'.join(sections)


class HelpSchema(BaseModel):
    action: str = Field(description='The action name to get detailed schema for')


def manager_help_command(
    params: dict[str, Any],
    valid_operates: dict[str, type[BaseModel] | None],
    help_msg: str,
    tool_call_id: str,
) -> Command:
    """Automatically generate help information from Pydantic data classes."""
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    if params:
        try:
            validate_param = HelpSchema.model_validate(params)
        except ValidationError as e:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=format_validation_error('help', e, HelpSchema),
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


def handle_formatter(handles: list[Handle], space: int = 6) -> str:
    """Format node handles as XML-like blocks."""
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
            'args_editable': node.data.get('modifiable_params'),
        }
    else:
        data_obj['data'] = node.data

    data_obj['input_handles'] = [{'name': h.name, 'description': h.description} for h in node.input_handles]
    data_obj['output_handles'] = [{'name': h.name, 'description': h.description} for h in node.output_handles]

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


def format_marker_registry(
    project_marks: list[ProjectFileMapping],
    sample_marks: list[tuple[SampleFiles, str]],
    workflow_drafts: list[WorkflowDraft],
) -> str:
    """List persisted resources and marks declared by the current builder drafts."""
    draft_markers = [
        (draft, node)
        for draft in workflow_drafts
        for node in draft.nodes
        if node.type in {'project_mark', 'sample_mark'}
    ]
    if not project_marks and not sample_marks and not draft_markers:
        return 'No project or sample markers are currently registered.'

    sample_groups: dict[tuple[bool, str], dict[str, Any]] = {}
    for mark, sample_name in sample_marks:
        group = sample_groups.setdefault(
            (mark.is_dynamic, mark.tag),
            {'sample_names': set(), 'descriptions': set()},
        )
        group['sample_names'].add(sample_name)
        if mark.description:
            group['descriptions'].add(mark.description)

    sections = [
        (False, '=== INITIAL RESOURCES ==='),
        (True, '=== PUBLISHED WORKFLOW MARKS ==='),
    ]
    output: list[str] = []
    for is_dynamic, title in sections:
        entries: list[str] = []
        for mark in project_marks:
            if mark.is_dynamic == is_dynamic:
                description = mark.description or '(no description)'
                entries.append(f'- `proj:{mark.keyword}` — {description}')

        for (dynamic, tag), group in sample_groups.items():
            if dynamic != is_dynamic:
                continue
            descriptions = '; '.join(sorted(group['descriptions'])) or '(no description)'
            sample_count = len(group['sample_names'])
            entries.append(f'- `sample:{tag}` — {descriptions} [{sample_count} sample(s)]')

        output.append(title)
        output.extend(entries or ['- (none)'])

    draft_entries = []
    for draft, node in draft_markers:
        prefix = 'proj' if node.type == 'project_mark' else 'sample'
        mark_name = str(node.data.get('mark_name', ''))
        description = str(node.data.get('description', '')) or '(no description)'
        draft_entries.append(
            f'- `{prefix}:{mark_name}` — {description} '
            f'[planned by {draft.draft_id} "{draft.name}", {draft.execution_scope}]'
        )

    output.append('=== DRAFT WORKFLOW MARKS (NOT PUBLISHED YET) ===')
    output.extend(draft_entries or ['- (none)'])

    return '\n'.join(output)
