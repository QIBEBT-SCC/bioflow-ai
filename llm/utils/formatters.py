import json
from typing import Any

from loguru import logger

from app.schemas.tools import Handle, ToolNodePublic
from app.utils.input_format import build_command_template
from llm.states import BuildNode, GlobalMarker, ToolGeneratorState
from llm.utils.uid_mapper import BidirectionalMapper


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
        data_obj['command'] = node.command
        data_obj['tool_uid'] = node.data.get('tool_uid')
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
