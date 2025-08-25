from llm.states import BuildNode
from app.schemas.tools import Handle


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
    if node.type == 'tool':
        node_data = (
            f'  <data>\n'
            f'    <tool_id>{node.data.tool_id}</tool_id>\n'
            f'    <args>{node.data.args}</args>\n'
            f'  </data>\n'
            f'  <command>{node.command}</command>\n'
        )
    else:
        node_data = f'  <data>\n    <args>{node.data.args}</args>\n  </data>\n'

    input_handles = (
        f'  <input_handles>\n{handle_formatter(node.input_handles, 4)}\n  </input_handles>\n'
        if node.input_handles
        else ''
    )
    output_handles = (
        f'  <output_handles>\n{handle_formatter(node.output_handles, 4)}\n  </output_handles>\n'
        if node.output_handles
        else ''
    )

    return (
        f'<node>\n'
        f'  <id>{node.id}</id>\n'
        f'  <type>{node.type}</type>\n'
        f'  <pos>(x:{node.x}, y:{node.y})</pos>\n'
        f'{input_handles}'
        f'{output_handles}'
        f'{node_data}'
        f'</node>'
    )
