from app.schemas.tools import Handle
from llm.states import BuildEdge, BuildNode


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
            f'    <tool_id>{node.data.resource_uid}</tool_id>\n'
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
        f'<node id="{node.id}">\n'
        f'  <name>{node.name}</name>\n'
        f'  <type>{node.type}</type>\n'
        f'  <pos x={node.x} y={node.y}/>\n'
        f'{input_handles}'
        f'{output_handles}'
        f'{node_data}'
        f'</node>'
    )


def simple_node_formatter(node: BuildNode) -> str:
    return f'<Node id="{node.id}" name="{node.name}" type="{node.type}" pos={{x:{node.x}, y:{node.y}}}/>'


def edge_formatter(edge: BuildEdge) -> str:
    return f'<Edge source_node="{edge.source_node}" source_handle="{edge.source_handle}" target_node="{edge.target_node}" target_handle="{edge.target_handle}"/>'
