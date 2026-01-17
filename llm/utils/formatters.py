from app.schemas.tools import Handle
from app.utils.input_format import build_command_template
from llm.states import BuildEdge, BuildNode, GlobalMarker, ToolGeneratorState


def tool_config_formatter(state: ToolGeneratorState) -> str:
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
        f'{complete_command.replace("{", "{{").replace("}", "}}")}\n'
        f'```\n\n'
        f'## Input Files\n'
        f'{input_files.replace("{", "{{").replace("}", "}}")}\n\n'
        f'## Output Files\n'
        f'{output_files.replace("{", "{{").replace("}", "}}")}'
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
