from pathlib import Path
from typing import Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.tools import ImageConfig
from app.utils.container.podman import create_container, remove_container, run_podman, run_podman_by_id


class ContainerMount(BaseModel):
    target: str = Field(description='Target path in container, a suitably named subdirectory under /data/')
    source: str = Field(description='Source path on host')
    read_only: bool


@tool(parse_docstring=True)
def build_micromamba_container(mounts: list[ContainerMount], runtime: ToolRuntime):
    """Create a new container using the micromamba image

    Args:
        mounts: Mounted volumes other than /data/testdata and /data/output, such as databases, etc.
    """
    image = ImageConfig.from_str('ghcr.io/mamba-org/micromamba:2.3.3-ubuntu22.04')
    container_id = _build_container(image, mounts)

    return Command(
        update={
            'mamba_container': container_id,
            'messages': [
                ToolMessage(
                    content='The micromamba container has been created. To run commands within this container, use the `use_container` tool to activate the corresponding container.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
def build_tool_container(mounts: list[ContainerMount], runtime: ToolRuntime):
    """Create a new container using tool image

    Args:
        mounts: Other mounts besides testdata and output folder
    """
    image = runtime.state.get('tool_image')
    container_id = _build_container(image, mounts)

    return Command(
        update={
            'tool_container': container_id,
            'messages': [
                ToolMessage(
                    content='The tool container has been created. To run commands within this container, use the `use_container` tool to activate the corresponding container.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool('run_command', parse_docstring=True)
def run_command_in_container(command: str, runtime: ToolRuntime, workdir: Optional[str] = None):
    """Run command, and return it`s output result.

    This tool must be called sequentially; parallel calls within a single response are prohibited.

    Args:
        command: The command you need to run
        workdir: Command execution working directory. Only pass this parameter when you need to enter a specific directory to execute a command.
    """
    if 'container_id' in runtime.state:
        container_id = runtime.state.get('container_id')
    else:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='You need to create and activate the container first before you can run commands within it.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    runtime.stream_writer({'run': command, 'dir': workdir})
    code, out = run_podman_by_id(container_id, command, workdir)

    run_status = 'successfully' if code == 0 else 'failed'
    return_msg = (
        f'Command ran {run_status}.\n\n......\n{_truncate_output(command, out)}'
        if out
        else f'Command ran {run_status}.'
    )

    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=return_msg,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool('remove_container', parse_docstring=True)
def stop_and_rm_container(remove_type: Literal['micromamba', 'tool', 'all'], runtime: ToolRuntime):
    """Stop and remove container

    Args:
        remove_type: Which container to remove. If the value is "all", it means delete all containers.
    """

    rm_ids = []
    rm_names = []
    update_dict = {}

    if remove_type in ('all', 'tool') and 'tool_container' in runtime.state:
        container_id = runtime.state.get('tool_container')
        rm_ids.append(container_id)
        rm_names.append('tool container')
        update_dict['tool_container'] = None

    if remove_type in ('all', 'micromamba') and 'mamba_container' in runtime.state:
        container_id = runtime.state.get('mamba_container')
        rm_ids.append(container_id)
        rm_names.append('micromamba container')
        update_dict['mamba_container'] = None

    for container_id in rm_ids:
        remove_container(container_id)

    # 构建返回消息
    if len(rm_names) == 1:
        message = f'The {rm_names[0]} has been deleted.'
    else:
        message = f'All containers ({", ".join(rm_names)}) have been deleted.'

    return Command(
        update={
            **update_dict,
            'messages': [
                ToolMessage(
                    content=message,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
def run_command(command: str, runtime: ToolRuntime):
    """run command in a container and return the result

    Args:
        command: the command you need to run
    """
    image = runtime.state.get('image')
    code, result = run_podman(image, command, [])

    return Command(
        update={'last_output': result, 'messages': [ToolMessage(content=result, tool_call_id=runtime.tool_call_id)]}
    )


def _build_container(image: ImageConfig, extra_mount: list[ContainerMount]) -> str:
    podman_mounts = [
        {'type': 'bind', 'target': m.target, 'source': m.source, 'read_only': m.read_only, 'relabel': 'z'}
        for m in extra_mount
    ]
    output_path = Path(get_settings().persistent.STORAGE_PATH).absolute() / 'testrun'
    output_path.mkdir(parents=True, exist_ok=True)
    podman_mounts.extend(
        [
            {
                'type': 'bind',
                'target': '/data/testdata',
                'source': get_settings().persistent.TEST_DATA_PATH,
                'read_only': True,
                'relabel': 'z',
            },
            {
                'type': 'bind',
                'target': '/data/output',
                'source': f'{output_path}',
                'read_only': False,
                'relabel': 'z',
            },
        ]
    )
    container_id = create_container(image, podman_mounts)

    return container_id


def _truncate_output(command: str, output: str) -> str:
    if any(k in command for k in ('ls', 'cat', 'head')) or len(output) <= 3000:
        return output
    return f'{output[:500]}\n\n{output[-500:]}'
