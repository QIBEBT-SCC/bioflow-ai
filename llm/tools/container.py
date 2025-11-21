from pathlib import Path
from typing import Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.tools import ImageConfig
from app.utils.container.podman import create_container, remove_container, run_podman, run_podman_by_id
from llm.states import TestRunContext


class ContainerMount(BaseModel):
    target: str = Field(description='Target path in container, a suitably named subdirectory under /data/')
    source: str = Field(description='Source path on host')
    read_only: bool


@tool(parse_docstring=True)
def build_container(mounts: list[ContainerMount], runtime: ToolRuntime[TestRunContext]):
    """Create a new container using tool image

    Args:
        mounts: Other mounts besides testdata and output folder
    """
    image: ImageConfig = runtime.state.get('image')
    container_id = _build_container(image, runtime.context.folder_name, mounts)

    return Command(
        update={
            'container_id': container_id,
            'messages': [
                ToolMessage(
                    content=f'The container based on image `{image.get_image()}` has been created. Now you can use `run_command` to run command in it.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool('run_command', parse_docstring=True)
def run_command_in_container(command: str, runtime: ToolRuntime, workdir: Optional[str] = None):
    """Run command, and return it`s output result.
    **This tool prohibits parallel calls! You are only allowed to run one command at a time!**

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
    if out.strip():
        return_msg = (
            _truncate_output(command, out)
            if code == 0
            else f'Command ran {run_status}.\n\n......\n{_truncate_output(command, out)}'
        )
    else:
        return_msg = f'Command ran {run_status}. No other output.'

    image: ImageConfig = runtime.state.get('image')
    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=f'[Container: {image.get_image()}]\n{return_msg}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool('remove_container')
def stop_and_rm_container(runtime: ToolRuntime):
    """Stop and remove container"""

    container_id = runtime.state.get('container_id')
    remove_container(container_id)

    return {
        'container_id': None,
        'messages': [
            ToolMessage(
                content='The container has been removed',
                tool_call_id=runtime.tool_call_id,
            )
        ],
    }


@tool(parse_docstring=True)
def run_command(command: str, runtime: ToolRuntime):
    """run command in a container and return the result
    **This tool prohibits parallel calls! You are only allowed to run one command at a time!**

    Args:
        command: the command you need to run
    """
    image = runtime.state.get('image')
    code, result = run_podman(image, command, [])

    return Command(
        update={'last_output': result, 'messages': [ToolMessage(content=result, tool_call_id=runtime.tool_call_id)]}
    )


def _build_container(image: ImageConfig, folder_name: str, extra_mount: list[ContainerMount]) -> str:
    podman_mounts = [
        {'type': 'bind', 'target': m.target, 'source': m.source, 'read_only': m.read_only, 'relabel': 'z'}
        for m in extra_mount
    ]
    output_path = Path(get_settings().persistent.STORAGE_PATH).absolute() / 'testrun' / folder_name
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
            },
        ]
    )
    container_id = create_container(image, podman_mounts)

    return container_id


def _truncate_output(command: str, output: str) -> str:
    if any(k in command for k in ('ls', 'head', 'find', 'grep')) or len(output) <= 3000:
        return output
    return f'{output[:1000]}\n...\n{output[-1000:]}'
