from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.utils.container.podman import create_container, remove_container, run_podman, run_podman_by_id


@tool
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


class ContainerMount(BaseModel):
    target: str = Field(description='Target path in container, a suitably named subdirectory under /data/')
    source: str = Field(description='Source path on host')
    read_only: str


@tool
def build_container(mounts: list[ContainerMount], runtime: ToolRuntime):
    """Create a container for the test run

    Args:
        mounts: Other mounts besides testdata
    """
    image = runtime.state.get('image')
    podman_mounts = [
        {'type': 'bind', 'target': m.target, 'source': m.source, 'read_only': m.read_only, 'relabel': 'z'}
        for m in mounts
    ]
    podman_mounts.append(
        {
            'type': 'bind',
            'target': '/data/testdata',
            'source': get_settings().persistent.TEST_DATA_PATH,
            'read_only': True,
            'relabel': 'z',
        }
    )
    container_id = create_container(image, podman_mounts)

    return Command(
        update={
            'container_id': container_id,
            'messages': [
                ToolMessage(
                    content=f'New container "{container_id}" created, commands can now be run.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def run_command_in_container(command: str, runtime: ToolRuntime):
    """Run command, and return it`s output result

    Args:
        command: The command you need to run
    """
    container_id = runtime.state.get('container_id')
    code, out = run_podman_by_id(container_id, command)

    run_status = 'successfully' if code == 0 else 'failed'
    if len(out.splitlines()) > 20:
        return Command(
            update={
                'last_output': out,
                'messages': [
                    ToolMessage(
                        content=f'Command ran {run_status}. A total of {len(out.splitlines())} lines of output have been cached. You can view the results of specified lines using the `view_output` tool.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    return out


@tool
def stop_and_rm_container(runtime: ToolRuntime):
    """Stop and remove container"""
    container_id = runtime.state.get('container_id')
    remove_container(container_id)
    return 'The container has been deleted'
