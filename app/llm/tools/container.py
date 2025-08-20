from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.tools import ImageConfig
from app.utils.container.podman import run_podman, run_podman_by_id, create_container


@tool
def run_command(
    command: str,
    image: Annotated[ImageConfig, InjectedState('image')],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """run command in a container and return the result

    Args:
        command: the command you need to run
    """
    code, result = run_podman(image, command, [])

    return Command(update={'last_output': result, 'messages': [ToolMessage(content=result, tool_call_id=tool_call_id)]})


class ContainerMount(BaseModel):
    target: str = Field(description='Target path in container, a suitably named subdirectory under /data/')
    source: str = Field(description='Source path on host')
    read_only: str


@tool
def build_container(
    mounts: list[ContainerMount],
    image: Annotated[ImageConfig, InjectedState('image')],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Create a container for the test run

    Args:
        mounts: Other mounts besides testdata
    """
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
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def run_command_in_container(command: str, container_id: Annotated[str, InjectedState('container_id')]):
    """Run command, and return it`s output result

    Args:
        command: The command you need to run
    """
    code, out = run_podman_by_id(container_id, command)

    if code != 0:
        raise ToolException(out)
    return out
