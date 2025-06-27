from typing import Optional, Annotated

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ArgsSchema, ToolException, InjectedToolArg
from pydantic import BaseModel, Field

from app.models.tools import ImageConfig
from app.utils.container.podman import run_podman, run_podman_by_id


class RunDockerInput(BaseModel):
    command: str = Field(description='the command you need to run')
    image_config: Annotated[ImageConfig, InjectedToolArg] = Field(..., description='docker image config')


class RunContainer(BaseTool):
    name: str = 'run_command'
    description: str = 'run command in a container and return the result'
    args_schema: Optional[ArgsSchema] = RunDockerInput
    return_direct: bool = False

    def _run(
        self, command: str, image_config: ImageConfig, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        code, result = run_podman(image_config, command, [])

        if code != 0:
            raise ToolException(result)

        return result


class RunCommandInContainerInput(BaseModel):
    command: str = Field(description='the command you need to run')
    container_id: Annotated[str, InjectedToolArg] = Field(..., description='container management')


class RunCommandInContainer(BaseTool):
    name: str = 'run_command_in_container'
    description: str = 'run command, and return it`s output result'
    args_schema: Optional[ArgsSchema] = RunCommandInContainerInput
    return_direct: bool = False

    def _run(self, command: str, container_id: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        code, out = run_podman_by_id(container_id, command)

        if code != 0:
            raise ToolException(out)
        return out
