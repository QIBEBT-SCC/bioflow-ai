from typing import Optional, Annotated

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ArgsSchema, ToolException, InjectedToolArg
from pydantic import BaseModel, Field

from app.models.tools import ImageConfig
from app.utils.container.podman import run_podman


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
