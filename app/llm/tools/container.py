from typing import Optional, Annotated

import docker
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ArgsSchema, ToolException, InjectedToolArg
from pydantic import BaseModel, Field

from app.models.tools import ImageConfig
from app.utils.container import check_image, get_current_user_identity


class DockerToolState(BaseModel):
    docker_repo: str
    docker_tag: str
    registry_host: str = 'docker.io'


class RunDockerInput(BaseModel):
    command: str = Field(description='the command you need to run')
    state: Annotated[DockerToolState, InjectedToolArg] = Field(..., description='docker config')


class RunDocker(BaseTool):
    name: str = 'run_command'
    description: str = 'run command in a docker container and return the result'
    args_schema: Optional[ArgsSchema] = RunDockerInput
    return_direct: bool = False

    def _run(
        self, command: str, state: DockerToolState, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        client = docker.from_env()
        check_image(
            client, ImageConfig(repository=state.docker_repo, tag=state.docker_tag), registry_host=state.registry_host
        )

        container = client.containers.run(
            f'{state.docker_repo}:{state.docker_tag}',
            f'"{command}"',
            entrypoint=['/bin/bash', '-c'],
            user=get_current_user_identity(),
            remove=False,
            detach=True,
        )
        exit_status = container.wait()['StatusCode']
        result = container.logs(stderr=True)
        container.remove()

        if exit_status != 0:
            raise ToolException(result.decode('utf-8'))

        return result.decode('utf-8')
