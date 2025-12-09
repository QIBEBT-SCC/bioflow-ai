from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langgraph.runtime import Runtime
from loguru import logger

from app.core.config import get_settings


class ClearContainerMiddleware(AgentMiddleware):
    def after_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        if 'container_id' in state and state['container_id']:
            if get_settings().docker.CONTAINER == 'docker':
                from app.utils.container.docker import remove_container

                remove_container(state['container_id'])
            else:
                from app.utils.container.podman import remove_container

                remove_container(state['container_id'])

            logger.debug(f'Remove {get_settings().docker.CONTAINER} containers')

    async def aafter_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
