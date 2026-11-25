from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langgraph.runtime import Runtime
from loguru import logger

from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.containers import remove_container


class ClearContainerMiddleware(AgentMiddleware[AgentState[Any], Any]):
    def after_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        """Remove any leftover container associated with the agent state."""
        container_id = state.get('container_id')
        if isinstance(container_id, str) and container_id:
            remove_container(container_id)
            logger.debug(f'Remove {get_settings().docker.CONTAINER} containers')

    async def aafter_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        """Asynchronously remove any leftover container associated with the agent state."""
        return self.after_agent(state, runtime)
