from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT, hook_config
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from bioflow_ai.utils.llm.artifacts import END_LOOP_ARTIFACT_KEY, trailing_tool_artifacts


class EndLoopMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Stop the agent loop when a trailing tool call sets the end-loop artifact."""

    @hook_config(can_jump_to=['end'])
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        if any(
            artifact.get(END_LOOP_ARTIFACT_KEY) is True
            for artifact in trailing_tool_artifacts(state.get('messages', ()))
        ):
            return {'jump_to': 'end'}

        return None

    @hook_config(can_jump_to=['end'])
    async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        return self.before_model(state, runtime)
