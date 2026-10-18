from typing import Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT, hook_config
from langgraph.runtime import Runtime
from langgraph.typing import ContextT


class EndLoopSchema(AgentState):
    should_end: NotRequired[bool]


class EndLoopMiddleware(AgentMiddleware):
    state_schema = EndLoopSchema

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        return {'should_end': False}

    @hook_config(can_jump_to=['end'])
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        if state.get('should_end', False):
            return {'jump_to': 'end'}

        return None

    @hook_config(can_jump_to=['end'])
    async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        return self.before_model(state, runtime)
