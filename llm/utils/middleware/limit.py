from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from loguru import logger

from llm.states import TestRunState


class DisableParallelToolCallsMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        request.model_settings['parallel_tool_calls'] = False
        return handler(request)

    async def awrap_model_call(self, request, handler):
        request.model_settings['parallel_tool_calls'] = False
        return await handler(request)


class TokenLimitMiddleware(AgentMiddleware):
    def __init__(self, max_token: int = 12000):
        self.max_token = max_token

    def before_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        return {'checked': False}

    def before_model(self, state: TestRunState, runtime: Runtime) -> dict[str, Any] | None:
        total_tokens = count_tokens_approximately(state['messages'])
        if not state['checked']:
            if total_tokens > self.max_token:
                logger.warning(f'Up to max token {total_tokens}')
                is_approved = interrupt({'max_token': total_tokens})

                return {'checked': is_approved}

        return None
