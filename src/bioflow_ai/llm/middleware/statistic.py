from decimal import Decimal
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import UsageMetadata
from langgraph.config import get_config
from langgraph.runtime import Runtime

from bioflow_ai.llm.context import BaseContext
from bioflow_ai.utils.observability.token_usage import store_token_usage, token_cost

MODEL_USAGE_KEY = 'model_usage'


class TokenStatisticMiddleware[ContextT: BaseContext](AgentMiddleware[AgentState[Any], ContextT]):
    def __init__(
        self,
        agent_name: str,
        model_id: int,
        setting_key: str,
        input_price: Decimal,
        output_price: Decimal,
        cache_read_price: Decimal,
        model_name: str,
        context_window: int | None,
    ):
        self.agent_name = agent_name
        self.model_id = model_id
        self.setting_key = setting_key
        self.input_price = input_price
        self.output_price = output_price
        self.cache_read_price = cache_read_price
        self.model_name = model_name
        self.context_window = context_window

    def after_model(self, state: AgentState, runtime: Runtime[BaseContext]) -> dict[str, Any] | None:  # noqa: D102
        token_usage = getattr(state['messages'][-1], 'usage_metadata', None)
        thread_id = get_config()['configurable'].get('thread_id', None)
        store_token_usage(
            self.agent_name,
            runtime.context.user_id,
            thread_id,
            setting_key=self.setting_key,
            model_id=self.model_id,
            input_price=self.input_price,
            output_price=self.output_price,
            cache_read_price=self.cache_read_price,
            token_usage=token_usage,
        )
        if token_usage:
            runtime.stream_writer({MODEL_USAGE_KEY: self.usage_payload(token_usage)})

    def usage_payload(self, token_usage: UsageMetadata) -> dict[str, Any]:
        """Describe one model call for live context and cost displays."""
        input_details = token_usage.get('input_token_details') or {}
        output_details = token_usage.get('output_token_details') or {}
        cost = token_cost(
            token_usage,
            input_price=self.input_price,
            output_price=self.output_price,
            cache_read_price=self.cache_read_price,
        )
        return {
            'agent': self.agent_name,
            'model': self.model_name,
            'context_window': self.context_window,
            'input_tokens': token_usage.get('input_tokens', 0),
            'output_tokens': token_usage.get('output_tokens', 0),
            'cache_read': input_details.get('cache_read', 0),
            'cache_creation': input_details.get('cache_creation', 0),
            'reasoning_tokens': output_details.get('reasoning', 0),
            'cost': float(cost),
        }
