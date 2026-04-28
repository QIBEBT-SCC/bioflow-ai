from decimal import Decimal
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

from llm.states import UserContext
from llm.utils.token import store_token_usage


class TokenStatisticMiddleware(AgentMiddleware):
    def __init__(
        self,
        agent_name: str,
        model_id: int,
        setting_key: str,
        input_price: Decimal,
        output_price: Decimal,
        cache_read_price: Decimal,
    ):
        self.agent_name = agent_name
        self.model_id = model_id
        self.setting_key = setting_key
        self.input_price = input_price
        self.output_price = output_price
        self.cache_read_price = cache_read_price

    def after_model(self, state: AgentState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        token_usage = getattr(state['messages'][-1], 'usage_metadata', None)
        thread_id = get_config()['configurable'].get('thread_id', None)
        store_token_usage(
            self.agent_name,
            self.model_id,
            self.setting_key,
            runtime.context.user_id,
            thread_id,
            token_usage,
            self.input_price,
            self.output_price,
            self.cache_read_price,
        )
